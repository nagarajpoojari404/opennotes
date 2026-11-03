---
title: "Online softmax"
date: "2026-10-31"
---

### The Standard 3-Pass Softmax Problem

Given an input vector $x = [x_1, x_2, \dots, x_N]$, standard numerically stable softmax prevents exponential overflow by subtracting the global maximum:

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^N e^{x_j - m}}, \quad \text{where } m = \max_{j} x_j
$$

In conventional GPU implementations, this requires **three separate memory passes**:

1. **Pass 1:** Read all $x$ from memory to compute $m = \max(x)$.
2. **Pass 2:** Read all $x$ again, compute $e^{x_i - m}$, and accumulate the normalizer $d = \sum_{j=1}^N e^{x_j - m}$.
3. **Pass 3:** Read $x$ a third time to evaluate $\frac{e^{x_i - m}}{d}$.

In standard self-attention, $x$ represents the attention logits $S = Q K^T \in \mathbb{R}^{N \times N}$. Materializing and reading $S$ across multiple passes incurs massive memory traffic that bottlenecks execution on memory-bandwidth-bound GPUs.

### The Mathematical Insight: Dynamic Rescaling

Online softmax computes the running maximum and the running denominator simultaneously by dynamically rescaling previous accumulations whenever a new, larger maximum is observed.

Let $m_k$ be the running maximum and $d_k$ be the running sum of exponents after processing the first $k$ elements:

$$
m_k = \max(m_{k-1}, x_k) \quad \text{with } m_0 = -\infty
$$

$$
d_k = \sum_{j=1}^k e^{x_j - m_k}
$$

To update $d_k$ from $d_{k-1}$ without re-evaluating earlier elements:

$$
d_k = \left(\sum_{j=1}^{k-1} e^{x_j - m_k}\right) + e^{x_k - m_k}
$$

Since $e^{x_j - m_k} = e^{x_j - m_{k-1}} \cdot e^{m_{k-1} - m_k}$:

$$
d_k = d_{k-1} \cdot e^{m_{k-1} - m_k} + e^{x_k - m_k}
$$

The correction factor $e^{m_{k-1} - m_k}$ is always $\le 1$ because $m_k \ge m_{k-1}$, preserving strict numerical stability.

### Block-Wise Online Softmax with Output Accumulation

In FlashAttention, online softmax is extended from scalar elements to blocks of tokens and fused directly with the value matrix multiplication $O = \text{softmax}(S) V$.

For a row block $b$ with local scores $S^{(b)}$ and local maximum $m^{(b)} = \max(S^{(b)})$:

1. **Update Running Maximum:**
	$$
	m_{\text{new}} = \max(m_{\text{old}}, m^{(b)})
	$$
2. **Compute Rescaling Factor:**
	$$
	\alpha = e^{m_{\text{old}} - m_{\text{new}}}
	$$
3. **Compute Current Block Exponents:**
	$$
	P^{(b)} = \exp(S^{(b)} - m_{\text{new}})
	$$
4. **Rescale & Update Running Normalizer ($d$):**
	$$
	d_{\text{new}} = d_{\text{old}} \cdot \alpha + \sum \text{row}(P^{(b)})
	$$
5. **Rescale & Update Running Output ($O$):**
	$$
	O_{\text{new}} = O_{\text{old}} \cdot \alpha + P^{(b)} V^{(b)}
	$$

After iterating through all blocks along the key/value sequence length, the final attention output vector is normalized once:

$$
O_{\text{final}} = \frac{O_{\text{accumulated}}}{d_{\text{final}}}
$$

This allows the entire $Q K^T$ matrix to be generated, softmax-normalized, and multiplied with $V$ tile-by-tile inside fast SRAM without ever writing the $N \times N$ attention matrix to global memory.
