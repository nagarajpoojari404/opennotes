---
title: "DSA"
date: "2026-10-27"
---

DeepSeek Sparse Attention (DSA) is a dynamic, content-dependent sparse attention mechanism designed to bypass the standard quadratic attention cost $O(L^2)$ for long sequences without relying on rigid, pre-defined static patterns (such as fixed sliding windows or static strided blocks).

Medium

Instead of attending to all $L$ tokens, DSA uses a **two-stage pipeline**:

1. A **Lightning Indexer** cheaply scans all past tokens to score their relevance for each query token.
	Emergent Mind
2. A **Top-$k$ Selector** gathers the highest-scoring $k$ key-value entries ($k \ll L$, typically $k \approx 2048$ out of $128\text{K}+$ context) to run the full-fidelity Multi-Head / Multi-head Latent Attention (MLA).

### The Lightning Indexer: Mathematical Formulation

In standard full attention, computing attention weights requires projecting hidden states into dimension $d_k$, performing an inner product $Q K^\top / \sqrt{d_k}$, and materializing an $L \times L$ attention matrix.

The **Lightning Indexer** avoids this bottleneck by performing scoring in a low-dimensional space using lightweight numerical formats (FP8).

Emergent Mind

#### 1\. Low-Rank Projection & Scoring

For a sequence with hidden states $H \in \mathbb{R}^{L \times d}$, the indexer maps the query at position $t$ and key at position $s \le t$ into $M$ small indexer heads of dimension $d_{\text{idx}} \ll d_k$:

$$
q_{t, m}^{\text{idx}} = H_t W_{q, m}^{\text{idx}} \in \mathbb{R}^{d_{\text{idx}}}, \quad k_{s, m}^{\text{idx}} = H_s W_{k, m}^{\text{idx}} \in \mathbb{R}^{d_{\text{idx}}}
$$

Where:

- $m \in \{1, \dots, M\}$ represents the indexer head index.
- $d_{\text{idx}}$ is small (often 16 to 32), keeping tensor operations minimal.
	Emergent Mind

The similarity score $I_{t, s}$ between query $t$ and historical token $s$ is computed by aggregating across the $M$ heads using learnable scalar weights $\alpha_m$ (or head projection weights):

Emergent Mind

$$
I_{t, s} = \sum_{m=1}^{M} \alpha_m \left( \frac{\langle q_{t, m}^{\text{idx}}, \, k_{s, m}^{\text{idx}} \rangle}{\sqrt{d_{\text{idx}}}} \right)
$$

Because $q^{\text{idx}}$ and $k^{\text{idx}}$ are low-dimensional and run in FP8 GEMM kernels, this scoring across all $L$ tokens takes negligible time compared to full-precision, high-dimensional multi-head attention.

Emergent Mind

### Top-$k$ Selection & Sparse Attention

#### 1\. Dynamic Sparsity Mask

For query position $t$, the top-$k$ most relevant token indices $\mathcal{S}_t$ are extracted using a partial top-$k$ operator:

Emergent Mind

$$
\mathcal{S}_t = \operatorname{argTopK}_{s \le t} \left( I_{t, s}, \, k \right), \quad \vert{}\mathcal{S}_t\vert{} = k
$$

#### 2\. Main High-Precision Attention

The expensive Multi-Head / MLA attention then computes queries, keys, and values *only* for the indices in $\mathcal{S}_t$:

$$
A_{t, s} = \begin{cases} \frac{\exp\left(\frac{Q_t K_s^\top}{\sqrt{d_k}}\right)}{\sum_{j \in \mathcal{S}_t} \exp\left(\frac{Q_t K_j^\top}{\sqrt{d_k}}\right)}, & s \in \mathcal{S}_t \\ 0, & s \notin \mathcal{S}_t \end{cases}
$$

$$
O_t = \sum_{s \in \mathcal{S}_t} A_{t, s} V_s
$$

### Training the Indexer: Distillation Alignment Loss

A major challenge with dynamic sparse selection is that the $\operatorname{argTopK}$ operator is non-differentiable. DeepSeek overcomes this via a **two-phase alignment strategy**:

1. **Dense Warm-up:** The base Transformer runs dense attention. The Lightning Indexer's prediction is treated as a probability distribution over the context window:
	$$
	P_{t, s}^{\text{idx}} = \operatorname{softmax}_s (I_{t, s}) = \frac{\exp(I_{t, s})}{\sum_{j \le t} \exp(I_{t, j})}
	$$
	The ground-truth target is derived from the full dense attention weight matrix, averaged across all attention heads $H$:
	$$
	\bar{A}_{t, s} = \frac{1}{H} \sum_{h=1}^H A_{t, s}^{(h)}
	$$
	The indexer is trained by minimizing the Kullback-Leibler (KL) divergence between $\bar{A}_t$ and $P_t^{\text{idx}}$:
	$$
	\mathcal{L}_{\text{align}} = \sum_{t=1}^L \mathcal{D}_{\text{KL}}\left(\bar{A}_t \parallel P_t^{\text{idx}}\right) = \sum_{t=1}^L \sum_{s \le t} \bar{A}_{t, s} \log \left(\frac{\bar{A}_{t, s}}{P_{t, s}^{\text{idx}}}\right)
	$$
2. **Sparse Continued Training:** Once the indexer reliably mimics the token preferences of the full attention heads, the model transitions to actual top-$k$ sparse routing, detaching the indexer graph to prevent gradient interference.
	Weile Luo

### Complexity Comparison

| Feature | Standard Dense Attention | DeepSeek Sparse Attention (DSA) |
| --- | --- | --- |
| **FLOPs (Main Attention)** | $O(L^2\cdot d)$ | $O(L\cdot k\cdot d)$ |
| **FLOPs (Indexer)** | None | $O(L^2\cdot d_{\text{idx}})$ (Low $d_{\text{idx}}$ in FP8) |
| **Total Memory Bound** | $O(L^2)$ | $O(L\cdot k)$ |
| **Token Selection** | All tokens | Dynamic, content-aware top-$k$ |
