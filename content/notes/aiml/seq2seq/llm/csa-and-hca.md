---
title: "CSA and HCA"
date: "2026-10-27"
---

**Compressed Sparse Attention (CSA)** is an attention architecture introduced in DeepSeek-V4 designed to scale sequence context to massive lengths (1M+ tokens) while addressing both the **compute bottleneck** and the **KV-cache memory capacity bottleneck**.


While standard DeepSeek Sparse Attention (DSA) solves compute complexity by selecting a top-$k$ subset of past tokens, the KV cache still scales linearly with the sequence length ($O(L)$ cache capacity). CSA combines **sequence compression** with **dynamic sparse routing** to slash both memory footprint and FLOPs.

### Core Idea: Dual-Stage Efficiency

CSA operates through a two-step reduction:

1. **Token Pooling / KV Compression ($m:1$):** Adjacent tokens along the sequence dimension are compressed into compact, pooled KV blocks using learned weights.
	Outcome School
2. **Dynamic Sparse Selection ($\text{Top-}k$):** Instead of evaluating full attention over all compressed blocks, a lightweight **Lightning Indexer** scores the compressed entries and routes attention only to the top-$k$ most relevant blocks.
	Outcome School
```
Raw Tokens (L) 
    │
    ▼ [Learned Overlapping Pooling (m:1)]
Compressed KV Blocks (L / m)
    │
    ▼ [Lightning Indexer Scoring + Top-k Selection]
Sparse High-Precision Attention (k blocks)
```

### Mathematical Mechanism

#### 1\. Overlapping KV Compression

Rather than doing naive downsampling or non-overlapping striding (which causes sharp information loss at boundaries), CSA performs **overlapping sequence pooling**.

Hugging Face

For hidden representations $H \in \mathbb{R}^{L \times d}$, every group of $m$ tokens (typically $m = 4$) is represented by an expanded receptive window of $2m$ tokens:

- Two projection streams produce compression logits $Z^a, Z^b \in \mathbb{R}^{L \times 1}$.
- For block index $b \in \{0, 1, \dots, \frac{L}{m} - 1\}$, local softmax pooling weights $\pi_b \in \mathbb{R}^{2m}$ are formed across the local span:
	$$
	\pi_{b, i} = \operatorname{softmax}_{i \in [0, 2m-1]}\left( Z_{b \cdot m + i} \right)
	$$
- The compressed KV state $C_b^{\text{Comp}} \in \mathbb{R}^{c}$ for block $b$ is computed as:
	$$
	C_b^{\text{Comp}} = \sum_{i=0}^{2m-1} \pi_{b, i} \cdot \left( H_{b \cdot m + i} W^{\text{comp}} \right)
	$$

This compresses the stored KV cache along the sequence length by a factor of $m$.

Outcome School

#### 2\. The Lightning Indexer on Compressed Blocks

To decide which compressed blocks query $t$ should attend to, a lightweight indexer projects query $H_t$ and the compressed cache $C_b^{\text{Comp}}$ into low-dimensional index spaces ($d_{\text{idx}} \ll d$):

- **Query index projection:**
	$$
	q_{t, h}^I = H_t W_{q, h}^I \in \mathbb{R}^{d_{\text{idx}}}
	$$
- **Compressed key index projection:**
	$$
	K_b^I = C_b^{\text{Comp}} W_{k}^I \in \mathbb{R}^{d_{\text{idx}}}
	$$

The relevance score $I_{t, b}$ between query $t$ and compressed block $b$ is computed across $n_h^I$ index heads with a gated activation (e.g., ReLU) and learnable head weights $w_{t, h}^I$:

$$
I_{t, b} = \sum_{h=1}^{n_h^I} w_{t, h}^I \cdot \operatorname{ReLU}\left( q_{t, h}^I \cdot K_b^I \right)
$$

Because scoring operates over $\frac{L}{m}$ entries in reduced precision (such as FP8/FP4) and low rank, index evaluation requires negligible compute compared to standard attention.

#### 3\. Top-$k$ Sparse Attention

Given all past compressed block scores $\{I_{t, b}\}_{b \le \lfloor t/m \rfloor}$, the model selects only the top $k$ blocks:

$$
\mathcal{B}_t = \operatorname{argTopK}_{b \le \lfloor t/m \rfloor} \left( I_{t, b}, \, k \right), \quad \vert{}\mathcal{B}_t\vert{} = k
$$

High-precision multi-head attention is then computed **only** against the retrieved $k$ compressed entries:

$$
A_{t, b} = \frac{\exp\left( \frac{Q_t (K_b^{\text{comp}})^\top}{\sqrt{d_k}} \right)}{\sum_{j \in \mathcal{B}_t} \exp\left( \frac{Q_t (K_j^{\text{comp}})^\top}{\sqrt{d_k}} \right)}, \quad b \in \mathcal{B}_t
$$

$$
O_t = \sum_{b \in \mathcal{B}_t} A_{t, b} V_b^{\text{comp}}
$$

### Architectural Context: CSA vs. HCA

Outcome School

In architectures like DeepSeek-V4, CSA is typically interleaved with **Heavily Compressed Attention (HCA)**:

Outcome School

| Attention Type | Compression Ratio ($m$) | Selection Strategy | Attention Density | Purpose |
| --- | --- | --- | --- | --- |
| **CSA** | Mild ($m=4$) | Dynamic Top-$k$ via Indexer | **Sparse** | High-detail retrieval over relevant history. |
| **HCA** | Aggressive ($m'=128$) | None (No indexer) | **Dense** | Global background context with near-zero cache footprint. |

### Summary of Benefits

- **Reduced KV Footprint:** Compressing $m$ tokens into 1 entry shrinks the physical memory consumption of the KV cache by $m\times$.
	Outcome School
- **Sub-linear Scaling:** With $k$ fixed (e.g., $k \ll L/m$), the computational cost of the main attention core scales with $O(k \cdot d)$ instead of $O(L \cdot d)$.
- **No Boundary Artifacts:** The $2m$ overlapping softmax pooling prevents rigid chunking boundaries from destroying local syntactic and semantic cohesion.
