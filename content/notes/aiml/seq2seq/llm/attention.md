---
title: "Attention"
date: "2026-10-27"
---

Attention mechanisms can be organized into six structural paradigms: **classical scoring alignments**, **routing configurations**, **KV-cache parameter sharing**, **sparse & linear asymptotic reductions**, **hardware-aware tiling engines**, and **hierarchical multi-scale compressed attentions** (including DeepSeek's recent architectures).

### 1\. Mathematical Scoring Functions (Compatibility Metrics)

Given query $q \in \mathbb{R}^{d_k}$ and key $k \in \mathbb{R}^{d_k}$:

- **Additive / Concat Attention (Bahdanau et al.):**
	$$
	e_{ij} = v_a^T \tanh(W_q q_i + W_k k_j)
	$$
	Maps queries and keys into a shared latent space before projecting down to a scalar via $v_a$. High expressivity, but non-vectorized addition makes it memory-bound on tensor cores.
- **General / Bilinear Attention (Luong et al.):**
	$$
	e_{ij} = q_i^T W_a k_j
	$$
	Uses a learnable matrix $W_a \in \mathbb{R}^{d_q \times d_k}$ to model asymmetric cross-correlation between different latent dimensions.
- **Scaled Dot-Product Attention (Vaswani et al.):**
	$$
	\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
	$$
	The $\sqrt{d_k}$ divisor stabilizes variance at $\approx 1$, preventing the softmax distribution from saturating with near-zero gradients at large dimensions.
- **Cosine / Scaled-Cosine Attention:**
	$$
	e_{ij} = \frac{q_i \cdot k_j}{\Vert{}q_i\Vert{}_2 \Vert{}k_j\Vert{}_2 \cdot \tau}
	$$
	Normalizes vector norms and scales by learnable temperature $\tau$. Used in Swin Transformer v2 and vision transformers to prevent extreme logit drift.

### 2\. Contextual Routing & Information Flow

- **Self-Attention:** $Q, K, V$ are projected from the same hidden sequence $H \in \mathbb{R}^{N \times d}$. Computes intra-sequence token interaction.
- **Cross-Attention:** $Q$ originates from target sequence $H_{\text{dec}}$, while $K, V$ originate from source sequence $H_{\text{enc}}$:
	$$
	Q = H_{\text{dec}} W_Q, \quad K = H_{\text{enc}} W_K, \quad V = H_{\text{enc}} W_V
	$$
- **Causal Masking:** Imposes $M_{ij} = -\infty$ for $j > i$. Enforces autoregressive token dependency ($p(x_t \mid x_{\lt t})$).
- **Sliding Window / Local Banded Attention:**
	$$
	M_{ij} = \begin{cases} 0 & \vert{}i - j\vert{} \le w \\ -\infty & \text{otherwise} \end{cases}
	$$
	Restricts computation to a local neighborhood of radius $w$, dropping complexity from $O(N^2)$ to $O(N \cdot w)$.

### 3\. Head Topology & KV-Cache Footprint

During autoregressive decoding, storing Key-Value states ($2 \times L \times n_{\text{heads}} \times d_k \times \text{seq\_len}$) is the primary bottleneck for serving throughput.

| Architecture | Query Heads ($H_Q$) | KV Heads ($H_{KV}$) | Memory Per Token |
| --- | --- | --- | --- |
| **Multi-Head Attention (MHA)** | $H$ | $H$ | $2\cdot H\cdot d_k$ |
| **Multi-Query Attention (MQA)** | $H$ | $1$ | $2\cdot d_k$ (shared across all heads) |
| **Grouped-Query Attention (GQA)** | $H$ | $G$ ($1\lt G\lt H$) | $2\cdot G\cdot d_k$ (each KV head serves $H/G$ queries) |
| **Multi-Head Latent Attention (MLA - DeepSeek-V2/V3)** | $H$ | Compressed Latent | Low-rank projected footprint |

#### Multi-Head Latent Attention (MLA)

MLA compresses the KV cache into a low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ (where $d_c \ll d_h \cdot n_h$), eliminating the need to store full uncompressed key and value vectors:

$$
\mathbf{c}_t^{KV} = W^{DKV} h_t, \quad k_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad v_t^C = W^{UV} \mathbf{c}_t^{KV}
$$

Because Rotary Position Embedding (RoPE) cannot be applied directly to low-rank compressed keys without breaking matrix associativity during decoding, MLA uses a **decoupled RoPE strategy**:

$$
q_{t, i} = [W^{UQ} \mathbf{c}_t^Q; \text{RoPE}(W^{QR} \mathbf{c}_t^Q)], \quad k_{t, i} = [k_t^C; \text{RoPE}(W^{KR} h_t)]
$$

During inference, $W^{UK}$ can be absorbed directly into $W^{UQ}$, allowing the model to compute attention directly over the compressed cache $\mathbf{c}^{KV}$.

### 4\. DeepSeek & Frontier Multi-Scale / Sparse Attentions

To scale context windows to 1M+ tokens without unbounded compute or KV-cache explosion, recent architectures move toward multi-tiered memory systems.

```
Raw Token Stream (Length N)
   │
   ├── Recent Context (Sliding Window: raw uncompressed tokens)
   │
   ├── Medium-Range Context (CSA: Compressed 4:1 + Learned Top-k Selection via DSA)
   │
   └── Distant Context (HCA: Heavily Compressed 128:1 + Dense Attention)
```

#### A. DeepSeek Sparse Attention (DSA)
Instead of attending across all visible tokens or relying on fixed static heuristics, DSA employs an auxiliary **lightweight indexer** to dynamically select a sparse subset of token blocks:
1. A small projection scores relevance between query $q_t$ and cached block representations $b_j$:
	$$
	s_{tj} = \text{ReLU}(q_t W_{\text{idx}} b_j)
	$$
2. Only the Top-$k$ scoring blocks are retrieved:
	$$
	\mathcal{S}_t = \text{Top-}k(\{s_{tj}\}_{j=1}^B)
	$$
3. Full scaled dot-product attention executes solely over tokens in index set $\mathcal{S}_t$:
	$$
	\text{Output}_t = \text{Attention}\left(q_t, K_{[\mathcal{S}_t]}, V_{[\mathcal{S}_t]}\right)
	$$

#### B. Compressed Sparse Attention (CSA)
CSA compresses blocks of adjacent tokens into single compressed KV representations before sparse selection.


- Given block size $m$ (typically $m=4$), adjacent tokens are pooled via learned saliency weighting:
	Medium
	$$
	\tilde{K}_b = \sum_{j \in \text{block}_b} \alpha_j K_j, \quad \tilde{V}_b = \sum_{j \in \text{block}_b} \alpha_j V_j
	$$
- The model executes DSA-style sparse index selection over the compressed blocks $\tilde{K}, \tilde{V}$, reducing both KV memory footprint and token selection overhead.

#### C. Heavily Compressed Attention (HCA)
For distant context where fine-grained syntactic details blur into macroscopic semantics, HCA applies an aggressive compression factor $m' \gg m$ (e.g., $m' = 64$ or $128$):


$$
K^{\text{HCA}} = \text{Pool}_{m'}(K), \quad V^{\text{HCA}} = \text{Pool}_{m'}(V)
$$

Because a 1M-token sequence is compressed down to $\approx 7,800$ tokens, **dense** attention can be evaluated over the entire distant history without sparse top-$k$ filtering, maintaining global receptive field at minimal FLOP cost.


### 5\. Linear & Kernelized Attention Approximations ($O(N)$)

Standard attention scales quadratically ($O(N^2)$) due to the $N \times N$ matrix. Linear variants eliminate softmax to exploit associative multiplication.

- **Linearized Attention via Feature Maps (Katharopoulos et al.):** Using a non-negative feature map $\phi(x) = \text{elu}(x) + 1$:
	$$
	\text{Sim}(Q, K) = \phi(Q)\phi(K)^T
	$$
	By matrix associativity:
	$$
	\text{Attention}(Q,K,V) = \frac{\phi(Q)\left(\phi(K)^T V\right)}{\phi(Q)\left(\phi(K)^T \mathbf{1}_N\right)}
	$$
	Computing $(\phi(K)^T V) \in \mathbb{R}^{d \times d}$ first drops complexity from $O(N^2 d)$ to $O(N d^2)$.
- **Performer / FAVOR+ (Random Feature Attention):** Uses orthogonal random projections $\omega_i \sim \mathcal{N}(0, I)$ to approximate the exponential softmax kernel without non-negativity breakdown:
	$$
	\phi(x) = \frac{h(x)}{\sqrt{M}} \left[ \exp\left(\omega_1^T x - \frac{\Vert{}x\Vert{}^2}{2}\right), \dots, \exp\left(\omega_M^T x - \frac{\Vert{}x\Vert{}^2}{2}\right) \right]^T
	$$
- **Linformer (Low-Rank Projection):** Projects key and value sequence length dimensions from $N$ down to fixed dimension $k$ using linear matrices $E, F \in \mathbb{R}^{k \times N}$:
	$$
	\bar{K} = E K \in \mathbb{R}^{k \times d}, \quad \bar{V} = F V \in \mathbb{R}^{k \times d}
	$$
	$$
	\text{Attention}(Q, \bar{K}, \bar{V}) = \text{softmax}\left(\frac{Q \bar{K}^T}{\sqrt{d_k}}\right)\bar{V} \implies O(N \cdot k)
	$$

### 6\. Hardware-Aware Exact Attention (Tiling & IO-Aware)

These methods compute standard scaled dot-product attention without mathematical approximation, restructuring operations to optimize data movement between GPU SRAM (fast, small) and High Bandwidth Memory (HBM, slow, large).

- **FlashAttention (Online Softmax Tiling):** Standard softmax requires computing $\sum_j e^{s_j}$ across the entire row before dividing. Online softmax computes incremental normalizers over tiles $B_c$ and $B_r$:
	$$
	m_{\text{new}} = \max(m_{\text{old}}, \max(S_{\text{tile}})), \quad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum e^{S_{\text{tile}} - m_{\text{new}}}
	$$
	$$
	O_{\text{new}} = \text{diag}\left(e^{m_{\text{old}} - m_{\text{new}}}\right) O_{\text{old}} + e^{S_{\text{tile}} - m_{\text{new}}} V_{\text{tile}}
	$$
	Never writes the $N \times N$ attention matrix to HBM.
- **FlashDecoding & FlashDecoding++:** Splits the KV cache along the sequence dimension into independent chunks during autoregressive generation ($M=1, N=\text{large}$). Computes partial attention sums concurrently in SRAM across multiple streaming multiprocessors (SMs), followed by a fast final reduction.
- **PagedAttention (vLLM):** Treats the KV cache as virtual memory pages. Allocates fixed-sized memory blocks on demand, eliminating internal fragmentation and supporting zero-copy beam search / prefix caching.
