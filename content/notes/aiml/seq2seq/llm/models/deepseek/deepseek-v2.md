---
title: "Deepseek-v2"
date: "2026-10-31"
---

The paper **"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"** (DeepSeek-AI, May 2024) introduced a 236B-parameter Sparse Mixture-of-Experts (MoE) transformer that drastically lowered training costs and inference serving overhead. With **236B total parameters** and only **21B activated parameters** per token, DeepSeek-V2 achieved performance competitive with contemporary closed and open front-runners while cutting KV cache memory requirements by **93.3%** and boosting generation throughput by **5.76×**.


Its core architectural breakthroughs are **Multi-head Latent Attention (MLA)** and the **DeepSeekMoE** architecture.


### 1\. Multi-Head Latent Attention (MLA)

Standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) run into heavy memory bandwidth bottlenecks during autoregressive generation because they must cache uncompressed Key and Value states across all layers. GQA reduces cache size by forcing heads to share keys and values, which can compromise representational capacity.

Syntaxradar

DeepSeek-V2 resolved this by projecting Keys and Values into low-rank compressed latent spaces.

arXiv

#### A. Low-Rank Key-Value Joint Compression

Instead of caching $K$ and $V$ vectors for all heads, the model projects the hidden representation $h_t \in \mathbb{R}^d$ into a single compressed latent vector $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ (where $d_c \ll n_h d_h$):

$$
\mathbf{c}_t^{KV} = W_{DKV} h_t
$$

During generation, **only $\mathbf{c}_t^{KV}$ is retained in the KV cache**. When computing attention, the model up-projects the cached latent into the attention keys and values on the fly:

arXiv

$$
\mathbf{k}_t^C = W_{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W_{UV} \mathbf{c}_t^{KV}
$$

Because matrix multiplication is associative, the up-projection matrix $W_{UK}$ can be absorbed into the Query projection during inference, computing dot products directly against the low-rank latent vector to save FLOPs and memory access.

#### B. Decoupled Rotary Position Embedding (RoPE)

A major mathematical challenge with low-rank Key projection is that **Rotary Position Embeddings (RoPE)** are position-dependent and cannot be commutatively combined through a linear down-/up-projection matrix. DeepSeek solved this by **decoupling** content and position:

- **Key:** Concatenates an up-projected content key with an explicit, low-dimensional RoPE key carrying positional information:
	$$
	\mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C ; \text{RoPE}(\mathbf{k}_t^R)]
	$$
- **Query:** Similarly projected into compressed latent space, up-projected to content queries, and concatenated with a RoPE query:
	$$
	\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C ; \text{RoPE}(\mathbf{q}_{t, i}^R)]
	$$

The total cached footprint per token per layer drops from $2 n_h d_h$ down to just $(d_c + d_h^R)$, reducing KV cache memory by **93.3%** compared to standard MHA.

Syntaxradar

### 2\. DeepSeekMoE Architecture

Conventional MoEs (such as Mixtral 8x7B) split the MLP into a few large experts (e.g., 8 experts, choosing top-2). DeepSeek argued that this granularity is too coarse, limiting the combinatorial combinations of knowledge and causing individual experts to overlap in responsibility.

DeepSeek-V2 implements two foundational MoE principles:

```
┌───────── Input Token x ─────────┐
                  │                                 │
                  ▼                                 ▼
      ┌───────────────────────┐         ┌───────────────────────┐
      │     Shared Experts    │         │    Routed Experts     │
      │   (Always Activated)  │         │ (Fine-Grained Top-m)  │
      │   [Expert 1, 2, ...]  │         │ [Expert 1, 2, ..., N] │
      └───────────┬───────────┘         └───────────┬───────────┘
                  │                                 │
                  └───────────────┬─────────────────┘
                                  ▼
                            Fused Output
```
1. **Fine-Grained Expert Segmentation:** Instead of $N$ standard-sized experts, DeepSeek divides expert capacity into $mN$ smaller experts (each having $1/m$ the hidden dimension) and activates $mk$ of them. DeepSeek-V2 scales this to **160 routed experts**, selecting **6 routed experts** per token. This vastly expands the combinatorial flexibility of expert coordination without increasing active compute.
2. **Dedicated Shared Experts:** A subset of experts (**2 shared experts**) is permanently active for every token, bypassing the router.
	- **Why this matters:** In standard MoEs, generalist knowledge (syntax, standard grammar, formatting) ends up redundantly learned across many routed experts. Isolating shared parameters forces common representations into the shared experts, freeing routed experts to specialize strictly in distinct semantic domains.

The output for token $x$ is:

$$
\mathbf{y} = \sum_{i=1}^{N_s} \text{SharedExpert}_i(x) + \sum_{j \in \text{Top-}K} g_j(x) \cdot \text{RoutedExpert}_j(x)
$$

### 3\. Structural Specifications & Parameter Count

| Specification                               | Dimension / Value                |
| ------------------------------------------- | -------------------------------- |
| **Total Parameters**                        | 236B                             |
| **Active Parameters / Token**               | 21B                              |
| **Number of Layers**                        | 60                               |
| **Hidden Dimension ($d_{\text{model}}$)**      | 5120                             |
| **Attention Heads ($n_h$)**           | 128                              |
| **Head Dimension ($d_h$)**            | 128                              |
| **KV Compression Dimension ($d_c$)**  | 512                              |
| **Decoupled RoPE Dimension ($d_h^R$)** | 64                               |
| **Shared Experts**                          | 2                                |
| **Routed Experts**                          | 160 (Top-6 routed per token)     |
| **Context Window**                          | 128K tokens (via YaRN extension) |

### 4\. Load Balancing & Training Dynamics

To prevent routing collapse across 160 fine-grained experts, DeepSeek-V2 avoided device-level hard capacity factors (which drop tokens and hurt quality) and instead used a multi-tier loss formulation:

- **Expert-Level Balance Loss ($\mathcal{L}_{\text{exp}}$):** Penalizes routing probability skew across individual experts.
- **Device-Level Balance Loss ($\mathcal{L}_{\text{dev}}$):** Groups experts across physical GPUs/nodes and penalizes imbalanced aggregate computation across accelerator devices, minimizing inter-device pipeline stalls.
- **Communication Optimization:** By co-locating groups of routed experts and overlapping all-to-all communication with token computation kernels, the communication overhead typical of fine-grained MoE dispatch was mitigated on dual-ring cluster topologies.

### 5\. Architectural Significance & Legacy

- **Inference Economy:** Standard MoEs reduce training compute relative to parameter size, but during decoding they still suffer under full KV cache memory limits. MLA directly decoupled sequence memory from head width, enabling massive batch sizes and 128K context scaling on modest hardware clusters.
- **Foundation for V3 and R1:** The twin innovations of MLA and fine-grained shared/routed MoE established the blueprint carried forward into **DeepSeek-Coder-V2**, **DeepSeek-V3**, and the **DeepSeek-R1** reasoning series.
