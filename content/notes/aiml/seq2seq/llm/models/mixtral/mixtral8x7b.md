---
title: "Mixtral8x7B"
date: "2026-10-31"
---

The paper **"Mixtral of Experts"** (Jiang et al., Mistral AI, Jan 2024) introduced **Mixtral 8x7B**, a Sparse Mixture of Experts (SMoE) decoder-only transformer that redefined efficiency tradeoffs in open-weight models.

arXiv

### 1\. High-Level Architecture & Parameter Mechanics

Despite the "8x7B" naming, Mixtral is **not an ensemble of eight distinct 7B models**.

- **Shared Parameters:** The self-attention blocks, layer normalizations, and input/output embeddings are fully shared across the entire model.
- **Sparse Components:** Only the standard Feed-Forward Network (FFN / MLP) blocks inside each transformer layer are replaced with **8 parallel expert networks**.
	Open Laboratory
- **Parameter Breakdown:**
	- **Total Parameters:** ~**46.7B**.
		Open Laboratory
		- **Active Parameters per Token:** ~**12.9B**.
		Open Laboratory
		- During inference, it computes at the latency and FLOP cost of a ~13B dense model while matching or outperforming Llama 2 70B and GPT-3.5 across reasoning, code, and math.
		Mistral

#### Key Hyperparameters:

- **Layers ($L$):** 32
	Open Laboratory
- **Hidden Dimension ($d_{\text{model}}$):** 4096
	Open Laboratory
- **FFN Intermediate Dimension ($d_{\text{ff}}$):** 14,336
	Open Laboratory
- **Attention:** Grouped Query Attention (GQA) with 32 query heads and 8 key/value heads ($d_{\text{head}} = 128$)
	Open Laboratory
- **Context Length:** 32,768 tokens (using RoPE embeddings)
	Open Laboratory
- **FFN Activation:** SwiGLU
	Open Laboratory

### 2\. MoE Formulation & Routing Mechanism

For each token embedding $x \in \mathbb{R}^d$ entering the MoE block at layer $l$:

1. **Routing / Gating:** A trainable linear gate $W_g \in \mathbb{R}^{E \times d}$ (where number of experts $E = 8$) computes routing logits:
	$$
	H(x) = x \cdot W_g^T
	$$
2. **Top-2 Selection & Normalization:** Mixtral selects the top $k = 2$ highest-scoring experts. The gating distribution over the chosen subset $\text{Top-2}(H(x))$ is computed via a Softmax over only those top-2 logits:
	Open Laboratory
	$$
	P_i(x) = \begin{cases} \frac{\exp(H(x)_i)}{\sum_{j \in \text{Top-2}(H(x))} \exp(H(x)_j)} & \text{if } i \in \text{Top-2}(H(x)) \\ 0 & \text{otherwise} \end{cases}
	$$
3. **Additive Aggregation:** The output of the MoE block is the weighted sum of outputs from the two selected experts:
	$$
	y = \sum_{i=1}^{E} P_i(x) \cdot \text{Expert}_i(x)
	$$
	Each expert is an independent SwiGLU MLP:
	$$
	\text{Expert}_i(x) = \left(\text{Swish}(x W_{1,i}) \odot x W_{3,i}\right) W_{2,i}
	$$

### 3\. Load Balancing & Training Dynamics

MoE routing networks naturally suffer from a "winner-takes-all" failure mode, where a handful of experts receive all traffic while others starve, wasting parameter capacity.

- **Auxiliary Loss:** To prevent expert collapse and ensure load balancing across GPUs/experts, Mixtral employs an auxiliary routing loss:
	$$
	\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i
	$$
	where $f_i$ is the fraction of tokens in the batch dispatched to expert $i$, and $P_i$ is the average routing probability assigned to expert $i$.
- **Co-training:** Unlike methods that upcycle pre-trained dense checkpoints into MoEs, Mixtral trained the router and expert weights **jointly from scratch**.

### 4\. Empirical Routing Analysis: How Do Experts Specialize?

One of the most notable analyses in the paper is examining *what* the experts actually specialize in:

- **Syntactic vs. Semantic Specialization:**
	- In the **early and late layers**, token routing correlates heavily with syntactic/grammatical structure (e.g., specific punctuation, indentation, whitespace, common function words).
		Open Laboratory
		- In the **middle layers**, routing displays higher semantic divergence, but **not** a clean topic-by-topic division (i.e., there is no single "Python expert" or "history expert").
		Open Laboratory
- **Temporal Locality:** Tokens appearing close to each other in the sequence frequently route to the same expert due to contextual similarity in hidden states.
- **Multilingual Uniformity:** Experts do not cleanly segregate by language; representations are shared across languages for abstract reasoning.

### 5\. Systems & Inference Implementation

The primary challenge of sparse MoEs is hardware utilization:

- **Memory Footprint:** Running Mixtral requires keeping all ~47B weights in VRAM (requiring ~90–100 GB in FP16/BF16, or ~24–30 GB with 4-bit / AWQ / GPTQ quantization).
	Open Laboratory
- **Dynamic Batching & Kernel Efficiency:** In standard dense layers, matrix multiplications are contiguous. In MoEs, dynamic token-to-expert assignments lead to irregular memory layouts.
- **MegaBlocks Integration:** Mixtral's open deployment relied on integrating **MegaBlocks** block-sparse matrix multiplication kernels into serving frameworks (vLLM). This avoids padding overheads and efficiently executes dynamic top-2 routing without dropping tokens.
	arXiv

### 6\. Key Takeaways

1. **Compute vs. Memory Disconnect:** Mixtral proved that model scale can be decoupled from per-token compute: you pay the memory capacity price of a ~47B model to get the throughput and compute latency of a ~13B model.
	Open Laboratory
2. **Open-Weight Milestone:** It demonstrated that properly regularized, from-scratch SMoE scaling could match dense flagship baselines (Llama 2 70B) at a fraction of active inference FLOPs.
