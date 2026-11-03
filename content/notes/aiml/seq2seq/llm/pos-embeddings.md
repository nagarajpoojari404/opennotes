---
title: "Pos Embeddings"
date: "2026-10-27"
---

Self-attention in Transformers is **permutation-equivariant**: without position information, the model treats a sequence as an unordered bag of tokens. Positional encodings break this symmetry, giving tokens an address or a distance metric.

Abhik Sarkar

Positional embeddings in LLMs fall into three broad architectural families: **Absolute**, **Relative**, and **Rotary / Bias-based** hybrids.

### 1\. Absolute Positional Embeddings (APE)

These assign a fixed or learned coordinate vector to each exact index ($0, 1, 2, \dots, N$) and add it directly to the token embedding before feeding it into the first transformer layer: $x_i = e_i + p_i$.

- **Sinusoidal Positional Encoding (Fixed APE):**
	- **Mechanism:** Introduced in the original Transformer (*Attention Is All You Need*). Uses interleaved geometric sine and cosine waves across embedding dimensions:
		$$
		PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
		$$
		- **Strengths:** Zero learned parameters; theoretically represents relative shifts via trigonometric identities.
		- **Limitations:** Added directly at the input, tangling positional information with semantic embeddings. Does not extrapolate reliably beyond the training sequence length.
		Wayland Zhang\+ 1
- **Learned Absolute Embeddings:**
	- **Mechanism:** Used in models like **BERT**, **GPT-2**, and **OPT**. Treats positions as a lookup table of trainable parameters matrix $W_{pos} \in \mathbb{R}^{L_{\max} \times d}$.
		- **Strengths:** Adapts directly to the pretraining corpus data distribution.
		- **Limitations:** Hard ceiling on sequence length ($L_{\max}$). It cannot process token $L_{\max} + 1$ without structural alterations or interpolation.

### 2\. Relative Positional Encodings (RPE)

Rather than caring where a token sits in the whole document, RPE cares about pairwise distance $(i - j)$ between the query token $i$ and key token $j$.

liveBook · Manning - Manning Publications

- **Shaw’s Relative Positional Encoding:**
	- **Mechanism:** Modifies the attention score and value aggregation by adding learned bias vectors corresponding to the clipped relative distance $\text{clip}(i - j, -k, k)$.
		- **Strengths:** Translates easily across sequence lengths; reflects the intuition that local syntactic context matters more than global indices.
		- **Limitations:** Adds memory and compute overhead when forming per-head attention matrices; tricky to cache efficiently for fast autoregressive generation.
- **T5 Relative Bias:**
	- **Mechanism:** Simplifies Shaw’s approach by directly adding a scalar bias $b_{i-j}$ to the attention logit:
		$$
		\text{Attention}_{ij} = \frac{q_i k_j^\top}{\sqrt{d}} + b_{i-j}
		$$
		Positions are bucketed logarithmically so that nearby tokens have fine-grained distance bins, while far tokens share coarse bins.
		- **Strengths:** Extremely lightweight scalar addition; scales to long contexts with few parameters.
- **Transformer-XL / DeBERTa Relative Disentangled Encoding:**
	- **Mechanism:** Decomposes attention into four disentangled terms: content-to-content, content-to-position, position-to-content, and position-to-position.
		- **Strengths:** Stronger representations for masked language modeling and natural language understanding.

### 3\. Modern Multiplicative & Attention-Bias Methods

Modern autoregressive LLMs avoid adding vectors at the bottom layer, injecting position information **directly into attention heads**.

- **RoPE (Rotary Position Embedding):**
	- **Mechanism:** Used by **LLaMA**, **Mistral**, **Qwen**, **Gemma**, and **DeepSeek**. Instead of adding vectors, it rotates 2D slices of query and key vectors by an angle proportional to the position index $m$:
		$$
		\mathbf{q}_m = \mathbf{R}_{\Theta, m} \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n} \mathbf{W}_k \mathbf{x}_n
		$$
		When computing $\mathbf{q}_m^\top \mathbf{k}_n$, the inner product naturally simplifies to an expression that depends purely on $(m - n)$.
		- **Strengths:** Preserves vector norms, integrates with KV-caching with zero overhead, and allows context expansion via frequency scaling (e.g., RoPE-NTK, YaRN).
- **ALiBi (Attention with Linear Biases):**
	- **Mechanism:** Used in **BLOOM** and **MPT**. Completely removes position embeddings. Instead, it adds a static, non-learned penalty proportional to distance directly to the attention logits:
		$$
		\text{Attention}_{i, j} = \frac{q_i k_j^\top}{\sqrt{d}} - m \cdot \vert{}i - j\vert{}
		$$
		where $m = 2^{-8h/H}$ is a fixed geometric slope unique to each attention head $h$.
		- **Strengths:** Zero learned position weights; achieves smooth zero-shot length extrapolation out of the box.
		- **Limitations:** Imposes a strict monotonic decay (local attention bias), which can hinder tasks needing long-range global reasoning.
		MetricGate
- **NoPE (No Positional Encoding):**
	- **Mechanism:** Eliminates positional encodings entirely.
		- **How it works:** In causal decoder-only models, the autoregressive lower-triangular causal mask and causal scratchpad provide an implicit directional order.
		- **Usage:** Primarily used in research settings; performs surprisingly well on specific algorithmic and recurrent tasks, though it generally lags behind RoPE on standard natural language benchmarks.

### Comparison Overview

| Method | Injection Point | Type | Extrapolation Ability | Dominant Use Cases |
| --- | --- | --- | --- | --- |
| **Learned APE** | Input layer (Additive) | Absolute | Fails past trained length | GPT-2, BERT, OPT |
| **Sinusoidal APE** | Input layer (Additive) | Absolute | Poor without scaling | Original Transformer |
| **T5 / Shaw RPE** | Attention logits (Additive) | Relative | Moderate | T5, Raffel et al. |
| **RoPE** | Attention $Q,K$ (Multiplicative) | Relative via Absolute | High (via NTK/YaRN) | LLaMA, Mistral, Qwen, DeepSeek |
| **ALiBi** | Attention logits (Linear penalty) | Relative | Very high (Zero-shot) | BLOOM, MPT |
