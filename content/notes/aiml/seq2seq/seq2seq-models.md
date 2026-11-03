---
title: "Seq2seq models"
date: "2026-10-26"
---

Sequence-to-Sequence (Seq2Seq) models map an input sequence $(x_1, \dots, x_{T_x})$ to an output sequence $(y_1, \dots, y_{T_y})$, where $T_x \neq T_y$ in general. Architectures are organized below by their recurrent, convolutional, and self-attention foundations, alongside their key mathematical formulations.

## 1\. Recurrent Seq2Seq (RNN / LSTM / GRU)

Recurrent models process tokens sequentially, updating an internal hidden state at each timestep.

### A. Vanilla (Information-Bottleneck) Seq2Seq

*Sutskever et al. (2014) / Cho et al. (2014)*

- **Encoder:** Reads the source sequence and compresses the entire context into the final hidden state $h_{T_x}$:
	$$
	h_t = f(h_{t-1}, x_t)
	$$
	The context vector is static: $c = h_{T_x}$ (or $[h_{T_x}; c_{T_x}]$ for LSTMs).
- **Decoder:** Autoregressively produces target tokens conditioned on $c$:
	$$
	s_t = g(s_{t-1}, y_{t-1}, c)
	$$
	$$
	P(y_t \mid y_{\lt t}, X) = \text{softmax}(W_o [s_t; y_{t-1}; c] + b_o)
	$$
- **Limitation:** The single fixed-size vector $c \in \mathbb{R}^d$ causes an **information bottleneck**, leading to catastrophic forgetting on sequences where $T_x > 20$.

### B. Additive Attention Seq2Seq (Bahdanau Attention)

*Bahdanau et al. (2014)*

Resolves the bottleneck by computing a dynamic context vector $c_i$ for each decoder step $i$ using a bidirectional encoder: $h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$.

- **Alignment Score:** Parameterized by a small feed-forward network:
	$$
	e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)
	$$
- **Attention Weights:**
	$$
	\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
	$$
- **Dynamic Context & State Update:**
	$$
	c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j, \quad s_i = g(s_{i-1}, y_{i-1}, c_i)
	$$

### C. Multiplicative Attention Seq2Seq (Luong Attention)

*Luong et al. (2015)*

Computes attention from the *current* decoder state $s_i$ (post-RNN step) and introduces three scoring variants:

$$
\text{score}(s_i, h_j) = \begin{cases} s_i^\top h_j & \text{Dot} \\ s_i^\top W_a h_j & \text{General} \\ v_a^\top \tanh(W_a [s_i; h_j]) & \text{Concat} \end{cases}
$$

The context vector $c_i = \sum_j \alpha_{ij} h_j$ is combined with the current state via an attentional hidden layer before token projection:

$$
\tilde{s}_i = \tanh(W_c [c_i; s_i]), \quad P(y_i \mid y_{\lt i}) = \text{softmax}(W_s \tilde{s}_i)
$$

- **Sub-variant (Local Attention):** Instead of computing attention over all $T_x$ positions (Global), the model predicts an aligned center position $p_t$ and computes weights within a Gaussian window $[p_t - D, p_t + D]$:
	$$
	p_t = T_x \cdot \sigma(v_p^\top \tanh(W_p s_t)), \quad \alpha_{tj} = \alpha_{tj} \exp\left(-\frac{(j - p_t)^2}{2\sigma^2}\right)
	$$

## 2\. Convolutional Seq2Seq (ConvS2S)

*Gehring et al. (2017)*

Replaces sequential recurrence with 1D temporal convolutions, enabling parallel training over the entire sequence length.

- **Gated Linear Units (GLU):** Given convolution output $Y = [A \ B] \in \mathbb{R}^{2d}$ from filter kernel $W \in \mathbb{R}^{2d \times k d}$:
	$$
	v(Y) = A \otimes \sigma(B)
	$$
	where $\otimes$ is element-wise multiplication and $\sigma$ acts as an input-controlled gating mechanism.
- **Residual Connections:**
	$$
	h_l = v(W_l * h_{l-1} + b_l) + h_{l-1}
	$$
- **Causal Convolutions:** The decoder uses masked/causal padding to prevent receptive fields from accessing future tokens $y_{>t}$.
- **Attention Mechanism:** Multi-step attention computed at each decoder layer $l$:
	$$
	d_i^l = W_d^l h_{i, \text{dec}}^l + \hat{y}_{i-1}, \quad e_{ij}^l = (d_i^l)^\top (z_j^e + e_j), \quad c_i^l = \sum_j \alpha_{ij}^l (z_j^e + e_j)
	$$
	where $z_j^e$ is the encoder output and $e_j$ is the input positional embedding.

## 3\. Transformer-Based Seq2Seq

*Vaswani et al. (2017)*

Discards recurrence and convolutions entirely, modeling dependencies via scaled dot-product self-attention and cross-attention.

### Standard Transformer Architecture

- **Scaled Dot-Product Attention:**
	$$
	\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
	$$
	where $Q \in \mathbb{R}^{N \times d_k}$, $K \in \mathbb{R}^{M \times d_k}$, $V \in \mathbb{R}^{M \times d_v}$. The scale factor $\frac{1}{\sqrt{d_k}}$ prevents vanishing gradients in the softmax for large projection dimensions.
- **Multi-Head Attention (MHA):**
	$$
	\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
	$$
	$$
	\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
	$$
- **Masking ($M$):**
	- *Encoder:* $M_{ij} = -\infty$ for padded positions, $0$ otherwise.
		- *Decoder Self-Attention:* Causal mask where $M_{ij} = -\infty$ for $j > i$ to enforce autoregression.
- **Cross-Attention:** $Q = H_{\text{dec}} W_i^Q$, while $K = H_{\text{enc}} W_i^K$ and $V = H_{\text{enc}} W_i^V$.

## 4\. Architectural Variants of the Transformer

| Model Class | Configuration | Key Mathematical / Architectural Mechanism | Primary Use Case |
| --- | --- | --- | --- |
| **Standard Encoder-Decoder** (T5, BART) | Bidirectional Encoder + Autoregressive Decoder | Cross-attention integrates corrupted/prefix encoder states into decoder generation: $L=-\sum \log P_\theta(y_t \mid y_{\lt t}, X_{\text{corrupt}})$ | Translation, Abstractive Summarization |
| **Prefix / Causal Decoder-Only** (LLaMA, GPT-4) | Single transformer stack | Unified attention matrix using a block prefix mask: $M_{ij}=0$ for $i,j\le T_x$; $M_{ij}=-\infty$ for $j>i$ when $i>T_x$ | General sequence completion, Few-shot Seq2Seq |
| **Pointer-Generator Networks** | Hybrid Generation + Extractive Copying | Soft-switch generation probability: $p_{\text{gen}}=\sigma(w_h^\top c_t+w_s^\top s_t+w_x^\top x_t+b_{\text{ptr}})$; combines vocab and copy distributions | Out-of-vocabulary reproduction, strict factuality |
| **Non-Autoregressive (NAT)** | Parallel Decoder | Removes causal masking; generates all output tokens conditionally independent given source $X$: $P(Y \mid X)=\prod_{t=1}^{T_y}P(y_t \mid X)$ | Ultra-low latency generation |

### Mathematical Details of Notable Variants

#### Pointer-Generator Networks *(See et al., 2017)*

Addresses the fixed-vocabulary issue by allowing the model to either generate a token from vocabulary $V$ or copy a word directly from the source sequence via attention weights:

$$
P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{j: x_j = w} \alpha_{tj}
$$

To prevent repetitive generations, **coverage mechanisms** penalize repeatedly attending to the same input locations:

$$
c^t = \sum_{t'=0}^{t-1} \alpha^{t'}, \quad e_{tj} = v^\top \tanh(W_h h_j + W_s s_t + w_c c_j^t + b)
$$

$$
\mathcal{L}_{\text{cov}} = \sum_j \min(\alpha_{tj}, c_j^t)
$$

#### Non-Autoregressive Seq2Seq (NAT) *(Gu et al., 2018)*

To bypass $O(T_y)$ sequential generation steps, NAT predicts target length with a fertility model $\phi$ and decodes in parallel:

$$
P(Y \mid X) = \sum_{f_1, \dots, f_{T_x}} \prod_{j=1}^{T_x} p(f_j \mid x_{1:T_x}) \prod_{t=1}^{T_y} p(y_t \mid x_1, \dots, x_{T_x}, f_1, \dots, f_{T_x})
$$

Because conditional independence causes the **multimodality problem** (inability to coordinate parallel outputs), modern variants use iterative refinement, Connectionist Temporal Classification (CTC) loss, or continuous diffusion processes over token representations.
