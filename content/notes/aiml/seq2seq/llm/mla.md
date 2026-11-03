---
title: "MLA"
date: "2026-10-27"
---

**Multi-Head Latent Attention (MLA)**, introduced by DeepSeek, solves a core bottleneck in Transformer inference: the massive size and memory bandwidth cost of the **KV Cache**.

Chris McCormick

While Grouped-Query Attention (GQA) reduces KV heads at the cost of model expressiveness, MLA achieves extreme compression without losing representational capacity by using **low-rank latent projections** and **algebraic weight absorption**.

Chris McCormick

## 1\. Why Standard Multi-Head Attention (MHA) Is Costly

For an input token vector $\mathbf{x}_t \in \mathbb{R}^d$ across $n_h$ heads, each head generates query, key, and value vectors of dimension $d_h$:

$$
\mathbf{q}_{t,i} = \mathbf{x}_t \mathbf{W}_i^Q, \quad \mathbf{k}_{t,i} = \mathbf{x}_t \mathbf{W}_i^K, \quad \mathbf{v}_{t,i} = \mathbf{x}_t \mathbf{W}_i^V \quad (i = 1, \dots, n_h)
$$

During autoregressive generation, we must store all past keys and values in memory:

$$
\text{KV Cache per token} = 2 \times n_h \times d_h \text{ numbers}
$$

For large models (e.g., 128 heads, head dimension 128), that is over 32,000 floats per token, dominating memory footprint and memory-bus bandwidth.

Chris McCormick

## 2\. Multi-Head Latent Attention: Without RoPE

The central intuition of MLA is simple: **instead of storing full multi-head keys and values, compress each token's key and value representations into a single low-dimensional latent vector.**

### The Compression (Down-Projection)

Instead of projecting $\mathbf{x}_t$ directly to $n_h$ heads, we down-project it into a compact latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ where $d_c \ll n_h \times d_h$:

$$
\mathbf{c}_t^{KV} = \mathbf{x}_t \mathbf{W}^{DKV}
$$

- $\mathbf{W}^{DKV} \in \mathbb{R}^{d \times d_c}$ is the down-projection matrix.
- **In the KV cache, we only store $\mathbf{c}_t^{KV}$** (a single vector of size $d_c$ per token, rather than $2 \cdot n_h \cdot d_h$).
	Chris McCormick

*(Queries are also compressed during training to save activation memory via $\mathbf{c}_t^Q = \mathbf{x}_t \mathbf{W}^{DQ}$, then up-projected to $\mathbf{q}_{t,i} = \mathbf{c}_t^Q \mathbf{W}_i^{UQ}$, though query compression does not affect KV cache sizing.)*

### Generation & The "Weight Absorption" Trick

To compute attention, keys and values for head $i$ are theoretically uncompressed using up-projection matrices:

$$
\mathbf{k}_{j,i} = \mathbf{c}_j^{KV} \mathbf{W}_i^{UK}, \quad \mathbf{v}_{j,i} = \mathbf{c}_j^{KV} \mathbf{W}_i^{UV}
$$

where $\mathbf{W}_i^{UK} \in \mathbb{R}^{d_c \times d_h}$ and $\mathbf{W}_i^{UV} \in \mathbb{R}^{d_c \times d_h}$.

The attention score between query token $t$ and key token $j$ for head $i$ is:

$$
\text{Score}_{i}(t, j) = \mathbf{q}_{t,i} \mathbf{k}_{j,i}^T = \mathbf{q}_{t,i} \left( \mathbf{c}_j^{KV} \mathbf{W}_i^{UK} \right)^T = \mathbf{q}_{t,i} (\mathbf{W}_i^{UK})^T (\mathbf{c}_j^{KV})^T
$$

Because matrix multiplication is associative, we can group the terms differently:

$$
\text{Score}_{i}(t, j) = \underbrace{\left(\mathbf{q}_{t,i} (\mathbf{W}_i^{UK})^T\right)}_{\mathbf{q}'_{t,i} \in \mathbb{R}^{d_c}} (\mathbf{c}_j^{KV})^T
$$

This provides a major performance optimization:

1. **Never decompress keys in memory:** We project the single active query vector down into the latent space: $\mathbf{q}'_{t,i} = \mathbf{q}_{t,i} (\mathbf{W}_i^{UK})^T \in \mathbb{R}^{d_c}$.
2. Compute the dot-product directly against the cached latent $\mathbf{c}_j^{KV}$.
3. For values, $\mathbf{W}_i^{UV}$ can similarly be absorbed directly into the output projection $\mathbf{W}^O$.

## 3\. The RoPE Problem

Rotary Position Embedding (**RoPE**) injects relative token distance by rotating vectors with a position-dependent orthogonal matrix $\mathbf{R}_t$:

Outcome School

$$
\mathbf{q}_{t} \to \mathbf{q}_t \mathbf{R}_t, \quad \mathbf{k}_{j} \to \mathbf{k}_j \mathbf{R}_j
$$

If we apply standard RoPE to the decompressed key $\mathbf{k}_{j,i} = \mathbf{c}_j^{KV} \mathbf{W}_i^{UK}$:

$$
\mathbf{k}_{j,i}^{\text{RoPE}} = \mathbf{c}_j^{KV} \mathbf{W}_i^{UK} \mathbf{R}_j
$$

The attention dot-product becomes:

$$
\text{Score}_{i}(t, j) = (\mathbf{q}_{t,i} \mathbf{R}_t) (\mathbf{k}_{j,i}^{\text{RoPE}})^T = \mathbf{q}_{t,i} \mathbf{R}_t \mathbf{R}_j^T (\mathbf{W}_i^{UK})^T (\mathbf{c}_j^{KV})^T
$$

Because $\mathbf{R}_j^T$ sits squarely between $\mathbf{W}_i^{UK}$ and $\mathbf{c}_j^{KV}$, **weight absorption breaks**. You would either have to:

- Decompress and rotate keys for every token in history at runtime (compute-prohibitive), or
	varmology
- Cache the uncompressed, rotated keys across all heads (defeating the purpose of compressing the KV cache).
	varmology

## 4\. Multi-Head Latent Attention: With Decoupled RoPE

MLA resolves this conflict by separating attention into two distinct parallel streams: a **content stream** and a **position stream**.


### The Positional Pathway

1. **Decoupled Key:** We project token $\mathbf{x}_t$ directly to a shared, low-dimensional position vector $\mathbf{k}_t^R \in \mathbb{R}^{d_R}$ (typically $d_R = 64$), and apply rotation $\mathbf{R}_t$ to it:
	$$
	\mathbf{k}_{t}^R = \text{RoPE}\left(\mathbf{x}_t \mathbf{W}^{KR}\right)
	$$
	*(Notice that all heads share this positional key or use a unified projection.)*
	arXiv
2. **Decoupled Query:** We project queries into both a content query $\mathbf{q}_{t,i}^C \in \mathbb{R}^{d_h}$ and a positional query $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d_R}$:
	$$
	\mathbf{q}_{t,i}^R = \text{RoPE}\left(\mathbf{c}_t^Q \mathbf{W}_i^{QR}\right)
	$$

### The Attention Score Calculation

The final query and key vectors are formed by concatenating their content and positional components:

$$
\mathbf{q}_{t,i} = \left[ \mathbf{q}_{t,i}^C \;;\; \mathbf{q}_{t,i}^R \right], \quad \mathbf{k}_{j,i} = \left[ \mathbf{k}_{j,i}^C \;;\; \mathbf{k}_j^R \right]
$$

The dot-product naturally splits into two additive terms:

$$
\text{Score}_{i}(t, j) = \mathbf{q}_{t,i}^C (\mathbf{k}_{j,i}^C)^T + \mathbf{q}_{t,i}^R (\mathbf{k}_j^R)^T
$$

Substitute the content and RoPE representations:

$$
\text{Score}_{i}(t, j) = \underbrace{\left(\mathbf{q}_{t,i}^C (\mathbf{W}_i^{UK})^T\right)}_{\text{Absorbed: } \mathbb{R}^{d_c}} (\mathbf{c}_j^{KV})^T + \underbrace{\mathbf{q}_{t,i}^R (\mathbf{k}_j^R)^T}_{\text{RoPE dot-product: } \mathbb{R}^{d_R}}
$$

- The **content component** retains full weight absorption—keys do not need to be uncompressed.
- The **position component** is small ($d_R \approx 64$) and decoupled from head-expansion.
	Chris McCormick

## Cache Size Comparison

| Architecture | Stored per token in KV Cache | Representative Size ($n_h=128,\ d_h=128$) |
| --- | --- | --- |
| **Standard MHA** | $2\cdot n_h\cdot d_h$ | $2\cdot 128\cdot 128=32{,}768$ floats |
| **GQA (e.g., 8 KV groups)** | $2\cdot n_{\text{kv\_heads}}\cdot d_h$ | $2\cdot 8\cdot 128=2{,}048$ floats |
| **MLA (with Decoupled RoPE)** | $d_c+d_R$ | $512+64=576$ floats |

MLA cuts cache memory by **$\sim 98\%$ compared to MHA** and **$\sim 70\%$ compared to standard GQA**, while preserving full multi-head expressiveness during training and inference.
