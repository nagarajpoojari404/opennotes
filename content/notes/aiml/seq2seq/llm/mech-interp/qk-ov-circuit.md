---
title: "QK OV circuit"
date: "2026-10-27"
---

Mechanistic interpretability treats neural networks not as black boxes, but as computational graphs where weights and activations can be reverse-engineered into human-understandable algorithms.

In the mathematical framework popularized by Anthropic (Elhage et al., *A Mathematical Framework for Transformer Circuits*), an attention layer is analyzed without the intermediate token dimensions by factoring it into two independent, end-to-end operations:

1. **The QK (Query-Key) circuit**: Determines **where** information moves.
2. **The OV (Output-Value) circuit**: Determines **what** information moves.

### The Residual Stream as a Communication Bus

In a decoder-only autoregressive Transformer, token representations live in an $n_{\text{model}}$\-dimensional space called the **residual stream** ($x \in \mathbb{R}^{d_{\text{model}}}$).

Layers read from the residual stream via linear projections and write back into it by addition:

$$
x_{l} = x_{l-1} + \sum_{h=1}^{H} \text{head}_{l,h}(x_{l-1})
$$

A standard multi-head attention head $h$ uses four weight matrices:

- $W_Q, W_K \in \mathbb{R}^{d_k \times d_{\text{model}}}$
- $W_V \in \mathbb{R}^{d_v \times d_{\text{model}}}$
- $W_O \in \mathbb{R}^{d_{\text{model}} \times d_v}$

Usually, we think of attention sequentially:

1. $q_i = W_Q x_i$, $k_j = W_K x_j$
2. $A_{i,j} = \text{softmax}_j\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)$
3. $v_j = W_V x_j$
4. $\text{head}(x)_i = W_O \sum_j A_{i,j} v_j$

Mechanistic interpretability collapses the internal dimension ($d_k$ or $d_v$) into direct end-to-end bilinear maps on the residual stream.

### 1\. The QK Circuit: Directing Attention

The attention score between query token at position $i$ and key token at position $j$ (ignoring positional embeddings for a moment) is:

$$
\text{score}_{i,j} = \frac{q_i^\top k_j}{\sqrt{d_k}} = \frac{(W_Q x_i)^\top (W_K x_j)}{\sqrt{d_k}} = \frac{x_i^\top W_Q^\top W_K x_j}{\sqrt{d_k}}
$$

We define the **QK matrix** as:

$$
W_{QK} = W_Q^\top W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

- **Rank**: $W_{QK}$ has a maximum rank of $d_k$ (typically 64 or 128), which is much smaller than $d_{\text{model}}$ (often 4096+). It acts as a low-rank bilinear form.
- **Interpretation**: Given a feature subspace active in the query token ($x_i$) and a feature subspace active in the key token ($x_j$), $W_{QK}$ directly computes their compatibility score:

$$
\text{score}_{i,j} \propto x_i^\top W_{QK} x_j
$$

- **Function**: It acts as an **addressing system**. The QK circuit reads queries and keys from the residual stream and outputs scalar logits over context positions, entirely agnostic to the semantic content being copied or transformed downstream.

### 2\. The OV Circuit: Transforming Content

Once attention weights $A_{i,j} \in [0, 1]$ are computed, the head moves information from source position $j$ to destination position $i$. Looking at a single source position $j$:

$$
\text{contribution}_j = W_O (W_V x_j) = (W_O W_V) x_j
$$

We define the **OV matrix** as:

$$
W_{OV} = W_O W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

- **Rank**: Like $W_{QK}$, $W_{OV}$ has a rank bounded by $d_v \ll d_{\text{model}}$.
- **Interpretation**: $W_{OV}$ is a linear map that reads a low-rank subspace from the residual stream at the source position and writes another low-rank subspace directly into the residual stream at the destination position.
- **Function**: It dictates **what** features get extracted from the source and **in what direction** they are written into the target. Crucially, $W_{OV}$ has zero influence on which positions are attended to—it simply specifies the payload.

### Complete Head Formulation

Putting both circuits together, the action of an attention head at destination token $i$ can be written without ever explicitly referencing the intermediate $d_k$ or $d_v$ dimensions:

$$
\text{head}(x)_i = \sum_{j \le i} \underbrace{\text{softmax}_j \left( \frac{x_i^\top W_{QK} x_j}{\sqrt{d_k}} \right)}_{\text{QK Circuit: Where to route}} \cdot \underbrace{\left( W_{OV} x_j \right)}_{\text{OV Circuit: What to write}}
$$

| Circuit | Matrix | Operation | Role |
| --- | --- | --- | --- |
| **QK** | $W_{QK}=W_Q^\top W_K$ | $x_i^\top W_{QK}x_j\to\text{Scalar}$ | Computes attention pattern (source-destination affinity). |
| **OV** | $W_{OV}=W_OW_V$ | $W_{OV}x_j\to\mathbb{R}^{d_{\text{model}}}$ | Transports and rotates features from source to destination. |

### Why Factorize into QK and OV?

1. **Separation of Concerns**: When analyzing trained models, an attention head can fail or succeed in two distinct ways:
	- **Addressing failure**: Looking at the wrong token (QK circuit issue).
		- **Transmission failure**: Looking at the right token but writing the wrong vector or projecting into an orthogonal subspace (OV circuit issue).
2. **Eigenvalue/SVD Analysis of $W_{OV}$**:
	- If $W_{OV}$ has positive real eigenvalues, the head acts as an **identity/copying head**, reinforcing existing token features.
		- If it has negative real eigenvalues, it acts as an **inhibition head**, erasing or dampening certain directions in the residual stream.
3. **Composability Across Layers**: In multi-layer models, the output of an early head's OV circuit feeds into the input of a downstream head's QK or OV circuit. This leads directly to **induction heads** and attention composition pathways ($Q$\-composition, $K$\-composition, and $V$\-composition).
