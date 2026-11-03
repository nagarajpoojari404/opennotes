---
title: "Induction head"
date: "2026-10-27"
---

An **induction head** is a two-layer attention sub-circuit that completes recurring sequences via pattern matching: given context `... [A] [B] ... [A]`, it predicts that `[B]` follows the current `[A]`.

Induction heads are considered the primary mechanistic engine driving in-context learning in Transformers. To implement this algorithm across layers, heads must compose by reading each other's outputs from the residual stream.

### The Algorithmic Goal of an Induction Head

Suppose the input sequence contains:

$$
\dots \underbrace{\text{[A]}}_{i-1} \ \underbrace{\text{[B]}}_{i} \ \dots \ \underbrace{\text{[A]}}_{j}
$$

The current token is at position $j$ (which is `[A]`). To predict that `[B]` comes next, the network must:

1. Search backward in the context for an earlier occurrence of token `[A]`.
2. Find the token that immediately followed that earlier `[A]` (which is `[B]` at position $i$).
3. Copy `[B]` to the current prediction at position $j$.

A single attention layer **cannot** do this on its own:

- At position $i$ (where `[B]` is located), the token `[B]` has no native awareness that token `[A]` occurred immediately before it, unless another head or embedding shifted that information into position $i$.
- A 1-layer model can implement a "previous-token head" or a "copy-current-token head", but it cannot perform "look for previous token matching current token, and extract the next token."

This requires a circuit consisting of at least two heads across two distinct layers:

- **Head 1 (Layer 0 or early layer)**: A **Previous-Token Head** ($H_1$).
- **Head 2 (Layer 1 or downstream layer)**: An **Induction Head** ($H_2$).

### The Attention Composition Framework

In a multi-layer Transformer, Layer 1 receives the residual stream modified by Layer 0:

$$
x^{(1)} = x^{(0)} + \sum_{h \in L_0} \text{head}_h(x^{(0)})
$$

When head $H_2$ in Layer 1 runs, its input vector $x$ contains terms from $H_1$'s OV output: $W_{OV}^{(1)} x$. Expanding the inputs to $H_2$ reveals three distinct ways an upstream head can feed into a downstream head:

```
Upstream Head H1 (Layer 0)
       |
    [W_OV]
       | (Residual Stream)
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
     [W_Q]                   [W_K]                   [W_V]
       |                       |                       |
Downstream Head H2 Q-Input   Downstream Head H2 K-Input   Downstream Head H2 V-Input
(Q-Composition)             (K-Composition)             (V-Composition)
```
1. **Q-Composition**: $H_1$'s output feeds into $H_2$'s Query projection:
	$$
	W_Q^{(2)} \left( W_{OV}^{(1)} x \right) = \left( W_Q^{(2)} W_{OV}^{(1)} \right) x
	$$
2. **K-Composition**: $H_1$'s output feeds into $H_2$'s Key projection:
	$$
	W_K^{(2)} \left( W_{OV}^{(1)} x \right) = \left( W_K^{(2)} W_{OV}^{(1)} \right) x
	$$
3. **V-Composition**: $H_1$'s output feeds into $H_2$'s Value projection:
	$$
	W_V^{(2)} \left( W_{OV}^{(1)} x \right) = \left( W_V^{(2)} W_{OV}^{(1)} \right) x
	$$

### Step-by-Step: How Induction Heads Work

The standard induction circuit uses **K-Composition** (or Q-Composition) to set up the attention pattern, followed by the OV circuit to copy the token.

#### Step 1: Head $H_1$ acts as a Previous-Token Head (Layer 0)

Head $H_1$ attends from position $i$ to position $i-1$ (often relying on positional embeddings/RoPE in its QK circuit):

- At position $i$ (which holds `[B]`), $H_1$ reads the representation of `[A]` from position $i-1$.
- $H_1$'s OV circuit writes a vector containing the identity of `[A]` directly into the residual stream at position $i$:
	$$
	x_i^{(1)} \approx x_i^{(0)} + W_{OV}^{(1)} x_{i-1}^{(0)}
	$$
- **Result**: Position $i$ (token `[B]`) now carries a tag in the residual stream saying *"I was preceded by token \[A\]"*.

#### Step 2: Head $H_2$ uses K-Composition to Match Queries and Keys (Layer 1)

At the current position $j$, the token is `[A]`. Head $H_2$'s Query vector at position $j$ is computed directly from `[A]`:

$$
q_j = W_Q^{(2)} x_j \approx W_Q^{(2)} x_{\text{[A]}}
$$

Meanwhile, for all earlier positions $i$, Head $H_2$'s Key vector receives the residual stream containing $H_1$'s output via **K-composition**:

$$
k_i = W_K^{(2)} x_i^{(1)} = W_K^{(2)} \left( x_i^{(0)} + W_{OV}^{(1)} x_{i-1}^{(0)} \right) = W_K^{(2)} x_i^{(0)} + \underbrace{W_K^{(2)} W_{OV}^{(1)} x_{i-1}^{(0)}}_{\text{K-Composition Term}}
$$

The QK score between current position $j$ and prior position $i$ contains the bilinear term:

$$
\text{score}_{j,i} \propto q_j^\top k_i \supset \left( W_Q^{(2)} x_{\text{[A]}} \right)^\top \left( W_K^{(2)} W_{OV}^{(1)} x_{i-1}^{(0)} \right) = x_{\text{[A]}}^\top \left( (W_Q^{(2)})^\top W_K^{(2)} W_{OV}^{(1)} \right) x_{i-1}^{(0)}
$$

Because $x_{i-1}^{(0)}$ is also token `[A]`, the combined composition matrix:

$$
C_K = (W_Q^{(2)})^\top W_K^{(2)} W_{OV}^{(1)}
$$

acts as an **identity/match detector** between token $j$ and token $i-1$.

When token $j$ matches token $i-1$ (both are `[A]`), the score spikes positively, causing Head $H_2$ at position $j$ to place nearly all its attention weight on position $i$ (the token `[B]`).

#### Step 3: Copying via $H_2$'s OV Circuit

Once $H_2$ attends to position $i$, its own Value and Output matrices execute a copying operation:

$$
\text{output}_j = W_O^{(2)} W_V^{(2)} x_i
$$

Since $x_i$ holds `[B]`, $W_{OV}^{(2)} = W_O^{(2)} W_V^{(2)}$ projects the token embedding of `[B]` into the direction of the residual stream that the final unembedding matrix ($W_U$) decodes as `[B]`.

### Q-Composition vs. K-Composition

Induction heads can be implemented using either mechanism depending on how the model factors the problem:

| Property | K-Composition Induction | Q-Composition Induction |
| --- | --- | --- |
| **Upstream Head ($H_1$)** | Writes "previous token is `[A]`" into position $i$ (`[B]`). | Writes "subsequent token should match `[A]`" into position $j$. |
| **Composed Matrix** | $W_K^{(2)}W_{OV}^{(1)}$ | $W_Q^{(2)}W_{OV}^{(1)}$ |
| **Mechanistic Role** | Changes what key is presented at candidate positions $i$. | Changes what query is searched for by current position $j$. |
| **Intuition** | Query searches for *"who had token \[A\] before them?"* Key responds *"I did!"* | Query says *"I am looking for the token following \[A\]"*. Key says *"I am token \[B\]"*. |

In empirical circuits research (e.g., in toy models and open LLMs), **K-composition** is the dominant form observed for induction heads, though mixed variants (both Q- and K-composition active simultaneously) also occur.

### Detecting Composition in the Wild

To confirm whether two heads $H_1$ and $H_2$ are actually composing rather than operating independently, interpretability researchers use two quantitative tests:

1. **Path Patching (Activation Patching)**: Run the model on a repeated sequence `[A][B] ... [A]`. Intercept the output of $H_1$ entering $H_2$'s $K$\-input and replace it with corrupted/random activations. If the model's ability to predict `[B]` vanishes, $H_2$ causally depends on $H_1$ via K-composition.
2. **Matrix Norm Metrics**: Measure the Frobenius norm of the product matrix:
	$$
	\Vert{} W_K^{(2)} W_{OV}^{(1)} \Vert{}_F
	$$
	Compare this against a baseline where the weight matrices are randomly rotated. In true induction heads, the norm is significantly higher than chance, indicating that $W_K^{(2)}$ specifically aligns with the subspace output by $W_{OV}^{(1)}$.
