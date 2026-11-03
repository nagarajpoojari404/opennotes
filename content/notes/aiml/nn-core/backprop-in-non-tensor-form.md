---
title: "Backprop in non-tensor form"
date: "2026-10-26"
---

Here is the same derivation using standard scalar calculus and matrix notation, avoiding tensor indices and Einstein summation.

### 1\. Forward Pass

For a mini-batch of size $B$, input dimension $D_{\text{in}}$, and output dimension $D_{\text{out}}$:

- Input matrix: $X \in \mathbb{R}^{B \times D_{\text{in}}}$
- Weight matrix: $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$
- Bias vector: $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ (broadcast across rows)
- Scalar loss: $\mathcal{L} \in \mathbb{R}$

The pre-activation matrix $Z$ and activation matrix $A$ are:

$$
Z = XW + \mathbf{1}_B b
$$

$$
A = \sigma(Z)
$$

At the individual element level, for sample $b \in \{1, \dots, B\}$ and feature $j \in \{1, \dots, D_{\text{out}}\}$:

$$
z_{bj} = \sum_{k=1}^{D_{\text{in}}} x_{bk} w_{kj} + b_j
$$

$$
a_{bj} = \sigma(z_{bj})
$$

### 2\. Error Matrix ($\Delta$)

Let the incoming upstream gradient with respect to the activation matrix be:

$$
G = \frac{\partial \mathcal{L}}{\partial A}, \quad \text{where } G_{bj} = \frac{\partial \mathcal{L}}{\partial a_{bj}}
$$

Using the standard single-variable chain rule for element $(b, j)$, since $a_{bj}$ only depends on $z_{bj}$:

$$
\Delta_{bj} \equiv \frac{\partial \mathcal{L}}{\partial z_{bj}} = \frac{\partial \mathcal{L}}{\partial a_{bj}} \cdot \frac{\partial a_{bj}}{\partial z_{bj}} = G_{bj} \cdot \sigma'(z_{bj})
$$

In matrix notation, this is the Hadamard (element-wise) product:

$$
\Delta = G \odot \sigma'(Z)
$$

### 3\. Gradient with Respect to Weights ($W$)

A single weight $w_{ij}$ connects input feature $i$ to output feature $j$. It influences the loss through every batch item's pre-activation $z_{bj}$ for that specific output $j$:

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial z_{bj}} \frac{\partial z_{bj}}{\partial w_{ij}}
$$

From the forward equation $z_{bj} = \sum_{k} x_{bk} w_{kj} + b_j$, differentiating with respect to $w_{ij}$ gives:

$$
\frac{\partial z_{bj}}{\partial w_{ij}} = x_{bi}
$$

Substitute this back into the sum:

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{b=1}^B x_{bi} \Delta_{bj} = \sum_{b=1}^B (X^T)_{ib} \Delta_{bj}
$$

Because the row index of $X$ matches the row index of $\Delta$ across the summation over $b$, this is the definition of matrix multiplication between $X^T$ and $\Delta$:

$$
\frac{\partial \mathcal{L}}{\partial W} = X^T \Delta
$$

- Dimensions check: $(D_{\text{in}} \times B) \times (B \times D_{\text{out}}) = D_{\text{in}} \times D_{\text{out}}$.

### 4\. Gradient with Respect to Bias ($b$)

Each bias component $b_j$ affects the pre-activation $z_{bj}$ across all $B$ batch samples:

$$
\frac{\partial \mathcal{L}}{\partial b_j} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial z_{bj}} \frac{\partial z_{bj}}{\partial b_j}
$$

Since $\frac{\partial z_{bj}}{\partial b_j} = 1$:

$$
\frac{\partial \mathcal{L}}{\partial b_j} = \sum_{b=1}^B \Delta_{bj}
$$

In matrix form, this sums the columns of $\Delta$ across the batch dimension:

$$
\frac{\partial \mathcal{L}}{\partial b} = \mathbf{1}_B^T \Delta
$$

- Dimensions check: $(1 \times B) \times (B \times D_{\text{out}}) = 1 \times D_{\text{out}}$.

### 5\. Gradient with Respect to Layer Input ($X$)

To backpropagate to previous layers, compute the gradient for a single input element $x_{bi}$. This input affects all pre-activations $z_{bk}$ for the same sample $b$ across all output units $k$:

$$
\frac{\partial \mathcal{L}}{\partial x_{bi}} = \sum_{k=1}^{D_{\text{out}}} \frac{\partial \mathcal{L}}{\partial z_{bk}} \frac{\partial z_{bk}}{\partial x_{bi}}
$$

Differentiating $z_{bk} = \sum_{m} x_{bm} w_{mk} + b_k$ with respect to $x_{bi}$:

$$
\frac{\partial z_{bk}}{\partial x_{bi}} = w_{ik}
$$

Substitute this into the summation:

$$
\frac{\partial \mathcal{L}}{\partial x_{bi}} = \sum_{k=1}^{D_{\text{out}}} \Delta_{bk} w_{ik} = \sum_{k=1}^{D_{\text{out}}} \Delta_{bk} (W^T)_{ki}
$$

Summing over the output dimension $k$ corresponds directly to multiplying $\Delta$ by the transpose of $W$:

$$
\frac{\partial \mathcal{L}}{\partial X} = \Delta W^T
$$

- Dimensions check: $(B \times D_{\text{out}}) \times (D_{\text{out}} \times D_{\text{in}}) = B \times D_{\text{in}}$.
