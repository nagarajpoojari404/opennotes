---
title: "Backprop in tensor form"
date: "2026-10-26"
---

In tensor index notation, backpropagation reduces to applying the multivariable chain rule via contractions. This eliminates ambiguity around transpositions and shapes across batches and multi-dimensional layers.

### 1\. Forward Pass Formulation

Consider a mini-batch through a single fully connected layer:

- **Input tensor:** $X \in \mathbb{R}^{B \times D_{\text{in}}}$ with components $X^b_{\ i}$
- **Weight tensor:** $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ with components $W^i_{\ j}$
- **Bias tensor:** $b \in \mathbb{R}^{D_{\text{out}}}$ with components $b_j$
- **Scalar loss:** $\mathcal{L} \in \mathbb{R}$

The pre-activation $Z^b_{\ j}$ and activation $A^b_{\ j}$ for batch sample $b$ and feature index $j$ are:

$$
Z^b_{\ j} = X^b_{\ i} W^i_{\ j} + b_j
$$

$$
A^b_{\ j} = \sigma\left(Z^b_{\ j}\right)
$$

where repeated indices denote summation (Einstein summation convention), and $\sigma$ is an element-wise activation function.

### 2\. The Adjoint / Error Tensor ($\Delta$)

Define the incoming gradient of the scalar loss $\mathcal{L}$ with respect to the post-activation $A^b_{\ j}$ as:

$$
G^b_{\ j} = \frac{\partial \mathcal{L}}{\partial A^b_{\ j}}
$$

By the multivariable chain rule, the gradient with respect to pre-activation $Z^b_{\ j}$ is:

$$
\Delta^b_{\ j} \equiv \frac{\partial \mathcal{L}}{\partial Z^b_{\ j}} = \frac{\partial \mathcal{L}}{\partial A^c_{\ k}} \frac{\partial A^c_{\ k}}{\partial Z^b_{\ j}}
$$

Because $\sigma$ acts element-wise, the Jacobian is diagonal:

$$
\frac{\partial A^c_{\ k}}{\partial Z^b_{\ j}} = \sigma'\left(Z^b_{\ j}\right) \delta^c_b \delta^k_j
$$

Contracting indices via the Kronecker deltas $\delta^c_b$ and $\delta^k_j$:

$$
\Delta^b_{\ j} = G^c_{\ k} \left(\sigma'\left(Z^b_{\ j}\right) \delta^c_b \delta^k_j\right) = G^b_{\ j} \odot \sigma'\left(Z^b_{\ j}\right)
$$

### 3\. Gradient with Respect to Weights ($W$)

Applying the chain rule to the weight tensor $W^i_{\ j}$:

$$
\frac{\partial \mathcal{L}}{\partial W^i_{\ j}} = \frac{\partial \mathcal{L}}{\partial Z^b_{\ k}} \frac{\partial Z^b_{\ k}}{\partial W^i_{\ j}} = \Delta^b_{\ k} \frac{\partial Z^b_{\ k}}{\partial W^i_{\ j}}
$$

Differentiating the forward relation $Z^b_{\ k} = X^b_{\ m} W^m_{\ k} + b_k$ with respect to $W^i_{\ j}$:

$$
\frac{\partial Z^b_{\ k}}{\partial W^i_{\ j}} = X^b_{\ m} \frac{\partial W^m_{\ k}}{\partial W^i_{\ j}} = X^b_{\ m} \delta^m_i \delta^j_k = X^b_{\ i} \delta^j_k
$$

Substitute this back into the contraction:

$$
\frac{\partial \mathcal{L}}{\partial W^i_{\ j}} = \Delta^b_{\ k} \left(X^b_{\ i} \delta^j_k\right) = X^b_{\ i} \Delta^b_{\ j}
$$

Notice that index $b$ is summed over all samples in the mini-batch:

- In matrix notation, this corresponds directly to:
	$$
	\nabla_W \mathcal{L} = X^T \Delta
	$$

### 4\. Gradient with Respect to Bias ($b$)

Differentiating $Z^b_{\ k}$ with respect to bias $b_j$:

$$
\frac{\partial Z^b_{\ k}}{\partial b_j} = \delta^k_j
$$

Applying the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial b_j} = \Delta^b_{\ k} \frac{\partial Z^b_{\ k}}{\partial b_j} = \Delta^b_{\ k} \delta^k_j = \sum_{b=1}^B \Delta^b_{\ j}
$$

The bias gradient is simply the contraction over the batch dimension $b$.

### 5\. Gradient with Respect to Layer Input ($X$)

To propagate errors back to earlier layers, differentiate with respect to $X^b_{\ i}$:

$$
\frac{\partial \mathcal{L}}{\partial X^b_{\ i}} = \Delta^c_{\ k} \frac{\partial Z^c_{\ k}}{\partial X^b_{\ i}}
$$

Evaluating the Jacobian of $Z$ with respect to $X$:

$$
\frac{\partial Z^c_{\ k}}{\partial X^b_{\ i}} = \frac{\partial}{\partial X^b_{\ i}} \left( X^c_{\ m} W^m_{\ k} + b_k \right) = \delta^c_b \delta^m_i W^m_{\ k} = \delta^c_b W^i_{\ k}
$$

Contracting over $c$:

$$
\frac{\partial \mathcal{L}}{\partial X^b_{\ i}} = \Delta^c_{\ k} \left( \delta^c_b W^i_{\ k} \right) = \Delta^b_{\ k} W^i_{\ k}
$$

- In matrix notation, summing over feature index $k$ corresponds to multiplying by the transposed weight matrix:
	$$
	\nabla_X \mathcal{L} = \Delta W^T
	$$
