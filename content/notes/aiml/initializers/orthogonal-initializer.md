---
title: "Orthogonal initializer"
date: "2026-10-26"
---

An **orthogonal weight initializer** initializes the weight matrix $W$ of a neural network layer such that its column or row vectors are mutually perpendicular (orthogonal) and have unit length (orthonormal).

In mathematical terms, for a square matrix $W \in \mathbb{R}^{n \times n}$:

$$
W^T W = W W^T = I
$$

For rectangular weight matrices ($m \times n$):

- If $m > n$: $W^T W = I_n$ (orthonormal columns)
- If $m < n$: $W W^T = I_m$ (orthonormal rows)

### The Problem It Solves

During forward and backpropagation through deep architectures (especially Deep Linear Networks or Recurrent Neural Networks), signals are repeatedly multiplied by weight matrices:

- **Vanishing Gradients / Activations:** If the eigenvalues (singular values) of $W$ are strictly less than $1$ ($\sigma < 1$), repeated multiplication causes the signal to shrink exponentially toward zero ($\sigma^L \to 0$ as layer depth $L$ grows).
- **Exploding Gradients / Activations:** If the singular values are strictly greater than $1$ ($\sigma > 1$), the signal magnitude compounds exponentially ($\sigma^L \to \infty$), causing numerical instability.

An orthogonal matrix preserves vector norms (lengths) under transformation:

$$
\Vert{}Wx\Vert{}_2 = \Vert{}x\Vert{}_2
$$

Because all singular values of an orthogonal matrix are exactly **1**, signals and gradients pass through layers without growing or decaying exponentially, drastically stabilizing training.

### How It Is Constructed

Orthogonal initializers typically use **Singular Value Decomposition (SVD)** or **QR Decomposition** on a random matrix:

1. Sample a random matrix $A \in \mathbb{R}^{m \times n}$ from a standard normal distribution $\mathcal{N}(0, 1)$.
2. Compute the QR decomposition of $A$:
	$$
	A = Q R
	$$
	where $Q$ is an orthogonal matrix and $R$ is upper triangular.
3. Adjust signs: Because standard QR algorithms do not guarantee a uniform distribution over the orthogonal group (the Haar measure), diagonal elements of $R$ are used to correct signs:
	$$
	Q \leftarrow Q \cdot \operatorname{diag}(\operatorname{sgn}(\operatorname{diag}(R)))
	$$
4. **Gain Scaling:** Since non-linear activation functions (like ReLU or Tanh) alter variance, libraries apply a multiplier $g$ (gain):
	$$
	W = g \cdot Q
	$$
	*(e.g., $g = \sqrt{2}$ for ReLU, $g = 1.0$ or $\approx 1.13$ for linear/tanh).*

### Key Applications

- **Recurrent Neural Networks (RNNs / LSTMs / GRUs):** Hidden-to-hidden recurrent transition matrices ($W_{hh}$) are applied once per time step. For long sequences ($T \ge 100$), non-orthogonal matrices quickly cause exploding or vanishing gradients over time.
- **Very Deep Networks Without Residual Connections:** Residual connections (ResNets) mitigate signal decay through identity skip paths, but for plain feedforward networks or deep linear networks, orthogonal weights enable training architectures hundreds of layers deep without normalization layers.
- **Deep Reinforcement Learning:** Widely used in policy and value network layers (e.g., in PPO, A2C) to stabilize early exploration trajectories by preventing dead or saturated initial neurons.

### Orthogonal vs. Xavier / He Initialization

| Feature | Xavier / Glorot & He Initialization | Orthogonal Initialization |
| --- | --- | --- |
| **Guaranteed Property** | Preserves variance on **average** (in expectation across random samples). | Preserves norm/variance **strictly** along each principal axis. |
| **Singular Value Distribution** | Singular values follow the Marchenko-Pastur law (spread out between $1-\sqrt{\lambda}$ and $1+\sqrt{\lambda}$). | All singular values are identically **1** (before gain scaling). |
| **Primary Use Cases** | Standard CNNs, Transformers, MLPs with batch/layer normalization. | RNN hidden weights, Deep RL policy networks, networks without skip connections. |
| **Computational Cost** | $O(mn)$ (trivial random sampling). | $O(\min(m,n)\cdot mn)$ (requires matrix decomposition). |
