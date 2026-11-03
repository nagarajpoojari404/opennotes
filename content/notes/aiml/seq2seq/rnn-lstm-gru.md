---
title: "RNN,LSTM,GRU"
date: "2026-10-26"
references: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
---

Recurrent architectures process sequential data by maintaining an internal hidden state that persists information across time steps.

## 1\. Standard Recurrent Neural Network (Vanilla RNN)

A standard RNN processes an input sequence $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ step-by-step. At each time step $t$, it updates its hidden state $\mathbf{h}_t$ based on the current input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$.

### Mathematical Formulation

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\hat{\mathbf{y}}_t = \sigma(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)
$$

- **$\mathbf{x}_t \in \mathbb{R}^{d}$**: Input vector at time step $t$.
- **$\mathbf{h}_t \in \mathbb{R}^{h}$**: Hidden state at time step $t$.
- **$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$**, **$\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$**, **$\mathbf{W}_{hy} \in \mathbb{R}^{k \times h}$**: Weight matrices.
- **$\mathbf{b}_h \in \mathbb{R}^{h}$**, **$\mathbf{b}_y \in \mathbb{R}^{k}$**: Bias vectors.
- **$\tanh$**: Hyperbolic tangent activation function, mapping values to $(-1, 1)$.

### Limitation: The Vanishing/Exploding Gradient Problem

When training via **Backpropagation Through Time (BPTT)**, the gradient of the loss $\mathcal{L}_T$ with respect to $\mathbf{h}_1$ requires computing a chain of Jacobian matrices:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=2}^T \text{diag}(1 - \mathbf{h}_k^2) \mathbf{W}_{hh}^T
$$

If the largest eigenvalue of $\mathbf{W}_{hh}$ is less than 1, gradients decay exponentially toward zero as $T$ grows, making it impossible to capture long-term dependencies.

## 2\. Long Short-Term Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997), LSTMs solve vanishing gradients by introducing a **cell state** ($\mathbf{c}_t$) that acts as an information highway, regulated by three multiplicative gating mechanisms.

### Gates and Operations

1. **Forget Gate ($\mathbf{f}_t$):** Determines what proportion of the old cell state to discard.
	$$
	\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
	$$
2. **Input Gate ($\mathbf{i}_t$) & Candidate Cell State ($\tilde{\mathbf{c}}_t$):** Determines what new information to store in the cell state.
	$$
	\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
	$$
	$$
	\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
	$$
3. **Cell State Update ($\mathbf{c}_t$):** Combines retained past information and scaled new candidate information via linear additive operations:
	$$
	\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
	$$
4. **Output Gate ($\mathbf{o}_t$) & Hidden State ($\mathbf{h}_t$):** Controls what parts of the cell state make it to the output and next hidden state:
	$$
	\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
	$$
	$$
	\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
	$$
- **$\sigma(z) = \frac{1}{1 + e^{-z}}$**: Sigmoid function outputting values in $(0, 1)$ to act as soft binary gates.
- **$\odot$**: Hadamard (element-wise) product.
- **$[\mathbf{h}_{t-1}, \mathbf{x}_t] \in \mathbb{R}^{h + d}$**: Concatenation of the previous hidden state and the current input.

### Why LSTM Fixes Vanishing Gradients

The derivative of the cell state with respect to the previous cell state contains an additive component:

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t + \dots
$$

When the network learns to keep $\mathbf{f}_t \approx 1$, the gradient flows backward across time steps without exponential decay.

## 3\. Gated Recurrent Unit (GRU)

Proposed by Cho et al. (2014), the GRU simplifies the LSTM by merging the cell state and hidden state into a single vector $\mathbf{h}_t$ and reducing the number of gates from three to two.

### Gates and Operations

1. **Reset Gate ($\mathbf{r}_t$):** Determines how much of the previous state $\mathbf{h}_{t-1}$ to forget when computing the new candidate state.
	$$
	\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
	$$
2. **Update Gate ($\mathbf{z}_t$):** Acts simultaneously as both the forget gate and input gate, regulating how much of the past state to keep versus how much of the candidate state to incorporate.
	$$
	\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
	$$
3. **Candidate Hidden State ($\tilde{\mathbf{h}}_t$):** Computes new candidate information, using $\mathbf{r}_t$ to selectively mask past context.
	$$
	\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)
	$$
4. **Hidden State Update ($\mathbf{h}_t$):** Linear interpolation between the previous hidden state and the candidate hidden state governed by $\mathbf{z}_t$:
	$$
	\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
	$$
