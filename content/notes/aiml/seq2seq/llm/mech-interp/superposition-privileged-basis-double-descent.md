---
title: "Superposition, privileged basis, double descent"
date: "2026-10-27"
---

### 1\. Basis & Basis-Free Operations

In mechanistic interpretability, operations within a Transformer are categorized by whether their behavior depends on the coordinate system chosen to represent the vector space:

- **Basis-Free (Rotational Invariance)**: An operation where rotating the vector space by an orthogonal matrix $R$ ($R^\top R = I$) preserves the overall transformation, provided the weights are rotated accordingly.
- **Privileged Basis**: An operation where individual coordinate axes (neurons) carry distinct mathematical and semantic meaning that cannot be rotated without breaking the computation.

#### The Residual Stream is Basis-Free

The residual stream vector $x \in \mathbb{R}^{d_{\text{model}}}$ does not have an intrinsic coordinate system. If you rotate the entire residual stream by an orthogonal transformation $R$:

$$
x' = R x
$$

The linear projections of an attention head ($W_Q, W_K, W_V, W_O$) adapt simply by multiplying by $R^\top$:

$$
W'_Q = W_Q R^\top, \quad W'_K = W_K R^\top, \quad W'_V = W_V R^\top, \quad W'_O = R W_O
$$

The internal dot products are preserved:

$$
(W'_Q x')^\top (W'_K x') = (W_Q R^\top R x)^\top (W_K R^\top R x) = (W_Q x)^\top (W_K x)
$$

Similarly, layer normalization (up to minor coordinate scaling) and linear read/write operations treat all directions symmetrically. There is no special meaning to the $k$\-th coordinate of the residual stream.

#### MLPs Introduce a Privileged Basis

The MLP blocks break rotational symmetry because activation functions ($\text{ReLU}$, $\text{GELU}$, $\text{SwiGLU}$) are applied **element-wise**:

$$
\text{MLP}(x) = W_{\text{out}} \ \sigma(W_{\text{in}} x + b_{\text{in}})
$$

For an arbitrary orthogonal matrix $R$, non-linearities generally do not commute with rotation:

$$
\sigma(R z) \ne R \ \sigma(z)
$$

Because $\sigma$ evaluates each scalar component along the standard coordinate axes independently:

1. The hidden layer of an MLP has a **privileged basis**: the coordinate axes correspond directly to individual neurons.
2. In contrast, the residual stream lacks a privileged basis, making it harder to assign individual dimensions to discrete features without techniques like Dictionary Learning / Sparse Autoencoders (SAEs).

### 2\. Superposition

**Superposition** is the phenomenon where a neural network represents significantly more independent features than it has dimensions ($M \gg d$).

#### The Geometric Mechanism

If features were strictly orthogonal, a $d$\-dimensional space could represent at most $d$ features without mutual interference. However, in high dimensions, an exponentially large number of vectors can be **almost orthogonal** (a consequence of the Johnson-Lindenstrauss lemma and high-dimensional sphere packing):

$$
\langle v_i, v_j \rangle \approx 0 \quad \text{for } i \ne j
$$

When a model maps $M$ features $\{f_i\}_{i=1}^M$ with activations $a_i \ge 0$ into a $d$\-dimensional space via unit feature directions $v_i \in \mathbb{R}^d$:

$$
x = \sum_{i=1}^M a_i v_i
$$

To decode feature $k$, we project $x$ onto $v_k$:

$$
\hat{a}_k = \langle v_k, x \rangle = a_k + \sum_{i \ne k} a_i \langle v_k, v_i \rangle
$$

The second term $\sum_{i \ne k} a_i \langle v_k, v_i \rangle$ acts as **interference noise** (crosstalk).

#### Why Superposition Works: Sparsity

Superposition is computationally viable when real-world features are **sparse**—meaning only a tiny fraction of features $a_i > 0$ fire simultaneously for any single input:

1. If most $a_i = 0$, the sum of interference noise remains small.
2. Non-linear thresholding (e.g., $\text{ReLU}$) can clear out small interference ripples:
	$$
	\text{ReLU}(\hat{a}_k - \tau) \approx a_k
	$$

#### Polysemanticity

Superposition directly causes **polysemanticity**: because feature directions are not aligned with standard basis vectors, individual neurons (the basis vectors) end up projecting onto multiple unrelated feature directions. Inspecting a single neuron activation reveals that it activates for multiple distinct, unrelated concepts.

### 3\. Double Descent

**Double descent** is a generalization phenomenon that reconciles classical statistical learning theory with the empirical success of deep, overparameterized neural networks.

       ```
Risk / Test Error
          |
          |       Underfitting    Interpolation Peak
          |            \                 /\
          |             \               /  \
          |  Classical   \  Overfitting/    \     Modern Overparameterized
          |    U-Curve    \   Spike   /      \           Regime
          |                \         /        \------------------
          |                 \_______/
          +------------------------------------------------------------>
                                             Capacity / Parameters (p)
                                  p = n


#### Classical Regime ($p < n$)

- In classical statistics, as model capacity (number of parameters $p$) increases relative to the number of training samples $n$, bias decreases while variance increases.
- The test error exhibits the standard **U-shaped curve**, reaching an optimal trade-off point and then increasing due to overfitting as $p \to n$.

#### The Interpolation Threshold ($p = n$)

- At the **interpolation boundary** where model capacity just barely matches the training set size, the model fits the training data and noise with zero training loss.
- To fit noisy training labels exactly without excess capacity, the model must contort its decision boundary violently (large norm solutions), driving the variance to its maximum. This causes a pronounced **spike in test error**.

#### Modern Overparameterized Regime ($p \gg n$)

- When capacity extends well beyond the interpolation point ($p \gg n$), there are infinitely many parameter configurations that achieve zero training error.
- Optimizers like Stochastic Gradient Descent (SGD) implicitly regularize the solution by selecting the interpolating function with the **minimum norm** (e.g., minimum $L_2$ norm in weights):
	$$
	w^* = \arg\min_w \Vert{}w\Vert{} \quad \text{s.t.} \quad Xw = y
	$$
- As parameter count increases further, the space of zero-loss interpolators expands, allowing SGD to find smoother, lower-norm functions that interpolate the training points while generalizing effectively.

Double descent manifests across three primary dimensions:

1. **Model-wise Double Descent**: Varying the width or depth of the architecture.
2. **Epoch-wise Double Descent**: Prolonged training where error drops, rises, and drops again.
3. **Sample-wise Double Descent**: Varying dataset size $n$ for a fixed model size $p$.
