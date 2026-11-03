---
title: "Muon"
date: "2026-10-31"
---

**Muon** (Momentum Orthogonalized by Newton-schulz) is an optimizer tailored for 2D weight matrices (linear and dense layers) in neural networks. Where standard SGD operates in Euclidean/Frobenius geometry and Adam normalizes coordinate-wise (inducing an $\ell_\infty$\-like update), Muon treats weights as linear operators and restricts updates under the **spectral norm**.

### 1\. The Variational Formulation: Steepest Descent on Spectral Ball

Let a weight matrix be $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and let $G = \nabla_W \mathcal{L}$ be its objective gradient (or momentum buffer).

We seek an update $\Delta W$ that maximizes descent in the first-order Taylor expansion while constraining the **spectral norm** $\Vert{}\Delta W\Vert{}_2 \le 1$ (the maximum operator stretch):

$$
\Delta W^* = \arg\min_{\Vert{}\Delta W\Vert{}_2 \le 1} \langle G, \Delta W \rangle = \arg\min_{\Vert{}\Delta W\Vert{}_2 \le 1} \operatorname{Tr}(G^\top \Delta W)
$$

#### Derivation via Singular Value Decomposition (SVD)

Let the singular value decomposition of $G$ be:

$$
G = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i u_i v_i^\top, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0
$$

By Von Neumann’s trace inequality and Hölder's inequality for Schatten norms (since $\Vert{}\cdot\Vert{}_2 = \Vert{}\cdot\Vert{}_{S_\infty}$ is dual to the nuclear norm $\Vert{}\cdot\Vert{}_* = \Vert{}\cdot\Vert{}_{S_1}$):

$$
\operatorname{Tr}(G^\top \Delta W) \ge -\sum_{i=1}^{r} \sigma_i(G) \sigma_i(\Delta W)
$$

Because $\Vert{}\Delta W\Vert{}_2 \le 1$, each singular value $\sigma_i(\Delta W) \le 1$. The lower bound is achieved when all $\sigma_i(\Delta W) = 1$ aligned along the singular vectors of $G$:

$$
\Delta W^* = - U V^\top
$$

$U V^\top$ is the polar factor (orthogonal projection) of $G$. Instead of letting a few large singular values dominate step sizes, Muon flattens all non-zero singular values to $1$, allowing updates along all active spectral directions equally.

### 2\. Fast Orthogonalization via Newton–Schulz Iteration

Computing the exact SVD via Golub-Kahan per step is $O(\min(d_{\text{in}}, d_{\text{out}})^3)$, which is prohibitively slow on GPUs. Muon approximates $UV^\top$ using a polynomial **Newton–Schulz iteration** consisting purely of matrix multiplications.

Assume without loss of generality that $X \in \mathbb{R}^{n \times m}$ is wide ($n \le m$; if tall, transpose first).

1. **Initial Frobenius normalization:**
	$$
	X_0 = \frac{G}{\Vert{}G\Vert{}_F + \epsilon}
	$$
	Since $\Vert{}G\Vert{}_F = \sqrt{\sum \sigma_i^2} \ge \sigma_{\max}(G)$, this guarantees that every singular value $\sigma_i(X_0) \le 1$.
2. **Odd polynomial iteration (Quintic form):**
	$$
	X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k
	$$
	Evaluating on singular values, this applies a 1D mapping $\phi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ to every singular value while leaving the singular vectors $U, V$ unchanged.
	With tuned coefficients:
	$$
	a = 3.4445, \quad b = -4.7750, \quad c = 2.0315
	$$
	The function $\phi(\sigma)$ rapidly drives all $\sigma \in (0, 1]$ toward the fixed point $\sigma^* = 1$ in typically $5$ iterations:
	$$
	O = \text{NewtonSchulz}(X_0, k=5) \approx U V^\top
	$$

### 3\. Complete Muon Update Algorithm

Muon integrates standard Nesterov momentum and adjusts for rectangular matrix geometries:

1. **Momentum accumulation:**
	$$
	M_t = \beta M_{t-1} + (1 - \beta) G_t
	$$
2. **Nesterov blend:**
	$$
	\widetilde{M}_t = (1 - \beta) G_t + \beta M_t
	$$
3. **Newton–Schulz orthogonalization:**
	$$
	O_t = \text{NewtonSchulz}(\widetilde{M}_t)
	$$
4. **Aspect ratio scaling:** For an $n \times m$ matrix, an orthogonal matrix has entries scaling as $O(1/\sqrt{\max(n, m)})$. To maintain uniform Root-Mean-Square (RMS) parameter update magnitudes across arbitrary layer dimensions, the update is scaled by:
	$$
	\gamma = \sqrt{\max\left(1, \frac{n}{m}\right)}
	$$
5. **Weight update with decoupled decay:**
	$$
	W_{t+1} = W_t - \eta_t \left( \gamma O_t + \lambda W_t \right)
	$$

### Comparison: Why Muon Differs from Adam and SGD

| Feature | SGD | Adam / AdamW | Muon |
| --- | --- | --- | --- |
| **Geometry** | Euclidean ($\ell_2$) norm | Coordinate-wise ($\ell_\infty$) norm | Matrix Operator (Spectral / Schatten-$\infty$) |
| **Effective Update** | $G$ | $\operatorname{sign}(G)$ (entrywise) | $UV^\top$ (orthogonal polar factor) |
| **Singular Value Handling** | Propagates $\Sigma$ directly | Distorts directions entrywise | Equalizes all non-zero $\sigma_i\to 1$ |
| **Optimizer States** | 1 buffer ($M$) | 2 buffers ($M,V$) | 1 buffer ($M$) |
| **Application Domain** | General | All layers (embeddings, biases, 1D) | Exclusively 2D hidden linear/attention matrices |
