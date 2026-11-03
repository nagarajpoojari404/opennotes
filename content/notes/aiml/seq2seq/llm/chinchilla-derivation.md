---
title: "Chinchilla derivation"
date: "2026-10-30"
---

Here is a clean, rigorous step-by-step mathematical derivation of why scaling laws take the form of power laws, and why the **1:1 Chinchilla ratio** emerges theoretically.

## Part 1: Why Loss Follows a Power Law ($L \propto N^{-\alpha}$)

The power-law relationship between parameters $N$ and loss $L$ can be derived directly from **manifold geometry** and **kernel spectral theory**.

### 1\. The Geometric Cover Argument

Assume human language exists on a $d$\-dimensional manifold (surface) embedded within a higher-dimensional vector space.

1. **Space Partitioning:** A neural network with $N$ parameters can partition this $d$\-dimensional manifold into $N$ distinct region cells (or "boxes") to approximate the true language distribution.
2. **Cell Radius:** The volume of the entire manifold is constant. Dividing it into $N$ equal cells means each cell has a volume $V_{\text{cell}} \propto \frac{1}{N}$.
3. **Linear Characteristic Length ($\epsilon$):** Since volume in $d$ dimensions scales as length to the $d$\-th power ($V \propto \epsilon^d$), the average linear diameter $\epsilon$ of each cell is:
	$$
	\epsilon \propto \left(\frac{1}{N}\right)^{\frac{1}{d}} = N^{-\frac{1}{d}}
	$$
4. **Approximation Loss:** For smooth functions, the local error (loss) inside a cell of size $\epsilon$ scales quadratically or linearly with the cell size $\epsilon$ (via Taylor expansion):
	$$
	L(N) - L_{\infty} \propto \epsilon^\beta
	$$
	where $L_{\infty}$ is the irreducible entropy of language, and $\beta \ge 1$ depends on the smoothness of the model.
5. **The Power-Law Exponent:** Substituting $\epsilon \propto N^{-1/d}$ into the error equation gives:
	$$
	L(N) - L_{\infty} \propto \left(N^{-\frac{1}{d}}\right)^\beta = N^{-\frac{\beta}{d}}
	$$

Setting $\alpha_N = \frac{\beta}{d}$ yields the empirical parameter scaling law:

$$
L(N) - L_{\infty} = A \cdot N^{-\alpha_N}
$$

> **Key Takeaway:** The scaling exponent $\alpha_N$ reflects the ratio of network smoothness ($\beta$) to the **intrinsic dimensionality ($d$)** of natural language. Because text has a high intrinsic dimension $d$, $\alpha_N$ is a small fraction (typically $0.05$ to $0.34$).

## Part 2: Formulating the Joint Loss Function $L(N, D)$

When training with both finite parameters ($N$) and finite tokens ($D$), the model faces two independent sources of error:

1. **Representation Error ($A \cdot N^{-\alpha}$):** Bottleneck from having too few parameters to store patterns.
2. **Estimation Error ($B \cdot D^{-\beta}$):** Bottleneck from seeing too few data samples to estimate parameters accurately (variance from Stochastic Gradient Descent).

Assuming these two errors are additive alongside the irreducible loss ($E$), the joint loss equation formulated in the Chinchilla paper is:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

- $E$: Irreducible loss (entropy of true natural language).
- $A, B$: Constants related to data distribution scale.
- $\alpha, \beta$: Empirical exponents governing parameter and data efficiency ($\alpha \approx 0.34$, $\beta \approx 0.28$).

## Part 3: Deriving the 1:1 Compute-Optimal Scaling (Chinchilla)

To find how to allocate parameters $N$ and dataset size $D$ for a fixed compute budget $C$, we use constrained optimization.

### 1\. The Compute Constraint

Training a Transformer model with $N$ parameters on $D$ tokens requires approximately 6 floating-point operations (FLOPs) per parameter per token (3 FLOPs for forward pass, 3 FLOPs for backward pass):

$$
C = 6ND
$$

### 2\. Setting Up the Optimization Problem

We want to minimize the reducible loss subject to a fixed compute budget $C_0$:

$$
\min_{N, D} \left( \frac{A}{N^\alpha} + \frac{B}{D^\beta} \right) \quad \text{subject to} \quad 6ND = C_0
$$

Substitute $D = \frac{C_0}{6N}$ directly into the loss function to convert it into a single-variable function of $N$:

$$
f(N) = A N^{-\alpha} + B \left( \frac{C_0}{6N} \right)^{-\beta} = A N^{-\alpha} + B \cdot (6/C_0)^\beta \cdot N^\beta
$$

### 3\. Taking the Derivative

To find the optimal parameter count $N^*$, set the derivative $\frac{df(N)}{dN} = 0$:

$$
\frac{df}{dN} = -\alpha A N^{-\alpha - 1} + \beta B (6/C_0)^\beta N^{\beta - 1} = 0
$$

Equating the two terms:

$$
\alpha A N^{-\alpha - 1} = \beta B \left(\frac{6}{C_0}\right)^\beta N^{\beta - 1}
$$

### 4\. Solving for Optimal $N^*$ and $D^*$ in Terms of Compute ($C_0$)

Rearrange the terms to group $N$ on one side:

$$
\frac{N^{\beta - 1}}{N^{-\alpha - 1}} = N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \cdot \left(\frac{C_0}{6}\right)^\beta
$$

Taking both sides to the power of $\frac{1}{\alpha + \beta}$:

$$
N^* = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}} \left(\frac{C_0}{6}\right)^{\frac{\beta}{\alpha + \beta}}
$$

By symmetry, repeating this for $D^* = \frac{C_0}{6N^*}$ yields:

$$
D^* = \left( \frac{\beta B}{\alpha A} \right)^{\frac{1}{\alpha + \beta}} \left(\frac{C_0}{6}\right)^{\frac{\alpha}{\alpha + \beta}}
$$

## Part 4: Plugging in Empirical Exponents ($\alpha \approx \beta$)

Hoffmann et al. (2022) measured the exponent values empirically across 400+ runs and found that $\alpha$ and $\beta$ are remarkably close in value:

- $\alpha \approx 0.34$
- $\beta \approx 0.28$

Taking the balanced approximation where $\alpha \approx \beta$:

$$
\frac{\beta}{\alpha + \beta} \approx \frac{\alpha}{\alpha + \beta} \approx \frac{1}{2} = 0.5
$$

Substituting this back into our optimal $N^*$ and $D^*$ solutions:

$$
N^* \propto C_0^{0.5} = \sqrt{C_0}
$$

$$
D^* \propto C_0^{0.5} = \sqrt{C_0}
$$

### Conclusion

$$
\frac{N^*}{D^*} = \text{Constant} \implies D^* \propto N^*
$$

Because both optimal parameters ($N^*$) and optimal tokens ($D^*$) scale proportional to $\sqrt{C_0}$, **when compute $C$ is scaled by a factor of $k$, both model parameters $N$ and training tokens $D$ must be scaled by $\sqrt{k}$ (a $1:1$ ratio).**
