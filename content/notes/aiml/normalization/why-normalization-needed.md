---
title: "Why normalization needed"
date: "2026-10-26"
---

Deep neural networks are stacks of non-linear matrix operations. Without normalization, training deep architectures quickly fails due to four fundamental mathematical and hardware phenomena.

### 1\. Vanishing and Exploding Gradients

Consider a simple linear chain of $L$ layers where activations propagate forward as $x_l = W_l x_{l-1}$. During backpropagation, the gradient of the loss with respect to the input follows the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=1}^L W_l^T
$$

- If the singular values of the weight matrices $W_l$ are strictly greater than $1$, the variance of activations grows exponentially with depth: $\text{Var}(x_L) \propto \sigma^{2L} \to \infty$ (**exploding gradients** / NaN loss).
- If the singular values are strictly less than $1$, the activations and gradient signals shrink exponentially to zero: $\text{Var}(x_L) \to 0$ (**vanishing gradients** / stalled training).

Normalization resets the activation scale at each layer to unit variance ($\approx 1$), preventing activations and gradients from compounding uncontrollably across 30 to 100+ layers.

### 2\. Loss Landscape Smoothing

In unnormalized networks, the loss surface is dominated by ill-conditioned curvature. The Hessian matrix $H = \nabla^2 \mathcal{L}$ of the loss with respect to parameters exhibits a huge condition number:

$$
\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} \gg 1
$$

- When $\kappa(H)$ is large, the loss landscape forms an elongated, narrow ravine: gradients bounce wildly across steep ravine walls while making negligible progress along the flat base.
- Normalization bounds the Lipschitz constant of the loss gradient:

$$
\Vert{}\nabla \mathcal{L}(x_1) - \nabla \mathcal{L}(x_2)\Vert{} \le L_g \Vert{}x_1 - x_2\Vert{}
$$

By bounding $L_g$ and balancing the eigenvalues of the Hessian ($\kappa(H) \approx 1$), the loss surface becomes isotropic (bowl-shaped). This allows first-order optimizers (like AdamW or SGD) to use significantly higher learning rates without diverging.

### 3\. Preventing Activation Saturation in Non-Linearities

Many classic and contemporary non-linear activation functions (e.g., Sigmoid, Tanh, or even GELU/SwiGLU in extreme tails) have saturating regimes where the derivative approaches zero:

$$
\sigma'(z) \to 0 \quad \text{as} \quad \vert{}z\vert{} \gg 0
$$

If incoming activations shift into large absolute values without normalization, neurons enter permanent saturation. Their local gradients $\frac{\partial a}{\partial z} \approx 0$ effectively kill backpropagation through those pathways. Keeping activations centered around zero with standard deviation near 1 ensures inputs stay within the active, non-saturating dynamic range.

### 4\. Mitigating Internal Covariate Shift

During training, all layers update their weights simultaneously. As layer $l$ modifies its weights, the distribution of its outputs changes. From the perspective of layer $l+1$, the input distribution is a continuously moving target.

By normalizing intermediate tensor outputs, each layer receives inputs with a stable, predictable mean and variance, decoupling a layer's input statistics from parameter oscillations in preceding layers.

### 5\. Hardware Precision Constraints (bfloat16 and float16)

Modern deep learning architectures and LLMs are trained in lower-precision floating-point formats to maximize tensor core compute throughput:

| Precision Format | Exponent Bits | Mantissa Bits | Dynamic Range (Max Magnitude) | Smallest Positive Normal |
| --- | --- | --- | --- | --- |
| **float32** | 8 | 23 | $\approx 3.4\times10^{38}$ | $\approx 1.18\times10^{-38}$ |
| **bfloat16** | 8 | 7 | $\approx 3.39\times10^{38}$ | $\approx 1.17\times10^{-38}$ |
| **float16** | 5 | 10 | $\approx 65{,}504$ | $\approx 6.10\times10^{-5}$ |

- In **float16**, any intermediate activation value exceeding $65,504$ results in an immediate hardware overflow (`+Inf` / `NaN`).
- Even in **bfloat16** (which preserves float32 exponent range), limited mantissa bits (7 bits $\approx$ 3 decimal digits of precision) mean large magnitude disparities between activation channels cause catastrophic cancellation and underflow of smaller updates.

Normalization explicitly bounds the activation magnitudes at every residual block, keeping tensors well within the stable numeric bounds of 16-bit and 8-bit hardware formats.
