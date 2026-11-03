---
title: "Optimizers"
date: "2026-10-26"
---

Optimizers update neural network parameters $\theta$ to minimize a scalar loss function $L(\theta)$. They determine the trajectory across the non-convex loss landscape by estimating step sizes, directions, momentum, and curvature corrections.

They fall into five main families: **first-order stochastic gradient methods**, **adaptive learning rate methods**, **second-order & quasi-Newton methods**, **large-batch scale optimizers**, and **modern memory-efficient LLM optimizers**.

## 1\. Classical First-Order Methods

These rely directly on the first derivative (gradient) $g_t = \nabla_\theta L(\theta_t)$.

### Stochastic Gradient Descent (SGD)

Updates parameters in the opposite direction of the gradient evaluated on a mini-batch:

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

- **Mechanism:** Computes unbiased but high-variance estimates of the full-batch gradient.
- **Limitations:** Oscillates wildly across narrow ravines (ill-conditioned curvature) and easily stalls at saddle points or flat plateaus where $\Vert{}g_t\Vert{} \approx 0$.

### SGD with Polyak Momentum

Accelerates SGD along consistent gradient directions while damping high-frequency oscillations:

$$
v_{t+1} = \beta v_t + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

- **Mechanism:** Accumulates an exponentially decaying moving average of past gradients ($v_t$), acting like a heavy ball rolling down a potential well. Typical setting is $\beta = 0.9$.

### Nesterov Accelerated Gradient (NAG)

Computes the gradient at an anticipated future position rather than the current position ("look-ahead" momentum):

$$
g_t = \nabla_\theta L(\theta_t - \beta \eta v_t)
$$

$$
v_{t+1} = \beta v_t + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

- **Mechanism:** Acts as an adaptive predictive brake. If momentum is carrying the parameters uphill, the look-ahead gradient detects the ascent early and reduces momentum before overshooting the basin.

## 2\. Adaptive Learning Rate Methods

These scale the learning rate per parameter based on historical gradient magnitudes, automatically giving frequent/large gradients smaller steps and infrequent/small gradients larger steps.

### AdaGrad (Adaptive Gradient Algorithm)

Accumulates the sum of squared historical gradients coordinate-wise:

$$
G_{t} = G_{t-1} + g_t \odot g_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t
$$

- **Mechanism:** Scales down learning rates for frequently occurring features, making it well-suited for sparse data (text TF-IDF, embeddings).
- **Failure Mode:** $G_t$ monotonically increases with every step ($G_t \ge G_{t-1}$). The effective step size decays to zero prematurely, permanently freezing training before convergence.

### RMSprop (Root Mean Square Propagation)

Fixes AdaGrad's premature freezing by replacing the cumulative sum with an exponentially decaying moving average:

$$
v_t = \beta v_{t-1} + (1 - \beta) (g_t \odot g_t)
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t
$$

- **Mechanism:** Limits gradient history to an effective window of size $\frac{1}{1 - \beta}$ (typically $\beta = 0.99$). Retains the ability to adjust step sizes indefinitely. Standard in early reinforcement learning (DQN).

### Adam (Adaptive Moment Estimation)

Combines **Polyak Momentum** (first moment) with **RMSprop** (second raw moment) and includes analytical bias corrections:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{first moment estimate})
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t \odot g_t) \quad (\text{second moment estimate})
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad (\text{bias corrections for } t=1, 2, \dots)
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
$$

- **Mechanism:** Defaults are standard ($\beta_1 = 0.9, \beta_2 = 0.999 \text{ or } 0.95, \epsilon = 10^{-8}$). Bias correction accounts for initialization of $m_0 = 0, v_0 = 0$, preventing tiny denominators and unstable early steps.

### AdamW (Decoupled Weight Decay)

Fixes the mathematical flaw in applying $L_2$ regularization inside adaptive optimizers (Loshchilov & Hutter, 2017):

In standard Adam with $L_2$ regularization, the gradient becomes $g_t + \lambda \theta_t$. This means weight decay is scaled inversely by $\sqrt{v_t}$: weights that receive large, frequent gradients get decayed *less* than weights with infrequent gradients.

**AdamW decouples weight decay entirely from the adaptive gradient update:**

$$
\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
$$

- **Status:** The universal standard optimizer for Transformer and LLM pre-training (GPT, Llama, Mistral, Gemma).

### Adafactor

Reduces the memory footprint of Adam's second moment buffer $v_t$ for large models (Shazeer & Stern, 2018):

- Instead of storing an $N \times M$ matrix of squared gradients for a linear layer, it maintains **row-wise and column-wise factorized sums**:
	$$
	R_t \in \mathbb{R}^N, \quad C_t \in \mathbb{R}^M
	$$
	Reconstructing the denominator as $\hat{V}_t = \frac{R_t C_t^T}{\mathbf{1}^T R_t}$.
- **Memory Savings:** Reduces second-moment state from $O(NM)$ to $O(N + M)$, cutting optimizer memory by more than half. Commonly used with an updated decay schedule and no first-moment buffer (e.g., T5, PaLM).

## 3\. Large-Batch Distributed Optimizers

When scaling LLM training to thousands of GPUs, mini-batch sizes grow to tens of thousands of tokens or images. Large batches cause standard adaptive optimizers to diverge due to layer-wise gradient scale disparities.

### LARS (Layer-wise Adaptive Rate Scaling)

Designed for massive-batch vision training (e.g., ResNet on ImageNet with batch size $> 32\text{k}$):

$$
\eta_t^{(l)} = \eta \cdot \phi\left( \frac{\Vert{}\theta^{(l)}\Vert{}}{\Vert{}g^{(l)}\Vert{} + \beta \Vert{}\theta^{(l)}\Vert{}} \right)
$$

- **Mechanism:** Computes a separate local learning rate per layer $l$ based on the ratio between weight norm $\Vert{}\theta^{(l)}\Vert{}$ and gradient norm $\Vert{}g^{(l)}\Vert{}$. Prevents unstable layers from dominating the network update.

### LAMB (Layer-wise Adaptive Moments for Batch Training)

Combines LARS's layer-wise trust ratio with AdamW's coordinate-wise adaptive moments:

$$
r_t^{(l)} = \frac{\hat{m}_t^{(l)}}{\sqrt{\hat{v}_t^{(l)}} + \epsilon} + \lambda \theta_t^{(l)}
$$

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot \frac{\Vert{}\theta^{(l)}\Vert{}}{\Vert{}r_t^{(l)}\Vert{}} \odot r_t^{(l)}
$$

- **Mechanism:** Enables stable training with massive batch sizes (e.g., scaling BERT pre-training from batch size 512 to 65,536 without loss of accuracy).

## 4\. Modern LLM & Memory-Efficient Optimizers

As models scaled beyond 100B parameters, optimizer memory became the primary training bottleneck: AdamW in FP32 requires **8 bytes per parameter** (4 bytes for $m_t$, 4 bytes for $v_t$) on top of master weights.

### 8-Bit Adam (bitsandbytes)

Quantizes the first and second moment tensors ($m_t, v_t$) dynamically from 32-bit floats into custom 8-bit non-linear representations (Dettmers et al., 2021):

- **Mechanism:** Uses block-wise dynamic quantization (typically 256 elements per block) that non-linearly allocates more quantization bins to values closer to zero.
- **Memory Impact:** Drops optimizer state overhead from 8 bytes/parameter down to 2 bytes/parameter with near-zero degradation in perplexity.

### Lion (EvoLved Sign Momentum)

Discovered via program search by Google Brain (Chen et al., 2023):

$$
c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
\theta_{t+1} = \theta_t - \eta \left( \text{sign}(c_t) + \lambda \theta_t \right)
$$

$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$

- **Mechanism:**
	- Discards the second-moment tensor $v_t$ completely—reducing optimizer states by **50%** compared to AdamW (stores only $m_t$).
		- Uses the `sign` operator ($\pm 1$), applying a uniform step size across every parameter like a continuous coordinate-descent method.
		- More compute-efficient and memory-efficient than AdamW, though sensitive to batch size and learning rate tuning.

### Muon (Momentum Orthogonalized by Newton-schulz)

A matrix-valued optimizer developed for the hidden linear weights of deep neural networks:

- **Mechanism:** Instead of applying elementwise updates or simple normalization, Muon treats 2D weight matrices $W$ as linear operators. It takes the momentum matrix $M$ and computes an approximate matrix polar decomposition using an iterative **Newton-Schulz** method:
	$$
	O = \text{NewtonSchulz}(M)
	$$
	where $O$ is an orthogonal matrix minimizing $\Vert{}O - M\Vert{}_F$.
- **Properties:** Ensures the spectral norm of the update is strictly bounded ($\Vert{}O\Vert{}_2 \approx 1$), providing uniform curvature-free updates across singular vectors. Used in high-performance LLM training runs (e.g., Keller Jordan / Nanotron community benchmarks) to achieve faster convergence per step than AdamW.

## 5\. Second-Order & Hessian-Based Optimizers

These incorporate second derivatives to adjust step size and direction according to the local curvature of the loss surface.

### Newton-Raphson Method

Uses the exact inverse Hessian matrix $H = \nabla^2 L(\theta)$:

$$
\theta_{t+1} = \theta_t - H^{-1} g_t
$$

- **Mechanism:** Computes the step directly to the minimum of a local quadratic approximation.
- **Limitation:** Infeasible for deep learning. Computing and storing $H$ requires $O(P^2)$ memory, and inverting it takes $O(P^3)$ time ($P$ is parameter count).

### BFGS & L-BFGS (Limited-memory BFGS)

Quasi-Newton methods that approximate the inverse Hessian $H^{-1}$ using rank-2 updates derived from successive gradient vectors:

$$
s_t = \theta_{t+1} - \theta_t, \quad y_t = g_{t+1} - g_t
$$

- **L-BFGS:** Avoids storing the full $P \times P$ matrix by retaining only the last $m$ pairs of vectors $\{s_i, y_i\}_{i=t-m}^t$ (typically $m \in [5, 20]$).
- **Limitations:** Highly sensitive to stochastic mini-batch noise. Primarily used in deterministic, full-batch optimization regimes (e.g., style transfer, physics-informed neural networks / PINNs, small linear models).

### K-FAC (Kronecker-Factored Approximate Curvature)

A scalable second-order method that approximates the **Fisher Information Matrix** (which matches the Hessian under negative log-likelihood):

- **Mechanism:** For a linear layer $W$, the gradient is the outer product of activations $a$ and backpropagated derivatives $s$: $g = s a^T$. K-FAC approximates the covariance $\mathbb{E}[g g^T] = \mathbb{E}[(s a^T) \otimes (s a^T)]$ by factoring it into the Kronecker product of two smaller matrices:
	$$
	F_l \approx \mathbb{E}[a a^T] \otimes \mathbb{E}[s s^T] = A \otimes S
	$$
- **Inversion:** Uses the identity $(A \otimes S)^{-1} = A^{-1} \otimes S^{-1}$. Instead of inverting an $NM \times NM$ matrix, it inverts an $N \times N$ matrix and an $M \times M$ matrix independently, making true curvature-aware updates tractable for deep architectures.

## Optimizer Selection Guide

| Optimizer | State Memory Overhead | Primary Use Case | Key Strength |
| --- | --- | --- | --- |
| **SGD + Momentum** | 4 bytes / param ($v$) | Computer Vision (ResNets, ConvNets) | High generalization capability; robust to noise |
| **AdamW** | 8 bytes / param ($m,v$) | Transformers, LLMs, Multimodal models | Stable, fast convergence across diverse architectures |
| **Adafactor** | $\sim 0$ to $4$ bytes / param | Memory-constrained LLMs (T5) | Eliminates/compresses second-moment memory |
| **Lion** | 4 bytes / param ($m$) | Large-scale vision & language | 50% less optimizer memory than AdamW; uniform step size |
| **LAMB** | 8 bytes / param ($m,v$) | Ultra-large batch distributed training | Layer-wise scaling prevents divergence at large batch size |
| **Muon** | 4 bytes / param ($m$) | Matrix layers in modern LLMs | Spectral orthogonalization leads to faster wall-clock convergence |
| **L-BFGS** | $2m$ vectors | PINNs, full-batch non-stochastic setups | Fast superlinear convergence without manual learning rates |
