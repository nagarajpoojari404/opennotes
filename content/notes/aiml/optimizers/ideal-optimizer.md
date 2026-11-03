---
title: "Ideal optimizer"
date: "2026-10-26"
---

An ideal optimizer must balance convergence rate, generalization performance, numerical stability, memory footprint, and compute overhead. In high-dimensional deep learning and LLM training, optimization is dominated by ill-conditioned curvature, stochastic gradient noise, and GPU hardware constraints.

The core characteristics that define an ideal optimizer include:

### 1\. Curvature Invariance (Handling Ill-Conditioned Surfaces)

- **The Problem:** Loss landscapes are heavily ill-conditioned ravines where the condition number of the Hessian $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ is massive ($10^4$ to $10^8$). Standard gradient descent oscillates violently along directions of high curvature while making near-zero progress along flat directions.
- **The Ideal Behavior:** The optimizer should adapt its effective step size per coordinate or per subspace according to the local curvature.
	- In second-order methods, pre-multiplying by $H^{-1}$ reshapes the anisotropic valley into an isotropic sphere.
		- In adaptive first-order methods (AdamW, RMSprop), dividing by $\sqrt{v_t}$ acts as a diagonal pre-conditioner approximating the inverse diagonal Hessian:
		$$
		\theta_{t+1} = \theta_t - \eta \, \text{diag}(H)^{-1/2} g_t
		$$
- **Modern Invariance:** Recent optimizers (like Muon) go further by enforcing **spectral norm invariance** via orthogonal updates, ensuring update steps do not privilege dominant singular vectors.

### 2\. Fast Convergence and High Sample Efficiency

- **Escape from Saddles and Plateaus:** Deep networks contain exponentially more saddle points than local minima. An ideal optimizer must maintain non-zero velocity through flat regions ($g_t \approx 0$) via momentum ($m_t$).
- **Lookahead and Damping:** It should incorporate predictive momentum (such as Nesterov acceleration) to damp oscillations when approaching a minimum, preventing overshooting and allowing higher learning rates early in training.

### 3\. Generalization to Unseen Data (Flat vs. Sharp Minima)

- **The Sharpness Problem:** An optimizer that converges rapidly to the nearest empirical minimum often lands in a **sharp minimum**—a steep valley where small shifts between the training set and validation set distribution cause drastic increases in loss.
- **The Ideal Behavior:** The optimizer should actively seek **flat minima** with wide basins, which generalize significantly better.
	- SGD with moderate batch sizes naturally discovers flat minima due to stochastic isotropic noise:
		$$
		\text{Cov}(\Delta \theta) \propto \frac{\eta}{B} \Sigma
		$$
		- Modern loss-aware optimizers (e.g., Sharpness-Aware Minimization / SAM) explicitly optimize for flatness:
		$$
		\min_\theta \max_{\Vert{}\epsilon\Vert{} \le \rho} L(\theta + \epsilon)
		$$

### 4\. Minimal Memory Footprint (Optimizer State Overhead)

- In large-scale training, model parameters are rarely the primary memory bottleneck; **the optimizer states are**:
	- FP32 Master Weights: $4 \text{ bytes / parameter}$
		- AdamW ($m_t + v_t$): $4 + 4 = 8 \text{ bytes / parameter}$
		- For a 70B parameter model, AdamW alone requires $\sim 560\text{ GB}$ of VRAM just to store state tensors.
- **The Ideal Metric:** The optimizer should achieve adaptive, second-order-like behavior while approaching **$0$ to $4$ bytes per parameter** in auxiliary memory:
	- Discarding the second moment entirely (Lion: $4\text{ bytes}$).
		- Factorizing 2D states into low-rank representations (Adafactor: $O(R + C)$ instead of $O(RC)$).
		- Block-wise non-linear 8-bit dynamic quantization (8-bit Adam: $2\text{ bytes}$).

### 5\. Scale & Reparameterization Invariance

- **Layer-Wise Scale Disparities:** Gradients in early transformer layers often differ in magnitude by multiple orders of magnitude compared to final projection heads.
- **The Ideal Behavior:** The update step should be invariant to affine re-scalings of the loss, the weights, or the activations:
	- If weights $W$ are scaled by a constant $c$, the effective trajectory should remain unchanged.
		- Optimizers like LARS and LAMB achieve this by scaling updates using layer-wise trust ratios:
		$$
		\Delta \theta^{(l)} \propto \frac{\Vert{}\theta^{(l)}\Vert{}}{\Vert{}r^{(l)}\Vert{}}
		$$
		- Lion enforces scale invariance at the coordinate level using the `sign` operator: $\Delta \theta \in \{-\eta, +\eta\}^d$.

### 6\. Robustness to Hyperparameter Variations

- **Hyperparameter Sensitivity:** Real-world training at scale cannot afford grid searches over learning rates, warmups, and decays due to multi-million-dollar compute budgets.
- **The Ideal Behavior:**
	- Wide stability plateaus for learning rate $\eta$.
		- Minimal reliance on hand-crafted warmups or ad-hoc learning rate decay schedules (e.g., cosine, WSD).
		- Insensitivity to the batch size $B$, smoothly scaling from small single-GPU batches to massive distributed clusters without divergence.

### 7\. Decoupled Regularization Mechanics

- **Proper Weight Decay:** An optimizer must separate parameter updates derived from gradient error signals from regularizing decay.
- **The Flaw in Classical Optimizers:** Coupling weight decay into the gradient $g_t + \lambda \theta_t$ within adaptive optimizers (as in standard Adam with $L_2$) causes heavily updated parameters to receive *less* weight decay than rarely updated ones.
- **The Requirement:** Decoupled weight decay (the AdamW formulation):
	$$
	\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \cdot \text{Update}(g_t)
	$$
	This guarantees a uniform rate of decay per step independent of gradient magnitude history.

### 8\. Hardware & Kernel Friendliness

- **Compute vs. Memory Bandwidth:** In modern GPU/TPU architectures, optimizer steps are bounded by memory bandwidth (HBM I/O), not arithmetic FLOPs.
- **The Ideal Optimizer:**
	- Avoids frequent non-local tensor communications or global reductions.
		- Can be implemented as a single **fused kernel** (combining parameter load, gradient load, state update, weight decay, and store in one sweep through GPU SRAM).
		- Minimizes transcendental function evaluations or uncoalesced memory reads across distributed ranks.

### Evaluation of Common Optimizers Against Ideal Criteria

| Criterion | SGD + Momentum | AdamW | Lion | Muon | Ideal Target |
| --- | --- | --- | --- | --- | --- |
| **Curvature Adaptation** | Poor | High (Diagonal) | Moderate | High (Spectral) | Full / Block Curvature |
| **Generalization (Flatness)** | Excellent | Moderate | Good | Excellent | Maximum |
| **Optimizer Memory** | 4 B/param | 8 B/param | 4 B/param | 4 B/param | $\le 4$ B/param |
| **Scale Invariance** | Low | High | Very High | High | Exact |
| **Batch Size Scalability** | Low (diverges large) | Moderate | High | High | Fully Invariant |
| **Kernel Fusion Ease** | High | High | Highest (`sign`) | Moderate (SVD/Newton) | High |
