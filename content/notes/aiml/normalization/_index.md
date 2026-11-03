---
title: "Normalization"
weight: 5
date: "2026-10-26"
---

Normalization techniques stabilize deep network training by controlling internal covariate shift, smoothing the loss landscape, and maintaining well-behaved gradient flows.

In tensor terminology, consider activations shaped as $(N, C, H, W)$ for computer vision (Batch, Channel, Height, Width) or $(B, T, D)$ for sequence models and LLMs (Batch, Sequence/Token, Hidden Dimension). The core difference across techniques is **which dimensions are pooled to compute statistics**.

### 1\. Batch Normalization (BatchNorm)

BatchNorm computes statistics across the batch dimension and spatial/temporal positions for each individual channel or feature.

$$
\mu_c = \frac{1}{N \times H \times W} \sum_{n, h, w} x_{n, c, h, w}, \quad \sigma_c^2 = \frac{1}{N \times H \times W} \sum_{n, h, w} (x_{n, c, h, w} - \mu_c)^2
$$

$$
\hat{x} = \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

- **Pooling axes:** $(N, H, W)$ or $(B, T)$ per channel $C$ or feature $D$.
- **Characteristics:** Tracks running empirical means and variances during training to use as fixed constants during inference.
- **Why it fails in LLMs/RNNs:**
	1. Sequence lengths vary across batches, causing padding artifacts in batch statistics.
		2. Autoregressive inference generates one token at a time ($B=1$), where running statistics often mismatch generation dynamics.
		3. Small per-device batch sizes in distributed LLM training introduce high variance into gradient updates.

### 2\. Layer Normalization (LayerNorm)

LayerNorm computes statistics across the feature dimensions for each individual sample independently.

For a hidden state vector $x \in \mathbb{R}^D$ at batch item $b$ and token position $t$:

$$
\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i
$$

- **Pooling axes:** Feature/channel dimension $D$ (per sample and per token position).
- **Characteristics:** Zero batch dependency. Training and inference compute identical statistics on the fly.
- **LLM Role:** The default normalization for original Transformers (Vaswani et al., BERT, GPT-2/GPT-3).

### 3\. Root Mean Square Normalization (RMSNorm)

RMSNorm hypothesizes that the re-centering property (subtracting the mean $\mu$) in LayerNorm is non-essential; the primary stabilizing effect comes from scaling by the magnitude.

$$
\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}
$$

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(x)}, \quad y_i = \gamma_i \hat{x}_i
$$

- **Characteristics:** Removes mean computation $\mu$ and the learnable bias parameter $\beta$.
- **Computational Benefit:** Saves two passes over the hidden dimension vector in memory-bound GPU kernels (avoids mean accumulation and subtraction), improving kernel throughput by 10%–50% in standard fused operators.
- **LLM Role:** The industry standard for modern open-weights and frontier LLMs (LLaMA 1/2/3, Mistral, Gemma, Qwen, DeepSeek).

### 4\. Instance Normalization (InstanceNorm)

InstanceNorm normalizes across spatial dimensions $(H, W)$ independently for each batch item and channel.

$$
\mu_{n,c} = \frac{1}{H \times W} \sum_{h, w} x_{n, c, h, w}, \quad \sigma_{n,c}^2 = \frac{1}{H \times W} \sum_{h, w} (x_{n, c, h, w} - \mu_{n,c})^2
$$

- **Pooling axes:** Spatial dimensions $(H, W)$ per single $(n, c)$ instance.
- **Characteristics:** Strips contrast and global stylistic features from an image while preserving content structures.
- **Primary Use Case:** Style transfer, generative image models (GANs), and visual domain adaptation. Rarely used in text/LLMs.

### 5\. Group Normalization (GroupNorm)

GroupNorm divides the channel dimension $C$ into $G$ sequential groups (typically $G=32$) and computes the mean and variance across each group along with spatial dimensions.

$$
\mu_g = \frac{1}{(C/G) \times H \times W} \sum_{c \in G_k} \sum_{h, w} x_{n, c, h, w}
$$

- **Pooling axes:** A slice of channels $(C/G)$ and spatial dimensions $(H, W)$ per sample.
- **Characteristics:** Acts as a hybrid between LayerNorm (when $G=1$) and InstanceNorm (when $G=C$). It does not rely on batch size, making it a drop-in replacement for BatchNorm in computer vision tasks with small batch sizes (e.g., high-resolution object detection, 3D medical imaging, and Stable Diffusion U-Nets).

### 6\. Weight Normalization & Spectral Normalization

These methods reparameterize the network's weight matrices $W$ rather than the activation vectors $x$.

#### Weight Normalization (WeightNorm)

Decouples the norm of the weight vector from its direction:

$$
W = g \frac{V}{\Vert{}V\Vert{}_2}
$$

- $g$ is a scalar parameter controlling magnitude; $V$ is a tensor parameterizing orientation.
- Accelerates convergence in convolutional architectures and audio/waveform models (e.g., WaveNet).

#### Spectral Normalization

Constrains the Lipschitz constant of linear/convolutional layers by dividing the weight matrix by its largest singular value (spectral norm $\sigma(W)$):

$$
W_{\text{SN}} = \frac{W}{\sigma(W)}, \quad \text{where } \sigma(W) = \max_{h \neq 0} \frac{\Vert{}Wh\Vert{}_2}{\Vert{}h\Vert{}_2}
$$

- $\sigma(W)$ is approximated via power iteration during the forward pass.
- Prevents exploding gradients and mode collapse in Generative Adversarial Networks (GANs).

### 7\. Modern LLM-Specific Normalizations

#### Pre-LN vs. Post-LN vs. DeepNorm

The architectural placement of the normalization layer relative to residual connections determines signal propagation:

| Placement | Formula | Stability | Gradient Highway |
| --- | --- | --- | --- |
| **Post-LN** (Original Transformer) | $x_{l+1}=\mathrm{LN}(x_l+\mathrm{SubLayer}(x_l))$ | Unstable at scale; requires careful warmup | Degrades at large depths |
| **Pre-LN** (GPT-2, modern LLMs) | $x_{l+1}=x_l+\mathrm{SubLayer}(\mathrm{LN}(x_l))$ | Very stable; trains easily out of the box | Clear residual stream |
| **DeepNorm** | $x_{l+1}=\mathrm{LN}(x_l\cdot\alpha+\mathrm{SubLayer}(x_l))$ | Stabilizes ultra-deep networks (1000+ layers) | Scales variance deterministically |

#### QK-Norm (Query-Key Normalization)

As model sizes exceed 7B–13B parameters, attention logits $\frac{Q K^T}{\sqrt{d_k}}$ can grow unchecked, causing the softmax distribution to become overly peaked (entropy collapse) and triggering large float16/bfloat16 activation outliers.

$$
\tilde{Q} = \text{RMSNorm}(Q), \quad \tilde{K} = \text{RMSNorm}(K)
$$

$$
\text{Attention}(\tilde{Q}, \tilde{K}, V) = \text{softmax}\left(\frac{\tilde{Q} \tilde{K}^T}{\sqrt{d_k}}\right)V
$$

- Adopted in models like **Chinchilla, PaLM, Command R+, and Gemma 2** to prevent training instability and loss spikes at scale.

### Comparative Reference

| Technique | Dependent on Batch Size? | Inference State Required? | Primary Domain | Normalization Dimension |
| --- | --- | --- | --- | --- |
| **BatchNorm** | Yes | Yes (Running $\mu,\sigma^2$) | Vision (ResNet, ConvNets) | Across $(N,H,W)$ |
| **LayerNorm** | No | No | NLP, BERT, GPT-2/3 | Across feature dim $D$ |
| **RMSNorm** | No | No | Modern LLMs (LLaMA, Mistral) | Across feature dim $D$ (no mean) |
| **GroupNorm** | No | No | Vision (Diffusion, Detection) | Across $(C/G,H,W)$ |
| **InstanceNorm** | No | No | Style Transfer, GANs | Across $(H,W)$ |
| **QK-Norm** | No | No | LLM Attention Heads | Head dimension $d_k$ |
