---
title: "Batch-norm vs layer-norm"
date: "2026-10-26"
---

Batch Normalization (BatchNorm) and Layer Normalization (LayerNorm) perform the exact same mathematical transformation—standardizing activations to zero mean and unit variance—but they carve up the activation tensor across orthogonal dimensions.

### 1\. The Dimensional Contrast (The Axes)

Consider a 3D activation tensor in deep learning: $X \in \mathbb{R}^{B \times T \times D}$, where:

- $B$ is the **Batch** dimension (different samples/prompts)
- $T$ is the **Sequence/Spatial** dimension (tokens or pixels)
- $D$ is the **Channel/Feature** dimension (embedding dimension or hidden units)
        ```
Batch (B)
        ┌───────────────┐
       /               /│
      /               / │
     ┌───────────────┐  │
     │               │  │ Feature (D)
     │   ACTIVATION  │  │   ▲
     │     TENSOR    │  │   │
     │               │ /    │
     └───────────────┘/     ▼
       Sequence (T)
```

#### BatchNorm: Pools along $(B, T)$, fixes $D$

- **Slice direction:** Vertical/Batch-wise.
- For each individual feature channel $d \in \{1, \dots, D\}$, it collects all values across every sample in the batch and every position in time:
	$$
	\mu_d = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T X_{b, t, d}
	$$
- **Constraint:** A sample's normalized value depends directly on what other samples are in the current batch.

#### LayerNorm: Pools along $D$, fixes $(B, T)$

- **Slice direction:** Horizontal/Per-sample.
- For each individual token position $t$ of each individual sample $b$, it collects all hidden features $d \in \{1, \dots, D\}$:
	$$
	\mu_{b, t} = \frac{1}{D} \sum_{d=1}^D X_{b, t, d}
	$$
- **Constraint:** Completely self-contained. A sample's normalization is calculated using only its own internal features, completely agnostic to other samples in the batch.

### 2\. How Both Achieve the Same Underlying Goal

Despite slicing the tensor orthogonally, both techniques achieve the primary objective of normalization: **bounding scale and conditioning the gradient dynamics**.

#### Variance Stabilization Across Depth

In an unnormalized deep network, forward pass activations either explode or vanish as they pass through dozens of weight matrices:

$$
\text{Var}(x_l) \approx \prod_{k=1}^l \sigma^2(W_k) \cdot \text{Var}(x_0)
$$

Both normalizations force the output variance back to $\approx 1$ before passing it to the next layer:

- **BatchNorm** achieves this by ensuring that *each neuron across the dataset* fires with unit variance.
- **LayerNorm** achieves this by ensuring that the *signal vector entering each layer* has a unit norm across its feature space.

In both cases, downstream matrix multiplications $W x$ receive bounded inputs, preventing the exponential growth or collapse of signal energy through deep layers.

#### Scale Invariance (Weight and Activation Invariance)

Both BatchNorm and LayerNorm induce scale invariance on the weights and activations:

$$
\text{Norm}(c \cdot x) = \text{Norm}(x) \quad \text{for scalar } c > 0
$$

Because the normalization automatically divides out any scalar factor $c$, scaling a weight matrix $W \to \alpha W$ does not change the normalized representation:

$$
\text{Norm}(\alpha W x) = \text{Norm}(W x)
$$

This yields an automatic self-regulating learning rate: when weights grow large, effective gradient updates scale down by $\frac{1}{\Vert{}W\Vert{}}$, naturally stabilizing optimization.

#### Loss Hessian Conditioning

Both methods eliminate directional extremes in the loss landscape. By forcing the scale of inputs to be comparable across dimensions or across samples, the Hessian $\nabla^2 \mathcal{L}$ becomes well-conditioned ($\lambda_{\max} / \lambda_{\min} \approx 1$), eliminating narrow ravines and allowing faster training convergence with larger learning rates.

### 3\. Key Differences and Trade-offs

| Dimension | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
| --- | --- | --- |
| **Statistical Pooling** | Across Batch & Time/Space: $(B,T)$ per feature $D$ | Across Features: $D$ per sample $(b,t)$ |
| **Batch Size Dependency** | **High.** Fails or degrades significantly when $B<16$ | **Zero.** Works identically for $B=1$ as for $B=4096$ |
| **Inference Behavior** | **Requires state.** Must save and freeze running mean/variance during training | **Stateless.** Calculates mean and variance dynamically on each forward pass |
| **Sequence Lengths** | Struggles with variable-length sequences (padding distorts statistics) | Trivially handles variable sequence lengths and masked attention |
| **Regularization Effect** | Stochastically perturbs activations via mini-batch noise (acts like weak dropout) | Deterministic; introduces no mini-batch noise regularization |
| **Distributed Scaling** | Requires `SyncBatchNorm` across GPUs, adding all-reduce latency | No communication across GPUs needed for normalization |
| **Primary Domain** | **Computer Vision** (ResNet, ConvNeXt, standard CNNs) | **NLP & Sequence Models** (Transformers, BERT, GPT, LLaMA) |

### Why LLMs Can Use LayerNorm but Cannot Use BatchNorm

1. **Autoregressive Generation ($B=1$):** During inference, an LLM generates tokens sequentially. The batch size is often $B=1$. BatchNorm cannot compute variance with a single point ($N=1$), and relying on training moving averages fails because the dynamic range of generative sequences shifts over multi-turn context.
2. **Padding Contamination:** In NLP batches, sentences have different lengths. Shorter sentences are padded with zero tokens. In BatchNorm, these zeros get pooled into the batch statistics, corrupting the mean and variance of actual tokens. LayerNorm computes statistics token-by-token, avoiding cross-token leakage.
3. **Information Leakage Across Samples:** In BatchNorm, the prediction for sequence $A$ depends on the contents of sequence $B$ in the same batch. In self-attention architectures where token representations are already complex, introducing cross-batch interdependencies complicates loss convergence and attention calibration.
