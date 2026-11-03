---
title: "Initializers"
date: "2026-10-26"
---

Weight initialization sets the starting values of trainable parameters to prevent gradients from vanishing or exploding during the forward and backward passes. In modern deep learning and Large Language Models (LLMs), initializers are derived by analyzing how variance propagates through linear, recurrent, or attention layers.

### Variance Propagation Fundamentals

Consider a forward linear layer:

$$
y = W x = \sum_{i=1}^{n_{\text{in}}} w_i x_i
$$

Assuming:

- The inputs $x_i$ are independent and identically distributed with $\mathbb{E}[x_i] = 0$ and $\text{Var}(x_i) = \sigma_x^2$.
- The weights $w_i$ are independent, drawn from a distribution with $\mathbb{E}[w_i] = 0$ and $\text{Var}(w_i) = \sigma_w^2$.
- $w_i$ and $x_i$ are mutually independent.

The variance of $y$ is:

$$
\text{Var}(y) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(w_i x_i) = \sum_{i=1}^{n_{\text{in}}} \left( \mathbb{E}[w_i^2 x_i^2] - (\mathbb{E}[w_i] \mathbb{E}[x_i])^2 \right) = n_{\text{in}} \sigma_w^2 \sigma_x^2
$$

To maintain constant signal magnitude across depth ($\text{Var}(y) \approx \text{Var}(x_i)$):

$$
\sigma_w^2 = \frac{1}{n_{\text{in}}}
$$

During backpropagation, the gradient flow yields:

$$
\frac{\partial L}{\partial x_j} = \sum_{k=1}^{n_{\text{out}}} w_{kj} \frac{\partial L}{\partial y_k} \implies \text{Var}\left(\frac{\partial L}{\partial x_j}\right) = n_{\text{out}} \sigma_w^2 \text{Var}\left(\frac{\partial L}{\partial y}\right)
$$

Preserving gradient variance requires $\sigma_w^2 = \frac{1}{n_{\text{out}}}$.

### Classical Neural Network Initializers

#### 1\. Zero & Constant Initialization

- **Formula:** $W = 0$ or $W = c$
- **Problem:** If all weights in a hidden layer are initialized to the same value, every neuron in that layer computes identical outputs and receives identical gradients:
	$$
	\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial y_i} x_j
	$$
	Symmetry is never broken, collapsing the layer to the representation capacity of a single unit.
- **Valid Use:** Biases ($b = 0$) and normalization scaling parameters ($\gamma = 1, \beta = 0$).

#### 2\. Standard Normal & Uniform Initializers

- **Uniform:** $W \sim \mathcal{U}(-a, a)$ with $\sigma_w^2 = \frac{a^2}{3}$.
- **Normal:** $W \sim \mathcal{N}(0, \sigma^2)$ where commonly $\sigma = 0.01$ or $0.02$.
- **Limitation:** Fixed variance independent of fan-in/fan-out causes exponential growth or decay across deep networks ($L$ layers):
	$$
	\text{Var}(y_L) \approx (n \sigma^2)^L \text{Var}(x)
	$$

#### 3\. LeCun Initialization

Designed for linear transformations and sigmoid-like activations in their near-linear regime:

- **Condition:** $\text{Var}(y) = \text{Var}(x) \implies \sigma_w^2 = \frac{1}{n_{\text{in}}}$
- **Normal:**
	$$
	W \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)
	$$
- **Uniform:** Set $a = \sqrt{3 \sigma_w^2} = \sqrt{\frac{3}{n_{\text{in}}}}$:
	$$
	W \sim \mathcal{U}\left(-\sqrt{\frac{3}{n_{\text{in}}}}, \sqrt{\frac{3}{n_{\text{in}}}}\right)
	$$

#### 4\. Xavier / Glorot Initialization

Accounts for both forward activation variance and backward gradient variance by taking the harmonic mean of fan-in ($n_{\text{in}}$) and fan-out ($n_{\text{out}}$):

$$
\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

- **Normal:**
	$$
	W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
	$$
- **Uniform:**
	$$
	W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
	$$
- **Best suited for:** Symmetric, zero-centered activation functions ($\tanh$, softsign, linear projections).

#### 5\. He / Kaiming Initialization

Designed for asymmetric non-linearities like ReLU ($f(x) = \max(0, x)$). Because ReLU zeros out negative values, assuming zero-centered symmetric inputs yields:

$$
\mathbb{E}[f(y)^2] = \frac{1}{2} \mathbb{E}[y^2] \implies \text{Var}(f(y)) \approx \frac{1}{2} n_{\text{in}} \sigma_w^2 \text{Var}(x)
$$

To counteract this 50% signal attenuation:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}}}
$$

- **Normal:**
	$$
	W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
	$$
- **Uniform:**
	$$
	W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)
	$$
- **Leaky ReLU Extension:** With negative slope parameter $\alpha$:
	$$
	\sigma_w^2 = \frac{2}{(1 + \alpha^2) n_{\text{in}}}
	$$

#### 6\. Orthogonal Initialization

Addresses vanishing/exploding dynamics in deep linear and recurrent networks (RNNs/LSTMs) by initializing weight matrices as orthogonal transformations:

$$
W^T W = I \quad \text{or} \quad W W^T = I
$$

Every singular value $\sigma_i(W) = 1$, preserving vector norms under repeated multiplications:

$$
\Vert{}W x\Vert{}_2 = \Vert{}x\Vert{}_2
$$

- **Construction:** Generate a random Gaussian matrix $M \sim \mathcal{N}(0, 1)$, compute its QR decomposition $M = QR$, and assign $W = Q \cdot \text{diag}(\text{sgn}(\text{diag}(R)))$.
- **Scaled Orthogonal:** Often scaled by a gain factor $g$:
	$$
	W = g \cdot Q \quad (g = \sqrt{2} \text{ for ReLU})
	$$

### Large Language Model (Transformer) Initializers

Deep decoder-only autoregressive models (GPT, LLaMA, Mistral, PaLM) introduce residual streams, attention scaling, and normalization layers that require specialized initialization schemes.


#### 7\. Truncated Normal Initialization

Standard Gaussian sampling occasionally produces extreme outliers in the tails ($\vert{}w\vert{} > 3\sigma$), causing early softmax saturation or training instability in low-precision regimes (FP16/BF16).

- **Formula:**
	$$
	w \sim \mathcal{N}(0, \sigma^2), \quad \text{constrained to } w \in [-2\sigma, 2\sigma]
	$$
	Values outside $[-2\sigma, 2\sigma]$ are discarded and resampled until they fall within the band.
- **Common LLM Baseline:** $\sigma = 0.02$ or $\sigma = \sqrt{\frac{2}{d_{\text{model}} + d_{\text{ffn}}}}$.

#### 8\. Residual-Scaled Initialization (GPT-2 / GPT-NeoX)

In models using standard Pre-LayerNorm or Post-LayerNorm architectures with residual connections:

$$
x_{l+1} = x_l + f_l(x_l)
$$

If each sub-layer $f_l$ outputs unit variance, the variance of the residual stream accumulates linearly with the number of layers $N$:

$$
\text{Var}(x_N) \approx N \cdot \text{Var}(x_0)
$$

To keep variance bounded at the output of the residual network, layers contributing to the residual branch (specifically the output projections: `c_proj` in MHA and `down_proj` in MLP) are scaled down by $\frac{1}{\sqrt{2N}}$:

- **Feedforward & Attention Input/Intermediates:**
	$$
	W \sim \mathcal{N}\left(0, \sigma_0^2\right), \quad \sigma_0 = \frac{1}{\sqrt{d_{\text{model}}}}
	$$
- **Residual Output Projections (`c_proj`, `out_proj`, `down_proj`):**
	$$
	W_{\text{res}} \sim \mathcal{N}\left(0, \frac{\sigma_0^2}{2N}\right) = \mathcal{N}\left(0, \frac{1}{2N \cdot d_{\text{model}}}\right)
	$$
	Here, $N$ denotes the total number of Transformer blocks (accounting for two residual additions per block: multi-head self-attention and MLP).

#### 9\. DeepNorm Initialization

Introduced to stabilize very deep Transformers (e.g., 1000+ layers) without needing Post-LN warmup or failing to converge:

- **Residual formulation:**
	$$
	x_{l+1} = \text{LayerNorm}(x_l \cdot \alpha + f_l(x_l, \theta_l))
	$$
- **Variance scale condition:** Bounds the expected activation magnitude:
	$$
	\alpha = (2N)^{\frac{1}{4}}
	$$
- **Parameter scaling:** All weights in $f_l$ are scaled down by $\beta$:
	$$
	W \leftarrow \beta \cdot W, \quad \beta = (8N)^{-\frac{1}{4}}
	$$
	This ensures the variance growth per residual update stays below $O(1/\sqrt{N})$.

#### 10\. $\mu\text{P}$ (Maximal Update Parametrization)

Standard initializers like Xavier or He assume fixed model widths ($d_{\text{model}}$). When scaling from a small proxy model to a 70B+ LLM, learning dynamics drift because the optimal learning rate and activation scales depend on width. $\mu\text{P}$ coordinates weight variance and learning rate scaling with width $d$:

| Weight Matrix | Shape | Standard Parameterization ($\sigma^2$) | $\mu\text{P}$ Variance ($\sigma^2$) | $\mu\text{P}$ Learning Rate |
| --- | --- | --- | --- | --- |
| **Embedding** | $V\times d$ | $1$ | $1$ | $\eta \cdot d$ |
| **Attention $Q,K$** | $d\times d$ | $\frac{1}{d}$ | $\frac{1}{d}$ | $\eta$ |
| **Attention $V$** | $d\times d$ | $\frac{1}{d}$ | $\frac{1}{d^2}$ | $\eta \cdot d$ |
| **Attention Output / MLP** | $d\times d$ | $\frac{1}{d}$ | $\frac{1}{d^2}$ | $\eta \cdot d$ |
| **LM Output Head** | $d\times V$ | $\frac{1}{d}$ | $\frac{1}{d^2}$ | $\eta \cdot d$ |

- **Attention Logit Scaling:** Replaces standard $\frac{QK^T}{\sqrt{d_k}}$ with $\frac{QK^T}{d_k}$ to prevent logits from diverging as width increases.

### Comparative Summary

| Initializer | Primary Formula ($\sigma_w^2$) | Primary Target | Key Assumption |
| --- | --- | --- | --- |
| **LeCun** | $\frac{1}{n_{\text{in}}}$ | MLPs, Linear layers | Linear / Sigmoid near origin |
| **Glorot / Xavier** | $\frac{2}{n_{\text{in}}+n_{\text{out}}}$ | RNNs, Tanh nets | Linear / zero-centered activations |
| **He / Kaiming** | $\frac{2}{n_{\text{in}}}$ | CNNs, Deep MLPs | ReLU / Leaky ReLU non-linearities |
| **Orthogonal** | $W^\top W=I$ | Vanilla RNNs, LSTMs | Unit singular values across depth |
| **Truncated Normal** | $\mathcal{N}(0,\sigma^2)$ bounded | LLM embeddings & projections | Eliminates extreme tail outliers |
| **Residual-Scaled** | $\frac{1}{2N\cdot d_{\text{model}}}$ | LLM Residual outputs | Dampens accumulation across $N$ blocks |
| **DeepNorm** | $\beta^2\approx\frac{1}{8N}$ | Extremely deep Transformers | Bounds residual expansion analytically |
| **$\mu\text{P}$** | $O(1/d^2)$ for output/value layers | Large LLM pre-training | Width-independent hyperparameter transfer |
