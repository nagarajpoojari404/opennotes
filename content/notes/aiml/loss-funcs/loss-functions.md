---
title: "Loss functions"
date: "2026-10-26"
---

Loss functions measure the discrepancy between a model's predictions and actual targets, guiding optimization via backpropagation. In neural networks and large language models (LLMs), they fall into distinct families based on the learning objective: regression, classification, distribution matching, embeddings, and alignment.

## 1\. Regression Loss Functions

Used when predicting continuous numerical targets.

### Mean Squared Error (MSE / $L_2$ Loss)

Measures the average squared difference between predictions and targets:

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

- **Mechanism:** Penalizes large errors quadratically, making it sensitive to outliers.
- **Use Cases:** Standard regression, image reconstruction (autoencoders), diffusion model score matching / denoising objectives ($\epsilon$\-prediction).

### Mean Absolute Error (MAE / $L_1$ Loss)

Measures the average linear distance between predictions and targets:

$$
L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^N \vert{}y_i - \hat{y}_i\vert{}
$$

- **Mechanism:** Constant gradient magnitude ($\pm 1$) regardless of error size; robust against extreme outliers.
- **Use Cases:** Financial forecasting, sparse signal recovery, image super-resolution where MSE causes excessive blurriness.

### Huber Loss & Smooth $L_1$

Blends $L_2$ behavior for small errors with $L_1$ behavior for large errors:

$$
L_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } \vert{}y - \hat{y}\vert{} \le \delta \\ \delta \cdot \left(\vert{}y - \hat{y}\vert{} - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}
$$

- **Mechanism:** Differentiable everywhere (unlike pure $L_1$ at zero) while preventing gradient explosions caused by $L_2$ on outliers.
- **Use Cases:** Object detection bounding-box regression (e.g., Fast/Faster R-CNN), deep reinforcement learning (DQN value targets).

## 2\. Classification & Probabilistic Losses

Used for discrete categorical prediction and probability estimation.

### Binary Cross-Entropy (BCE / Log Loss)

Quantifies divergence between binary ground-truth labels $y \in \{0, 1\}$ and predicted probabilities $\hat{y} \in (0, 1)$:

$$
L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \Big(y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Big)
$$

- **Mechanism:** Steeply penalizes confident, incorrect predictions as $\hat{y} \to 0$ when $y = 1$.
- **Use Cases:** Binary classification, multi-label classification (treating each label independently with a sigmoid head).

### Categorical Cross-Entropy (CCE)

The standard loss for multi-class classification where classes are mutually exclusive:

$$
L_{\text{CCE}} = -\sum_{c=1}^C y_c \log(\hat{y}_c) = -\log(\hat{y}_{\text{target}})
$$

- **Mechanism:** Evaluated over softmax distributions. Drives the logit of the correct class higher relative to all competing classes.
- **Use Cases:** Multi-class classification (ImageNet), next-token prediction in language modeling.

### Focal Loss

An adaptive cross-entropy variant addressing severe class imbalance:

$$
L_{\text{FL}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- **Mechanism:** Introduces a modulating factor $(1 - p_t)^\gamma$. When an example is well-classified ($p_t \to 1$), the modulating factor drops to near zero, down-weighting "easy" examples and focusing gradient updates on hard, ambiguous examples.
- **Use Cases:** Dense object detection (RetinaNet), rare token/edge-case classification.

## 3\. Distribution Divergence Losses

Quantify differences between continuous or discrete probability densities.

### Kullback-Leibler Divergence (KL Divergence)

Measures information lost when approximating true distribution $P$ with distribution $Q$:

$$
D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

- **Mechanism:** Asymmetric ($D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$). Heavily penalizes $Q$ if it places zero probability where $P$ has non-zero mass.
- **Use Cases:** Variational Autoencoders (VAEs) as a latent regularization term matching latent distributions to $\mathcal{N}(0, I)$; policy regularization in RLHF.

### Jensen-Shannon Divergence (JSD) & Wasserstein Distance

- **JSD:** A symmetrized, smoothed variant of KL divergence bounded between $[0, \log 2]$:
	$$
	\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)
	$$
- **Wasserstein ($L_1$ Earth Mover's Distance):** Measures the minimum cost of transporting probability mass from distribution $P$ to $Q$. Provides smooth, non-vanishing gradients even when support sets do not overlap.
- **Use Cases:** Generative Adversarial Networks (WGAN-GP), distribution matching in domain adaptation.

## 4\. Metric Learning & Contrastive Losses

Used to map semantic similarities into distance metrics within embedding spaces.

### Triplet Loss

Enforces relative geometric ordering among an Anchor ($A$), a Positive example ($P$), and a Negative example ($N$):

$$
L_{\text{Triplet}} = \max\Big(0, \, \Vert{}f(A) - f(P)\Vert{}^2 - \Vert{}f(A) - f(N)\Vert{}^2 + \alpha\Big)
$$

- **Mechanism:** Enforces $\Vert{}f(A) - f(N)\Vert{}^2 \ge \Vert{}f(A) - f(P)\Vert{}^2 + \alpha$, ensuring negative points are separated from positives by at least a safety margin $\alpha$.
- **Use Cases:** Face verification (FaceNet), speaker diarization, visual search.

### InfoNCE (Noise-Contrastive Estimation)

Extends contrastive estimation to multi-negative settings using a softmax formulation:

$$
L_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{i=1}^K \exp(\text{sim}(q, k_i^-) / \tau)}
$$

- **Mechanism:** Maximizes mutual information lower bounds between positive representations while repelling randomly sampled negatives across a temperature scaling parameter $\tau$.
- **Use Cases:** Self-supervised visual representation learning (SimCLR, MoCo), multi-modal alignment (CLIP), text embedding models (E5, BGE).

## 5\. LLM Pre-training & Fine-Tuning Losses

LLM training relies on cross-entropy adapted for sequence modeling and vocabulary token distributions.

| Objective | Formulation | Core Purpose |
| --- | --- | --- |
| **Causal Language Modeling (CLM / Autoregressive NLL)** | $L=-\sum_{t=1}^{T}\log P(x_t \mid x_{\lt t})$ | Predicts the next token given preceding context. Standard across GPT, Llama, and Mistral architectures. |
| **Masked Language Modeling (MLM)** | $L=-\sum_{m\in M}\log P(x_m \mid x_{\setminus M})$ | Predicts masked tokens ($15\%$) using bidirectional context. Standard in BERT and RoBERTa. |
| **Sequence-to-Sequence / Prefix LM** | $L=-\sum_{t=1}^{T}\log P(y_t \mid y_{\lt t}, X)$ | Autoregressively decodes a target sequence conditioned on an unmasked encoder prefix. Used in T5 and BART. |

### Auxiliary Losses in LLMs

- **Label Smoothing:** Replaces hard one-hot targets $y$ with $(1 - \epsilon)y + \frac{\epsilon}{\vert{}V\vert{}}$, preventing overconfident logit saturation and improving calibration.
- **Z-Loss:** Penalizes large partition functions ($\sum \exp(z_i)$) by adding $c \cdot (\log \sum \exp(z_i))^2$. Mitigates numerical overflow during FP16/BF16 training.
- **Router Load Balancing Loss (Mixture of Experts):** Enforces uniform expert dispatch across tokens in MoE models:
	$$
	L_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i
	$$
	where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average routing probability assigned to expert $i$.

## 6\. LLM Preference Alignment & Post-Training Losses

Used to align LLM behaviors with human or AI-evaluated preferences.

### RLHF / PPO Policy Objective

Proximal Policy Optimization trains an actor model $\pi_\theta$ against a learned scalar reward model $r_\psi$:

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right] + \beta \, D_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)
$$

- **Mechanism:** Maximizes expected advantage $\hat{A}_t$ derived from reward model scores, constrained by a clipped probability ratio $r_t(\theta) = \frac{\pi_\theta(y_t\vert{}x, y_{\lt t})}{\pi_{\text{old}}(y_t\vert{}x, y_{\lt t})}$ and a KL divergence penalty to prevent drift from base reference policy $\pi_{\text{ref}}$.
- **Use Cases:** Reinforcement Learning from Human Feedback (InstructGPT, Claude alignment).

### Direct Preference Optimization (DPO)

Re-parameterizes the Bradley-Terry preference model to eliminate the external reward model and RL loop entirely:

$$
L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

- **Mechanism:** Derives an exact closed-form expression for the implicit reward $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. Directly increases the probability of the winning completion $y_w$ relative to reference policy while suppressing the losing completion $y_l$.
- **Use Cases:** Post-training preference alignment in modern open-weight LLMs (Llama 3, Zephyr, Qwen).

### Identity Preference Optimization (IPO) & Kahneman-Tversky Optimization (KTO)

- **IPO:** Adds a regularizer to prevent DPO from quickly driving implicit rewards toward negative infinity when preferences are noisy or deterministic:
	$$
	L_{\text{IPO}}(\theta) = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{\tau}{2} \right)^2 \right]
	$$
- **KTO:** Replaces pairwise comparisons $(y_w, y_l)$ with unpaired binary feedback ("desirable" vs. "undesirable"), maximizing utility functions modeled after Prospect Theory loss aversion.

### Group Relative Policy Optimization (GRPO)

Eliminates the separate critic network in PPO by sampling a group of outputs $\{o_1, o_2, \dots, o_G\}$ for a prompt $q$, scoring them with a rule-based or model verifier, and computing normalized advantages across the group:

$$
A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}
$$

- **Mechanism:** Updates policy weights using group-normalized advantages directly, significantly reducing memory consumption and stabilization issues inherent to actor-critic frameworks.
- **Use Cases:** Large-scale reasoning model training (DeepSeek-R1, mathematical and coding RL regimes).
