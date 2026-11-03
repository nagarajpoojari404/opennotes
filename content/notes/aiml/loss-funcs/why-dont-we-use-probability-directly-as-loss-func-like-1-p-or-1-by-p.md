---
title: "Why dont we use probability directly as loss func like 1-p or 1_by_p ?"
date: "2026-10-26"
---

Why dont we use probability directly as loss func like 1-p or 1/p


Using raw probabilities like $L = 1 - p$ or $L = 1/p$ instead of $-\log(p)$ breaks down during gradient descent due to **gradient saturation**, **information-theoretic inconsistency**, and **exponential vanishing in joint probabilities**.

### 1\. The Gradient Vanishing Problem ($1 - p$)

In neural networks, the probability $p$ is obtained by passing raw logits $z$ through an activation function like Softmax or Sigmoid.

Consider a single binary output where $p = \sigma(z) = \frac{1}{1 + e^{-z}}$. The derivative of the sigmoid is:

$$
\frac{\partial p}{\partial z} = p(1 - p)
$$

Now evaluate the gradient of loss with respect to the logit $z$ for both functions:

#### Under Linear Loss $L = 1 - p$:

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} = (-1) \cdot p(1 - p) = p^2 - p
$$

- **When the model is completely wrong** ($z \ll 0 \implies p \approx 0$):
	$$
	\frac{\partial L}{\partial z} \approx 0
	$$
- **Result:** The gradient **vanishes**. When the model makes a confident, catastrophic error, the learning signal drops to zero, and the network cannot update its weights to escape the bad state.

#### Under Cross-Entropy Loss $L = -\log(p)$:

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} = \left(-\frac{1}{p}\right) \cdot p(1 - p) = -(1 - p) = p - 1
$$

- **When the model is completely wrong** ($p \approx 0$):
	$$
	\frac{\partial L}{\partial z} \approx -1
	$$
- **Result:** The log cancels out the activation's squashing derivative. The gradient is strictly proportional to the error $(p - y)$, delivering a constant, healthy gradient update regardless of how wrong the network is.

### 2\. The Instability and Curvature Problem ($1/p$)

If we choose $L = \frac{1}{p}$, the loss function scales inversely with probability:

$$
\frac{\partial L}{\partial p} = -\frac{1}{p^2}
$$

$$
\frac{\partial L}{\partial z} = \left(-\frac{1}{p^2}\right) \cdot p(1 - p) = -\frac{1 - p}{p} = 1 - \frac{1}{p}
$$

- **When $p \to 0$:** $\frac{\partial L}{\partial z} \to -\infty$.
- **The issue:** The gradient diverges as a hyperbola ($O(1/p)$) rather than the smooth linear error $(p - 1)$ of cross-entropy.
- A single out-of-distribution token or misclassified edge case causes an immediate **gradient explosion**, blowing up the optimizer's momentum buffers (Adam's first and second moments) and corrupting the weight tensors.

### 3\. Joint Probabilities and Sequence Likelihoods

In language models, the objective is to maximize the joint probability of a sequence of tokens $W = (w_1, w_2, \dots, w_T)$:

$$
P(W) = \prod_{t=1}^T P(w_t \mid w_{\lt t})
$$

#### Why linear loss fails on sequences:

If you define the sequence loss linearly:

$$
L = 1 - P(W) = 1 - \prod_{t=1}^T p_t
$$

- For a moderate sequence of 512 tokens, even if each token has high probability ($p_t \approx 0.99$):
	$$
	\prod_{t=1}^{512} 0.99 \approx 0.0057 \implies L \approx 0.9943
	$$
- For lower probabilities ($p_t \approx 0.5$):
	$$
	\prod_{t=1}^{512} 0.5 \approx 10^{-154}
	$$
- Under standard floating-point representation (`float32` or `bfloat16`), this leads to **immediate arithmetic underflow to zero**, rendering computation impossible.

#### Why the logarithm is required:

Applying $-\log$ converts the product into an additive sum:

$$
L_{\text{NLL}} = -\log P(W) = -\sum_{t=1}^T \log P(w_t \mid w_{\lt t})
$$

- Multiplication becomes addition, preventing numerical underflow.
- The total loss decomposes cleanly into token-level losses, allowing gradients to be computed and backpropagated independently to each token position:
	$$
	\frac{\partial L}{\partial z_t} = p_t - 1
	$$

### 4\. Maximum Likelihood and Information Theory

- **Maximum Likelihood Estimation (MLE):** Minimizing $-\log(p)$ is mathematically equivalent to maximizing the log-likelihood of the data distribution under the model parameters:
	$$
	\arg\max_\theta \sum_i \log P_\theta(y_i \mid x_i) \equiv \arg\min_\theta \sum_i -\log P_\theta(y_i \mid x_i)
	$$
- **Surprisal / Self-Information:** In information theory, the information content of an event is $I(p) = \log_2(1/p) = -\log_2(p)$. An event with $p=1$ carries $0$ bits of surprise, whereas an event with $p \to 0$ carries infinite surprise. Cross-entropy directly minimizes the average bit surprise (entropy) of the true labels under the predicted model distribution. Neither $1-p$ nor $1/p$ corresponds to a valid metric of information divergence.
