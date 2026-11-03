---
title: "Policy gradient theorem"
date: "2026-10-28"
---

The expression:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]
$$

is the core result of the **Policy Gradient Theorem** (introduced formally by Richard Sutton, David McAllester, Satinder Singh, and Yishay Mansour in 1999).

The fundamental problem it solves is this: the objective function $J(\theta)$ depends on $\theta$ in two separate places—the policy actions and the state distribution induced by that policy. Taking a naive derivative seems to require knowing the environment's transition probabilities $\mathcal{P}(s' \mid s, a)$, which are generally unknown. The theorem proves that the gradient with respect to $\theta$ does **not** depend on the gradient of the state distribution or the environment transition dynamics.

### Step-by-Step Derivation

Let a trajectory $\tau$ be a sequence of states and actions:

$$
\tau = (s_0, a_0, s_1, a_1, \dots, s_T)
$$

Let $R(\tau) = \sum_{t=0}^T R(s_t, a_t)$ be the total return of trajectory $\tau$.

The objective is to maximize the expected return over trajectories generated under policy $\pi_\theta$:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P(\tau; \theta) R(\tau) \, d\tau
$$

where $P(\tau; \theta)$ is the probability of trajectory $\tau$ occurring under policy parameter $\theta$:

$$
P(\tau; \theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)
$$

#### 1\. Differentiating the Objective

Differentiate $J(\theta)$ directly with respect to $\theta$:

$$
\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau; \theta) R(\tau) \, d\tau = \int \nabla_\theta P(\tau; \theta) R(\tau) \, d\tau
$$

#### 2\. The Log-Derivative Trick (Likelihood Ratio)

Notice that:

$$
\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)
$$

Substitute this back into the integral:

$$
\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]
$$

This reformulates the gradient as an **expectation under $\pi_\theta$**, meaning it can be estimated using Monte Carlo samples without computing explicit integrals.

#### 3\. Expanding the Trajectory Log-Probability

Take the logarithm of $P(\tau; \theta)$:

$$
\log P(\tau; \theta) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t)
$$

Now take the gradient with respect to $\theta$. Because the initial state distribution $\rho_0(s_0)$ and transition dynamics $P(s_{t+1} \mid s_t, a_t)$ do **not** depend on policy parameters $\theta$, their gradients are zero:

$$
\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Substituting this gives:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) R(\tau) \right]
$$

#### 4\. Causality and the Action-Value Function $Q^{\pi_\theta}(s, a)$

Expanding $R(\tau) = \sum_{t'=0}^{T-1} r_{t'}$, consider the product:

$$
\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=0}^{T-1} r_{t'}
$$

By causality, action $a_t$ cannot influence past rewards ($t' < t$). Under expectation, terms where $t' < t$ evaluate to zero:

$$
\mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=0}^{t-1} r_{t'} \right] = 0
$$

Thus, only the "reward-to-go" from step $t$ onward matters:

$$
G_t = \sum_{t'=t}^{T-1} r_{t'}
$$

Replacing the full return $R(\tau)$ with the reward-to-go:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]
$$

Because $Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}[G_t \mid s_t, a_t]$, taking the expectation across stationary state-action pairs $(s, a) \sim d^{\pi_\theta}(s)\pi_\theta(a \mid s)$ yields the compact form:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]
$$

### Intuitive Breakdown

- **$\nabla_\theta \log \pi_\theta(a \mid s)$ (Score function):** Points in the parameter direction that increases the probability of selecting action $a$ in state $s$.
- **$Q^{\pi_\theta}(s, a)$ (Scalar weight):** Scales the gradient step:
	- If $Q(s, a) > 0$, push parameters to make action $a$ **more** likely.
		- If $Q(s, a) < 0$, push parameters to make action $a$ **less** likely.
- **Why $\log$ appears:** Maximizing likelihood with sample averages naturally yields relative percentage changes ($\frac{\nabla p}{p} = \nabla \log p$), preventing high-probability actions from dominating updates simply because their baseline probability is already large.
