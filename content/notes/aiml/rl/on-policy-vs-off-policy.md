---
title: "On-Policy vs Off-Policy"
date: "2026-10-28"
---

The core distinction between **on-policy** and **off-policy** reinforcement learning comes down to whether the agent learns about the exact strategy it is actively executing, or learns about a target strategy while acting under an entirely different behavior strategy.

- **Target policy ($\pi$):** The policy the agent evaluates or aims to optimize.
- **Behavior policy ($\mu$ or $b$):** The policy used to select actions and generate environmental trajectories:
	$$
	\tau = (s_0, a_0, r_0, s_1, a_1, \dots)
	$$

### Key Conceptual Differences

| Dimension | On-Policy ($\pi =\mu$) | Off-Policy ($\pi\neq\mu$) |
| --- | --- | --- |
| **Data Generation** | Data must come from the current version of the policy $\pi$. | Data can come from an older policy, an exploratory policy, human demos, or an arbitrary source $\mu$. |
| **Sample Efficiency** | **Low.** Past rollout trajectories become obsolete after a gradient update and must be discarded. | **High.** Can utilize an **Experience Replay Buffer** ($D$) to reuse transitions repeatedly. |
| **Stability** | **High.** Updates directly align with the current state-action visitation distribution. | **Lower.** Susceptible to divergence when combined with function approximation and bootstrapping (Sutton's "Deadly Triad"). |
| **Exploration Strategy** | Exploration must be baked into $\pi$ itself (e.g., entropy regularization, high variance Gaussian actions). | Decoupled: behavior policy can explore aggressively (e.g., $\epsilon$-greedy, Ornstein-Uhlenbeck noise) while target policy remains deterministic. |
| **Representative Algorithms** | SARSA, REINFORCE, TRPO, PPO, A2C/A3C | Q-Learning, DQN, DDPG, TD3, SAC |

### Mathematical Mechanism Comparison

#### 1\. Temporal Difference (TD) Updates: SARSA vs. Q-Learning

Consider a single transition $(s, a, r, s')$.

- **On-Policy (SARSA):**
	$$
	Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
	$$
	Here, $a' \sim \pi(\cdot\vert{}s')$ is the actual next action chosen by the behavior policy. If that action is sub-optimal or exploratory, the target reflects that risk.
- **Off-Policy (Q-Learning):**
	$$
	Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
	$$
	The target assumes the greedy policy $\pi(s') = \arg\max_{a'} Q(s', a')$, regardless of which action $a'$ the behavior policy actually took next.

#### 2\. Policy Gradients & Importance Sampling

In policy gradients, objective functions rely on expectations under the policy:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \vert{} s_t) \hat{A}_t \right]
$$

If trajectories were collected by another policy $\mu$, evaluating expectations requires **importance sampling** ratios:

$$
\rho_t = \frac{\pi_\theta(a_t \vert{} s_t)}{\mu(a_t \vert{} s_t)}
$$

- In purely on-policy policy gradient methods, $\rho_t = 1$.
- Algorithms like **PPO** maintain an on-policy structure while taking multiple stochastic gradient descent passes over a collected batch. To stay stable without diverging, PPO clips the probability ratio:
	$$
	r_t(\theta) = \frac{\pi_\theta(a_t\vert{}s_t)}{\pi_{\theta_{\text{old}}}(a_t\vert{}s_t)}
	$$
	Even though $\pi_\theta \neq \pi_{\theta_{\text{old}}}$ during inner epochs, PPO remains categorized as an on-policy method because the samples are discarded after each mini-batch cycle rather than stored in a persistent replay buffer.

### Trade-offs in Practice

- **Use On-Policy (e.g., PPO)** when working in fast, parallelized simulation environments (e.g., Isaac Sim, Brax, Atari with vectorized environments) where sample collection is cheap, and algorithm stability/monotonic policy improvement is the priority.
- **Use Off-Policy (e.g., SAC, TD3)** when real-world interactions are expensive (e.g., physical robotics, slow chemical/aerodynamic sims), or when bootstrapping from offline demonstration datasets (Offline RL, CQL).
