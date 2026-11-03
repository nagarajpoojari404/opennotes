---
title: "Reinforcement Learning"
date: "2026-10-28"
---

Reinforcement Learning (RL) is a framework for learning how to map situations to actions so as to maximize a numerical reward signal. Unlike supervised learning (where a model learns from labeled targets) or unsupervised learning (where a model finds hidden structure in unlabeled data), an RL agent learns entirely through **trial, error, and delayed consequence** via interaction with an environment.

### The Fundamental Framework: Markov Decision Processes (MDP)

Almost all RL problems are formalized as a **Markov Decision Process**, defined by the 5-tuple $(S, A, P, R, \gamma)$:


1. **State ($S \in \mathcal{S}$):** A complete description of the state of the world. If the agent only receives partial observations, the problem becomes a POMDP (Partially Observable MDP).
2. **Action ($A \in \mathcal{A}$):** The decision space available to the agent (can be discrete like `{left, right, jump}` or continuous like torques applied to robot joints).
3. **Transition Dynamics ($P(s' \mid s, a)$):** The probability of transitioning to state $s'$ after taking action $a$ in state $s$.
4. **Reward Function ($R(s, a, s')$):** A scalar feedback signal indicating immediate performance.
5. **Discount Factor ($\gamma \in [0, 1)$):** Determines the present value of future rewards. It ensures mathematical convergence over infinite horizons and models temporal preference:
	$$
	G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
	$$
	where $G_t$ is the **discounted return** at timestep $t$.

The **Markov Property** assumes that the future is conditionally independent of the past given the present:

$$
P(S_{t+1} = s', R_{t+1} = r \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, \dots) = P(S_{t+1} = s', R_{t+1} = r \mid S_t = s_t, A_t = a_t)
$$

### Core Components of an RL Agent

An RL agent typically maintains one or more of the following:

#### 1\. Policy ($\pi$)

The agent's behavior strategy:

- **Deterministic:** $a = \pi(s)$
- **Stochastic:** $\pi(a \mid s) = P(A_t = a \mid S_t = s)$

#### 2\. Value Functions

- **State-Value Function ($V^{\pi}(s)$):** The expected return starting from state $s$ under policy $\pi$:
	$$
	V^{\pi}(s) = \mathbb{E}_{\pi} \left[ G_t \mid S_t = s \right]
	$$
- **Action-Value Function ($Q^{\pi}(s, a)$):** The expected return starting from state $s$, executing action $a$, and thereafter following policy $\pi$:
	$$
	Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ G_t \mid S_t = s, A_t = a \right]
	$$

#### 3\. The Bellman Equations

Value functions decompose recursively into an immediate reward plus the discounted expected value of the next state:

- **Bellman Expectation Equation for $V(s)$:**
	$$
	V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V^{\pi}(s') \right]
	$$
- **Bellman Optimality Equation for $Q^*(s, a)$:**
	$$
	Q^*(s, a) = \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
	$$

Finding the optimal policy $\pi^*$ is equivalent to finding $Q^*(s, a)$, since $\pi^*(s) = \arg\max_a Q^*(s, a)$.

### Taxonomy of RL Algorithms

```
Reinforcement Learning
                                   |
         +-------------------------+-------------------------+
         |                                                   |
    Model-Free                                           Model-Based
         |                                                   |
   +-----+-----+                                    +--------+--------+
   |           |                                    |                 |
Value-Based  Policy-Based                      Learn Model       Given Model
(Q-Learning, (REINFORCE)                      (World Models,     (AlphaZero,
 DQN)          |                               MBPO, MuZero)      MCTS)
               |
         Actor-Critic
         (A2C, PPO, SAC)
```

#### Model-Free vs. Model-Based

- **Model-Based:** Learns or uses an explicit model of transition dynamics $P(s' \mid s, a)$ to plan ahead (e.g., Monte Carlo Tree Search, Dyna-Q). Highly sample-efficient, but susceptible to model-bias/exploitation.
- **Model-Free:** Directly maps experiences to values or policies without explicitly modeling environment transitions. Less sample-efficient, but simpler and robust in complex environments.

#### On-Policy vs. Off-Policy

- **On-Policy:** Evaluates or improves the exact policy used to collect data (e.g., SARSA, REINFORCE, PPO).
- **Off-Policy:** Evaluates or improves a target policy $\pi$ using data collected by an arbitrary behavior policy $\mu$ or an experience replay buffer (e.g., Q-Learning, DQN, SAC).

### Three Core Algorithmic Paradigms

#### 1\. Value-Based Methods (e.g., Q-Learning & DQN)

Instead of parameterizing a policy directly, parameterize the value function $Q_\theta(s, a)$.

- **Tabular Q-Learning update rule:**
	$$
	Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]
	$$
	where $\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$ is the **Temporal Difference (TD) error**.
- **Deep Q-Networks (DQN):** Replaces tabular lookups with neural networks $\theta$. To stabilize training against non-stationary targets and correlated samples, DQN introduces:
	1. **Replay Buffer:** Decouples temporal correlations by uniform random batch sampling.
		2. **Target Network ($\theta^-$):** Periodic freezing of the target $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ to avoid moving-target instability.

#### 2\. Policy Gradient Methods (e.g., REINFORCE)

Directly parameterize policy $\pi_\theta(a \mid s)$ and optimize the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$ via gradient ascent.

By the **Policy Gradient Theorem**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]
$$

- **REINFORCE (Monte Carlo Policy Gradient):** Uses full trajectory empirical return $G_t$ as an unbiased sample estimate of $Q(s, a)$:
	$$
	\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t
	$$
	*Advantage:* Works seamlessly in continuous action spaces.
	*Disadvantage:* High variance because sample trajectory returns $G_t$ fluctuate wildly.

#### 3\. Actor-Critic Architectures (e.g., PPO, SAC)

Actor-Critic combines the strengths of value and policy methods:

- **Actor:** Parameterizes the policy $\pi_\theta(a \mid s)$ (decides what action to take).
- **Critic:** Parameterizes value function $V_\phi(s)$ or $Q_\phi(s, a)$ (evaluates how good the action was).

To reduce variance without adding bias, replace $Q(s, a)$ with the **Advantage Function** $A(s, a) = Q(s, a) - V(s)$, estimated via Generalized Advantage Estimation (GAE) or 1-step TD:

$$
A(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

- **Proximal Policy Optimization (PPO):** Prevents catastrophically large policy updates by clipping the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$:
	$$
	L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]
	$$

### Key Theoretical Challenges in RL

| Challenge | Description | Common Solutions |
| --- | --- | --- |
| **Exploration vs. Exploitation** | Choosing between trying unknown actions for information vs. taking known high-reward actions. | $\epsilon$-greedy, entropy regularization, Upper Confidence Bounds (UCB), intrinsic curiosity. |
| **Credit Assignment** | Determining which specific past action in a long trajectory was responsible for a delayed reward. | Temporal difference learning, eligibility traces $\mathrm{TD}(\lambda)$, reward shaping. |
| **Deadly Triad** | Combining **function approximation**, **bootstrapping**, and **off-policy training** can lead to divergence and unbounded value estimates. | Target networks, double Q-learning, clipped double-Q (as in TD3/SAC). |
| **Sample Inefficiency** | Model-free RL can require millions of environment interactions to converge. | Model-based RL, offline RL (CQL, IQL), pretraining via imitation learning. |
