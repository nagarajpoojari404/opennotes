---
title: "PPO"
date: "2026-10-28"
---

Proximal Policy Optimization (PPO) is an on-policy reinforcement learning algorithm designed to balance the sample efficiency and simplicity of first-order gradient descent with the training stability of trust region methods.

### The Fundamental Problem PPO Solves

In standard policy gradient methods like REINFORCE, the objective maximizes expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T R(s_t, a_t) \right]
$$

Using the Policy Gradient Theorem, the gradient of this objective with respect to network parameters $\theta$ is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{t} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
$$

Where $\hat{A}_t$ is the estimated advantage:

$$
\hat{A}_t = Q(s_t, a_t) - V(s_t)
$$

Standard policy gradients suffer from two core failure modes:

1. **Destructive Policy Updates:** A single bad parameter update step can shift the policy into a parameter region where it collects terrible trajectories. Because it is on-policy, the agent cannot easily recover from collapsed performance.
2. **Poor Sample Efficiency:** Data collected from trajectory rollouts can only be used for a single gradient update before the policy distribution shifts enough to invalidate the expectations.

**Trust Region Policy Optimization (TRPO)** solved this by constraining each update using the Kullback–Leibler (KL) divergence between the old and new policies:

$$
\max_\theta \mathbb{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t)) \right] \le \delta
$$

While theoretically sound, TRPO requires computing the Fisher Information Matrix and executing conjugate gradient optimization with second-order approximations, making it computationally heavy and complex to implement. PPO achieves similar stability using only standard first-order backpropagation.

### The PPO-Clip Objective

Instead of an explicit hard constraint on KL divergence, PPO defines a probability ratio $r_t(\theta)$:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

At the start of an update epoch, $\theta = \theta_{\text{old}}$, so $r_t(\theta) = 1$. As the policy parameters change, $r_t(\theta)$ diverges from 1.

The clipped surrogate objective function is defined as:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

Where $\epsilon$ is a hyperparameter (typically set to $0.1$ or $0.2$).

#### How the Clipping Mechanism Operates

The objective evaluates differently depending on whether the action outperformed or underperformed the baseline:

```
L^CLIP Behavior

   Positive Advantage (A > 0)          Negative Advantage (A < 0)
        Better Action                       Worse Action

      Objective                           Objective
          ^                                   ^
          |      Clipped                      |
(1+ε)A ---+-----------------        (1-ε)A ---+-----------------
          |    /                              |    /
          |   /                               |   /   Clipped
          |  /                                |  /
        0 +---+-------->                    0 +---+-------->
             1-ε 1 1+ε   r_t                     1-ε 1 1+ε   r_t
```
- **Positive Advantage ($\hat{A}_t > 0$):**
	- The action yielded higher-than-expected returns; the optimization aims to increase $\pi_\theta(a_t \mid s_t)$, which pushes $r_t(\theta)$ above 1.
		- When $r_t(\theta) > 1 + \epsilon$, the term is clipped to $(1 + \epsilon)\hat{A}_t$.
		- The $\min$ operator ensures the gradient drops to zero for further increases, preventing the update from over-committing to a single sample batch.
- **Negative Advantage ($\hat{A}_t < 0$):**
	- The action performed worse than expected; the optimization aims to decrease $\pi_\theta(a_t \mid s_t)$, pushing $r_t(\theta)$ below 1.
		- When $r_t(\theta) < 1 - \epsilon$, the clipped term is $(1 - \epsilon)\hat{A}_t$.
		- Because $\hat{A}_t$ is negative, $(1 - \epsilon)\hat{A}_t$ is numerically larger than $r_t(\theta)\hat{A}_t$. The $\min$ operator selects the lower bound ($r_t(\theta)\hat{A}_t$), still penalizing overly aggressive decreases while cutting off runaway gradients.

### Advantage Estimation: Generalized Advantage Estimation (GAE)

PPO typically relies on Generalized Advantage Estimation (GAE) to compute $\hat{A}_t$. GAE provides an explicit bias-variance trade-off using a discount factor $\gamma \in [0, 1]$ and an exponential weight parameter $\lambda \in [0, 1]$.

First, define the temporal difference (TD) residual at time step $t$:

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The GAE estimator $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ accumulates these residuals:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

In practice, for an episode or rollout batch of length $T$, it is computed backward from step $T-1$:

$$
\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}
$$

- When $\lambda = 0$: $\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$. This is the 1-step TD advantage, which has **low variance but high bias**.
- When $\lambda = 1$: $\hat{A}_t = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} - V(s_t)$. This matches the empirical Monte Carlo return minus baseline, exhibiting **zero bias but high variance**.
- Common practical values are $\gamma = 0.99$ and $\lambda = 0.95$.

### Complete Training Objective

When the policy network (actor) and the value network (critic) share underlying parameters, or during unified joint optimization, the full loss function maximized at each iteration is:

$$
L_t^{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

Where:

- **$L_t^{\text{VF}}(\theta)$** is the squared-error loss for the critic:
	$$
	L_t^{\text{VF}}(\theta) = \left( V_\theta(s_t) - V_t^{\text{targ}} \right)^2
	$$
	*(Note: Many implementations also clip the value function updates similarly to the policy updates).*
- **$S[\pi_\theta](s_t)$** is the entropy bonus of the policy distribution to ensure adequate exploration and prevent early collapse into deterministic policies:
	$$
	S[\pi_\theta](s_t) = -\sum_{a} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
	$$
- **$c_1, c_2$** are weighting coefficients (e.g., $c_1 \approx 0.5$, $c_2 \approx 0.01$).

### Step-by-Step Training Loop

1. **Trajectory Rollout:** Run current policy $\pi_{\theta_{\text{old}}}$ across $N$ parallel environments for $T$ timesteps. Collect trajectories $\{ (s_t, a_t, r_t, s_{t+1}, \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)) \}$.
2. **Compute Values & Advantages:**
	- Evaluate state values $V(s_t)$ with the critic network.
		- Compute TD residuals $\delta_t^V$ and recursively compute advantages $\hat{A}_t$ via GAE.
		- Standardize the batch advantages across minibatches: $\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + 10^{-8}}$.
3. **Mini-Batch Optimization:**
	- Flatten data of length $N \times T$.
		- Split data into mini-batches of size $M$.
		- For $K$ epochs (typically $3 \le K \le 10$):
		- Recompute action probabilities under current parameters $\theta$: $\pi_\theta(a_t \mid s_t)$.
			- Compute ratios $r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t))$.
			- Compute $L^{\text{CLIP}}(\theta)$, critic MSE loss, and entropy bonus.
			- Backpropagate the total objective using an optimizer like Adam with learning rate annealing.
4. **Update Baseline:** Set $\theta_{\text{old}} \leftarrow \theta$ and repeat the cycle.



The number of models required depends on whether PPO is deployed for **standard reinforcement learning** (robotics, games, control) or **RLHF for Large Language Models** (RL with Human Feedback).

### 1. How Many Models Are Needed?

#### Standard RL: 2 Functional Models

Standard PPO requires an **Actor-Critic** architecture:

1. **Actor (Policy Model, $\pi_\theta$):** Maps a state $s$ to an action distribution $\pi(a \mid s)$. It selects actions during environment rollouts and is updated via the clipped objective.
    
2. **Critic (Value Model, $V_\phi$):** Maps a state $s$ to a scalar baseline $V(s)$ representing the expected cumulative discounted return. It is updated via mean squared error (MSE) to make advantage estimation accurate.
    

_Implementation Note:_ These two models can either be implemented as **two separate neural networks** (most common in complex or continuous action spaces to prevent gradient interference) or as a **single shared backbone** with two separate output heads (one for actions, one for value).

#### RLHF / LLM Fine-Tuning: 4 Models

When applying PPO to language models, 4 distinct models are loaded into memory:

1. **Actor / Policy ($\pi_\theta$):** The language model being trained. Receives a prompt, generates tokens, and updates its weights via PPO.
    
2. **Critic / Value Model ($V_\phi$):** Initialized from the reward or SFT model, but modified to output a scalar value token-by-token (or for the sequence) estimating expected future rewards.
    
3. **Reference Model ($\pi_{\text{ref}}$):** A **frozen** copy of the initial supervised fine-tuned (SFT) model. Used to calculate a token-level KL divergence penalty to ensure the policy does not drift too far or degenerate into reward hacking.
    
4. **Reward Model ($r_\psi$):** A **frozen** model trained on human preference pairs. Evaluates the completed generation and assigns a scalar quality score.
    

### 2. The Complete End-to-End PPO Flow

The workflow operates in an alternating cycle: **Data Collection (Rollout Phase)** followed by **Optimization (Update Phase)**.

```
       [ Current Policy π_θ_old ] + [ Environment / Prompts ]
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  1. Collect Trajectories   │
                    │     (s_t, a_t, r_t, s_t+1)│
                    │   & log π_θ_old(a_t|s_t)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ 2. Compute Target Values  │
                    │    - Critic: V(s_t)       │
                    │    - GAE: Advantages A_t  │
                    └─────────────┬─────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
      ┌─────────────────┐                   ┌─────────────────┐
      │  Actor Update   │                   │  Critic Update  │
      │  Ratio r_t(θ)   │                   │  MSE:           │
      │  Clip Objective │                   │  (V_θ(s) - y)^2 │
      └────────┬────────┘                   └────────┬────────┘
               │                                     │
               └──────────────────┬──────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ 3. Mini-Batch SGD Updates │
                    │    (Run for K Epochs)     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ 4. Update Policy & Repeat │
                    │     θ_old ← θ             │
                    └───────────────────────────┘
```

### Step-by-Step Execution

#### Phase 1: Trajectory Rollout (Interaction)

1. **Action Generation:** The agent starts in state $s_t$. The actor network computes the action distribution $\pi_{\theta_{\text{old}}}(a \mid s_t)$.
    
2. **Sampling & Action Execution:** An action $a_t$ is sampled from this distribution and executed in the environment.
    
3. **Recording Transitions:** The environment returns the immediate reward $r_t$ and the next state $s_{t+1}$. The transition tuple is cached:
    
    $$\{s_t, \, a_t, \, r_t, \, s_{t+1}, \, \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\}$$
    
4. **Batch Accumulation:** This interaction runs for a fixed trajectory horizon of $T$ steps (often across $N$ parallel environments simultaneously, yielding $N \times T$ total transitions).
    

#### Phase 2: Advantage & Return Computation

Before training begins on the collected rollout:

1. **Value Evaluation:** The critic network evaluates all visited states to produce baseline value estimates:
    
    $$\hat{v}_t = V_\phi(s_t)$$
    
2. **Temporal Difference Residuals:** Compute the 1-step TD errors for all steps:
    
    $$\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
    
3. **Generalized Advantage Estimation (GAE):** Accumulate the residuals backwards from $t = T-1$ down to $t = 0$:
    
    $$\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}$$
    
    _(In LLM workflows, $r_t$ includes both the reward model score at the final token and per-token KL divergence penalties against $\pi_{\text{ref}}$)._
    
4. **Target Returns:** Target values for the critic are constructed by adding the estimated advantage back to the baseline:
    
    $$R_t^{\text{target}} = \hat{A}_t + V_\phi(s_t)$$
    
5. **Advantage Normalization:** Across the entire collected batch, advantages are normalized to zero mean and unit variance:
    
    $$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + 10^{-8}}$$
    

#### Phase 3: The Multi-Epoch Optimization Loop

The fixed dataset of $N \times T$ transitions is shuffled and divided into smaller mini-batches of size $M$. The model iterates over this same rollout dataset for $K$ epochs (typically 3 to 10 passes):

1. **Forward Pass on Updated Parameters:** For each sample in the mini-batch, pass $s_t$ through the _current_ parameter states $\pi_\theta$ and $V_\phi$ to obtain:
    
    - Current log-probability: $\log \pi_\theta(a_t \mid s_t)$
        
    - Current estimated state value: $V_\phi(s_t)$
        
    - Distribution entropy: $S[\pi_\theta](s_t)$
        
2. **Compute Probability Ratio:**
    
    $$r_t(\theta) = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \right)$$
    
3. **Compute Clipped Policy Loss:**
    
    $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t \right) \right]$$
    
4. **Compute Critic Loss:** Fit the value function to the target returns:
    
    $$L^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t \left[ \left( V_\phi(s_t) - R_t^{\text{target}} \right)^2 \right]$$
    
5. **Combine & Backpropagate:** Sum the losses into a single scalar objective to minimize:
    
    $$L_{\text{total}} = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 S[\pi_\theta](s_t)$$
    
    Compute gradients via standard backpropagation, apply gradient clipping (e.g., $\max \Vert{}\nabla\Vert{} \le 0.5$), and step the optimizer (Adam).
    

#### Phase 4: Sync and Discard

- Once $K$ epochs conclude, the old rollout transitions $\{s_t, a_t, r_t, \dots\}$ are **completely discarded**.
    
- Update the baseline weights: $\theta_{\text{old}} \leftarrow \theta$.
    
- The updated policy $\pi_\theta$ takes over to gather a fresh batch of trajectories in Phase 1.
