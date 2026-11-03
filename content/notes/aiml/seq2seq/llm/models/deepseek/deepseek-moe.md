---
title: "Deepseek-moe"
date: "2026-10-31"
---

The transition from the dense DeepSeek-LLM 67B to **DeepSeekMoE** marks the architectural pivot that defined all of DeepSeek’s later flagship models (DeepSeek-V2, V3, and R1).

While standard Mixture-of-Experts (MoE) architectures—like Mixtral 8x7B or Switch Transformer—simply swap dense Feed-Forward Networks (FFNs) for a small bank of large sub-networks, DeepSeek discovered severe structural flaws in this setup. DeepSeekMoE re-architected the MoE layer around two core principles: **fine-grained expert segmentation** and **shared expert isolation**.

### 1\. The Bottlenecks of Traditional MoE Architectures

Conventional MoEs divide the FFN layer into $N$ experts (typically $N = 8$) and activate the top-$K$ experts (typically $K = 2$) per token. DeepSeek identified two fundamental limitations in this paradigm:

1. **Knowledge Hybridity (Lack of Specialization):** Because $K=2$ is so small, each routed expert is forced to learn a vast, overlapping mixture of broad concepts (e.g., syntax, grammar, basic facts) rather than truly specializing in narrow domains (e.g., Python recursion, organic chemistry, or legal reasoning).
2. **Combinatorial Constraint:** With $N=8$ and $K=2$, the router can only produce:
	$$
	\binom{8}{2} = 28 \text{ possible expert combinations}
	$$
	This severely caps the expressive capacity of the model for diverse tokens.
3. **Knowledge Redundancy:** Multiple experts end up wasting capacity re-learning the exact same ubiquitous, cross-domain background knowledge.

### 2\. The Core Mechanics of Fine-Grained Routing

DeepSeek addressed this by dividing the standard expert size into significantly smaller slices and activating proportionally more of them, keeping total activated FLOPs constant.

```
Traditional MoE (e.g., Mixtral 8x7B):
  Input Token ──► [ Router: Top-2 of 8 ] ──► [ Large Expert 1 ] + [ Large Expert 4 ]
  Combinations: 28

DeepSeekMoE Fine-Grained:
  Input Token ──► [ Shared Experts (Always ON) ] ───────────────┐
              └──► [ Router: Top-m of N (e.g., Top-8 of 64) ] ──┴──► Combined Output
  Combinations: > 4.4 Billion
```

#### A. Fine-Grained Expert Segmentation

Instead of assigning an intermediate hidden dimension $d_{\text{ffn}}$ to a single expert, DeepSeek divides it by a splitting factor $m$:

- Each expert's intermediate dimension becomes $\frac{d_{\text{ffn}}}{m}$.
- To preserve the exact same compute budget per token, the number of activated routed experts scales from $K$ to $m \times K$.
- Total available experts increases from $N$ to $m \times N$.

**The Combinatorial Explosion:**

If $m = 4$, an $N=16, K=2$ system becomes $N=64, K=8$:

$$
\binom{64}{8} \approx 4.42 \times 10^9 \text{ combinations}
$$

By expanding the pool of available pathways from **28 to over 4.4 billion**, the model can assemble specialized micro-circuits custom-tailored to nuanced token contexts, drastically reducing knowledge hybridity.

#### B. Shared Expert Isolation

To eliminate redundancy, DeepSeek isolates a fixed set of $K_s$ **shared experts** that remain permanently active for every single token, bypassing the router entirely:

$$
\mathbf{h}_t = \mathbf{u}_t + \sum_{i=1}^{K_s} \text{FFN}_i^{\text{shared}}(\mathbf{u}_t) + \sum_{j \in \text{Top-}K_r} g_{j,t} \cdot \text{FFN}_j^{\text{routed}}(\mathbf{u}_t)
$$

Where:

- $\mathbf{u}_t$ is the input representation of token $t$.
- $\text{FFN}_i^{\text{shared}}$ processes universal common-sense patterns, punctuation, grammar, and generic sentence framing.
- $\text{FFN}_j^{\text{routed}}$ contains the fine-grained routed experts, which now only activate for domain-specific features.
- $g_{j,t}$ is the normalized routing weight computed over the selected top-$K_r$ candidates.

### 3\. Transitioning the Architecture: Dense 67B to MoE

To prove the paradigm, DeepSeek designed comparative models directly against the DeepSeek-LLM dense baselines:

| Architecture Metric | DeepSeek-LLM (Dense Baseline) | DeepSeekMoE 16B | Scaled DeepSeekMoE (V2 Base) |
| --- | --- | --- | --- |
| **Total Parameters** | 7B (or 67B) | 16.4B | 236B |
| **Activated Parameters / Token** | 7B (or 67B) | **2.8B** | **21B** |
| **Routing Setup** | None (Dense FFN) | 1 Shared + 64 Routed (Top-8 active) | 2 Shared + 160 Routed (Top-6 active) |
| **Compute Efficiency** | Baseline ($1\times$) | **~40% compute of 7B dense** | **~30% compute of 67B dense** |
| **Performance Benchmark** | Baseline | **Matches/beats 7B dense** | **Surpasses 67B dense** |

#### Why the Transition Succeeded:

1. **Parameter Decoupling:** In the dense 67B model, scaling knowledge required scaling FLOPs linearly on every token. With fine-grained MoE, DeepSeek decoupled memory capacity (total parameters) from inference execution cost (active parameters).
2. **No Routing Collapse:** Smaller, fine-grained experts combined with dedicated shared experts eliminated routing bottlenecks where one or two "celebrity experts" hog all the tokens, leading to far more uniform token distribution across hardware devices.

This fine-grained routing blueprint, validated in the January 2024 paper, served directly as the execution engine for DeepSeek-V2 and V3, allowing V3 to scale to 671B total parameters while activating only 37B per token.
