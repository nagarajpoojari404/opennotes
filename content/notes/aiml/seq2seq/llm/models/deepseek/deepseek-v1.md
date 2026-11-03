---
title: "Deepseek-v1"
date: "2026-10-31"
---

**DeepSeek-LLM: Scaling Open-Source Language Models with Longtermism** (January 5, 2024) serves as DeepSeek’s foundational technical paper. It established both the empirical methodology and the engineering infrastructure that later underpinned models like DeepSeekMoE, DeepSeek-V2/V3, and DeepSeek-R1.

### 1\. Motivation: Why "Longtermism"?

At the time of release, many open-source models scaled parameters by strictly copying LLaMA or following standard Chinchilla-scaling recipes without testing the limits of data-to-parameter saturation. DeepSeek approached scaling with two specific long-term goals:

1. **Rethink Scaling Laws Under Variable Data Quality:** Systematically derive how hyperparameter selection (batch size, learning rate) and compute-optimal parameter-to-data allocation shift as data quality improves.
2. **Build Continuous Pre-Training Infrastructure:** Move away from one-off, static model checkpoints by introducing training dynamics (such as multi-step learning rate schedules) that allow weights to be reused and expanded as more high-quality data arrives.

### 2\. Empirical Scaling Laws & Key Findings

#### A. Hyperparameter Scaling Laws

Before scaling up to 7B and 67B parameter models, the authors evaluated hundreds of small-scale training runs across compute budgets from $10^{17}$ to $10^{20}$ FLOPs to establish exact power-law trends for learning rate ($\eta$) and batch size ($B$):

- **Optimal Batch Size ($B$):** Monotonically grows as compute budget ($C$) increases.
- **Optimal Learning Rate ($\eta$):** Monotonically decays as compute budget ($C$) increases.
- **Tolerance Basin:** Generalization loss remained within 0.25% of the theoretical optimum across a relatively broad hyperparameter envelope, giving engineering flexibility when fitting models onto cluster hardware.

#### B. The Impact of Data Quality on Scaling Laws

A major discovery in this paper was that **data quality fundamentally changes the optimal ratio between parameters ($N$) and data tokens ($D$)**:

- When data quality is relatively low or noisy, optimal compute allocation favors scaling **model size** faster than token count, as larger models better absorb noisy signals.
- As data quality improves via aggressive filtering, deduplication, and synthetic curation, the optimal frontier shifts: the model can extract information efficiently from significantly **more tokens per parameter** without premature loss saturation.
- This justified training their 7B and 67B models on 2 trillion tokens—well beyond the canonical 20 tokens-per-parameter Chinchilla ratio.

### 3\. Model Architecture & Pipeline Design

DeepSeek adopted a modern dense Transformer backbone with targeted structural modifications:

| Specification | DeepSeek 7B | DeepSeek 67B |
| --- | --- | --- |
| **Layers** | 30 | 95 |
| **Hidden Dimension ($d_{\text{model}}$)** | 4096 | 8192 |
| **Attention Heads ($Q/KV$)** | 32 / 32 (MHA) | 64 / 8 (GQA) |
| **Intermediate FFN Dim** | 11,008 | 27,648 |
| **Context Length** | 4,096 tokens | 4,096 tokens |
| **Vocabulary Size** | 102,400 tokens | 102,400 tokens |

#### Structural Highlights:

- **95-Layer Depth for 67B:** Rather than using the typical 80-layer layout seen in models like LLaMA-2 70B, DeepSeek opted for 95 layers. This odd layer depth was specifically selected to divide evenly across 3D-parallel pipeline stages while favoring depth-over-width inductive bias for symbolic and logical reasoning.
- **Grouped-Query Attention (GQA):** 67B used 8 KV heads with 64 query heads (8:1 ratio) to cut KV cache memory consumption during decoding, whereas the 7B model retained standard Multi-Head Attention (MHA).
- **Pre-Norm + SwiGLU + RoPE:** Followed standard pre-normalization with RMSNorm, SwiGLU non-linearities, and Rotary Position Embeddings.

### 4\. Training Dynamics & The Multi-Step Learning Rate Schedule

Most pre-trained LLMs utilize a Cosine Learning Rate schedule with warmup, which decays the learning rate to near-zero at a pre-set step count. The paper points out two drawbacks of cosine decay:

1. It commits the team to an exact token budget upfront.
2. Extending training past the schedule's end requires clumsy restarts or causes loss instability.

To support their "longtermism" philosophy, DeepSeek introduced a **Multi-Step Learning Rate Scheduler** for pre-training:

- **Warmup:** Linear warmup over the first 2,000 steps to peak LR.
- **Constant High LR:** Maintained steady peak learning rate through 80% of the training budget.
- **Step 1 Decay:** At 80% of tokens, LR steps down to **31.6%** ($1/\sqrt{10}$) of maximum.
- **Step 2 Decay:** At 90% of tokens, LR steps down to **10%** of maximum.

The authors proved that this multi-step schedule achieves a final validation loss virtually identical to a cosine schedule, while keeping the model in a high-learning state for the majority of the training run so additional token sets can be appended without destabilizing convergence.

### 5\. Pre-Training Corpus & Alignment (SFT + DPO)

- **2.0 Trillion Tokens:** Pre-trained on a bilingual dataset (English and Chinese). The data pipeline was organized into three distinct phases: **deduplication**, **filtering** (linguistic heuristics and information-density scoring), and **remixing** (balancing code, scientific text, and informal forum discussions).
- **Supervised Fine-Tuning (SFT):** Fine-tuned across ~1.5 million instruction instances covering coding, conversational dialogue, reasoning, and multi-turn math.
- **Direct Preference Optimization (DPO):** DeepSeek applied DPO directly to the SFT base using a batch size of 512 and a learning rate of $5 \times 10^{-6}$. They found that DPO notably boosted open-ended generation, nuance, and human evaluation scores while preserving performance on standard deterministic benchmarks.

### 6\. Results and Benchmark Impact

- **Beating LLaMA-2 70B:** Despite having slightly fewer parameters (67B vs. 70B) and being trained on 2T tokens, DeepSeek-LLM 67B outperformed LLaMA-2 70B across coding (HumanEval), mathematics (GSM8K, MATH), and standard reasoning benchmarks.
- **Chat Competitiveness:** In Chinese and English open-ended evaluations, DeepSeek-LLM 67B Chat surpassed GPT-3.5, marking DeepSeek’s entrance as a top-tier open-weight model developer.
