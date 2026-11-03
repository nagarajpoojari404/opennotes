---
title: "Scaling laws"
date: "2026-10-29"
---

The two foundational papers that established and refined LLM scaling laws are:

1. **Kaplan et al. (OpenAI, 2020):** *"Scaling Laws for Neural Language Models"*
	GitHub
2. **Hoffmann et al. (DeepMind, 2022):** *"Training Compute-Optimal Large Language Models"* (commonly known as the **Chinchilla** paper)
	Analytics Vidhya

Below is a detailed synthesis of all the major insights from both papers.

## 1\. Kaplan et al. (OpenAI, 2020) Insights

Kaplan et al. provided the first empirical foundation showing that language model performance improves predictably according to strict **power laws** across many orders of magnitude.

### Key Insights:

- **Performance is highly predictable:** Test loss ($L$) scales smoothly as a power law with compute budget ($C$), model size ($N$), and dataset size ($D$):
	GitHub
	$$
	L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}
	$$
- **Architecture shape barely matters:** Specific design details—such as depth vs. width, feed-forward dimensions, or number of attention heads—have minimal effect on final performance compared to overall parameter count.
	GitHub
- **Bigger models are more sample-efficient:** Larger models learn faster and achieve a target loss using fewer total tokens than smaller models.
	GitHub
- **"Bigger is Better" allocation (The Flaw):** Kaplan et al. concluded that when compute increases, most of the new budget should go toward **increasing model size ($N$)**, while data size ($D$) needs to grow only slowly.
	arXiv
	- *Their rule:* Increase model size by $\sim 73\%$ and dataset size by only $\sim 27\%$ for each doubling of compute.
- **Early stopping & undertraining:** They suggested it is optimal to train massive models for relatively few steps and stop training long before full convergence.
	GitHub
- **Downstream capability gap:** Smooth, continuous scaling predicts cross-entropy loss improvement, but does not explicitly predict qualitative "emergent" capabilities (like few-shot reasoning).

## 2\. Hoffmann et al. / Chinchilla (DeepMind, 2022) Insights

DeepMind re-examined Kaplan et al.'s findings by training over **400 models** (ranging from 70M to 16B parameters on up to 500B tokens). They discovered that OpenAI's original learning rate schedules and hyperparameter setups artificially penalized smaller models trained on large token counts, leading to flawed conclusions.

GitHub

### Key Insights:

- **The 1:1 Equal Scaling Law:** For optimal training compute efficiency, **model size ($N$) and training data ($D$) must scale at equal rates (1:1)**.
	- If you double the compute budget, you should double **both** the parameter count and the number of tokens.
		Analytics Vidhya
- **The ~20:1 Golden Ratio:** A model trained to compute-optimal efficiency requires roughly **20 tokens per parameter** ($D \approx 20N$).
	Analytics Vidhya
- **Massive LLMs were severely undertrained:** Most high-profile LLMs built between 2020 and 2022 (e.g., GPT-3, Gopher, Megatron-Turing NLG) were far too large relative to the amount of data they were trained on.
	Michael Brenndoerfer
- **Proof with Chinchilla (70B):** DeepMind built **Chinchilla**, a 70B parameter model trained on 1.4 trillion tokens. Despite having **$2.5\times$ fewer parameters** than GPT-3 (175B) and Gopher (280B), Chinchilla outperformed both models across virtually every benchmark while using substantially less training compute.
	Analytics Vidhya
- **Mathematical Loss Equation:** They formalized the loss function incorporating an irreducible loss ($E$):
	$$
	L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
	$$
	where optimal parameters scale as $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$, finding $a \approx 0.5$ and $b \approx 0.5$.

## Direct Comparison & Modern Impact

| Metric / Dimension | Kaplan et al. (OpenAI 2020) | Hoffmann et al. / Chinchilla (DeepMind 2022) |
| --- | --- | --- |
| **Compute Scaling Strategy** | Favor Model Size ($N\propto C^{0.73}$) over Data ($D\propto C^{0.27}$) | Equal Proportional Scaling ($N\propto C^{0.5}$, $D\propto C^{0.5}$) |
| **Token-to-Parameter Ratio** | ~1.7 to 5 tokens per parameter | **~20 tokens per parameter** |
| **Model Paradigm Example** | GPT-3 (175B params, 300B tokens) | Chinchilla (70B params, 1.4T tokens) |
| **Primary Takeaway** | Train larger models and stop early. | Train smaller models on much more data. |

### Legacy & Post-Chinchilla Era:

- **Inference-Optimal Over-training:** While Chinchilla defined *compute-optimal training*, modern open-weights models (like Llama 3 8B trained on 15T+ tokens—a **1,800:1** ratio) intentionally break Chinchilla laws. They pay a higher **training cost** upfront to get a smaller model that is substantially **cheaper and faster to run at inference** across millions of user
