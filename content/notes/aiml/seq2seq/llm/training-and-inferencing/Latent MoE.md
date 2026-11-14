---
title:
source: "https://gemini.google.com/app/e01da2f969d4678e"
author:
published:
created: 2026-11-14
description:
tags:
---
Latent Mixture-of-Experts (Latent MoE) is an architectural pattern where the routing and execution of expert networks occur in a **compressed, low-dimensional latent space** rather than the full hidden dimension of the model.

Sebastian Raschka

Introduced to bypass the memory bandwidth and compute bottlenecks of scaling standard MoEs, it allows models to consult significantly more experts per token while maintaining or reducing the active parameter footprint. It has become a foundational architecture in mid-2026 models like NVIDIA's Nemotron 3 (Super and Ultra) and Moonshot's Kimi K3.

Sebastian Raschka\+ 1

![Latent MoE Projection and Routing, AI generated](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6XBq2nmk_NVfjNByO7KCGJSIIGlBCKDuh2C5hlLLfR5U41sRHtwF3EZAf&s=10)

Latent MoE Projection and Routing. Source: Sebastian Raschka

### The Mechanics: Standard vs. Latent MoE

In a **Standard MoE** block, a token embedding at full dimension $d$ is passed directly to the router and the selected experts. If the FFN intermediate dimension is $h$, each selected expert performs dense $d \to h \to d$ matrix transformations.

In a **Latent MoE** block, the computation follows three stages:

1. **Down-Projection:** A dense linear layer projects the token embedding from the full dimension $d$ to a narrower latent dimension $d_{latent}$ (often a 2x to 4x compression ratio).
	Sebastian Raschka
2. **Latent Routing & Execution:** The router evaluates the compressed token. The selected experts execute in this narrower space, performing $d_{latent} \to h_{latent} \to d_{latent}$ transformations.
	Sebastian Raschka
3. **Up-Projection:** The weighted sum of the expert outputs is projected back to the full dimension $d$ via another dense linear layer.
	Sebastian Raschka

*(Note: Just like DeepSeek-V3, Latent MoE setups usually preserve a dense **shared expert** that processes all tokens to capture common knowledge, preventing the sparse experts from wasting capacity on redundant baseline concepts).*

Sebastian Raschka

### Why Latent MoE? The Engineering Economics

Because the primary bottleneck in MoE decoding isn't FLOPs but **HBM read bandwidth**, fetching the weights of selected experts for every token is painfully slow. Latent MoE attacks this directly:

- **Shrinking the Experts:** By shrinking the input/output dimension of the experts by a factor of $c$ (e.g., $c=4$), the parameter count of each individual expert drops quadratically ($O(d_{latent}^2)$ vs $O(d^2)$).
- **Finer Granularity:** Because each expert is computationally cheaper and smaller in memory, the model can afford to instantiate hundreds of them and route to *more* of them per token without blowing up the inference budget.
- **Isolating the Compute Bottleneck:** The down- and up-projections ($W_{down}$ and $W_{up}$) are shared across all tokens. They are dense matrix multiplications, which GPU Tensor Cores chew through efficiently. You trade a slight increase in dense FLOPs (the projections) for a massive reduction in the sparse memory reads (the experts).

### 2026 Implementations

| Model | Bottleneck Ratio | Dimensionality Path | Routed Experts | Active per Token | Architecture Notes |
| --- | --- | --- | --- | --- | --- |
| **Nemotron 3 Super (120B)** | 4x | 4096 → 1024 → 4096 | \- | \- | Hybrid Mamba-2 + Transformer |
| **Nemotron 3 Ultra (550B)** | 4x | 8192 → 2048 → 8192 | 512 | 22 | Shared expert at dim 10,240 |
| **Kimi K3** | 2x | 7168 → 3584 → 7168 | 896 | 16 | Uses RMSNorm before up-projection |

**Trade-offs to Consider:**A 4x latent bottleneck does **not** mean a 4x end-to-end speedup. The projection matrices add dense compute overhead to every token. Furthermore, over-compressing the latent space risks information bottlenecking—similar to the rank-collapse risks in low-rank LoRA adapters—forcing models to rely heavily on the shared expert or adjacent dense attention/Mamba layers to carry high-fidelity token state.