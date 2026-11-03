---
title: "Techniques"
date: "2026-10-30"
---

LLM training optimization spans distributed parallelization, memory management, kernel-level execution, numerical precision, and algorithmic architecture.

Medium

### 1\. Distributed Parallelism Strategies

- **Tensor Parallelism (TP):** Splits individual matrix multiplications across GPUs within a single node (e.g., Megatron-LM splitting MLP and Attention weight matrices column-wise and row-wise). Requires low-latency interconnects (NVLink).
- **Pipeline Parallelism (PP):** Partitions layers sequentially across multiple nodes (e.g., GPipe, Megatron-LM 1F1B schedule). Uses micro-batching to reduce pipeline bubbles.
- **Context Parallelism (CP / RingAttention):** Shards the sequence dimension across devices to process ultra-long contexts ($64\text{k}+$ to $1\text{M}+$ tokens) without exhausting single-device memory.
- **Expert Parallelism (EP):** For Mixture-of-Experts (MoE) models, routes token embeddings dynamically to specialized feed-forward experts partitioned across different ranks via All-to-All communication.
- **3D/4D Parallelism:** Combines DP, TP, PP, and CP to scale training across tens of thousands of GPUs.

### 2\. Memory Reduction & Sharding

- **ZeRO (Zero Redundancy Optimizer) / FSDP:**
	- **Stage 1:** Shards optimizer states across data-parallel ranks (saves $\approx 4\times$ memory).
	- **Stage 2:** Shards optimizer states and gradients (saves $\approx 8\times$ memory).
	- **Stage 3 / Full FSDP:** Shards optimizer states, gradients, and model parameters. Dynamically gathers parameters on-demand during forward/backward passes and discards them immediately.
- **Activation / Gradient Checkpointing:** Discards intermediate forward-pass activations and recomputes them during the backward pass, trading $\sim 20\text{–}30\%$ compute overhead for drastic activation memory reduction.
- **Selective Activation Recomputation:** Recomputes only memory-heavy, compute-cheap operations (such as attention score softmax/dropout) while caching compute-heavy GEMM activations.
- **CPU / NVMe Offloading (ZeRO-Offload):** Offloads optimizer states and infrequently accessed weights from high-bandwidth HBM to host memory during step updates.

### 3\. Kernel & Compute Acceleration

- **FlashAttention (v1, v2, v3):** IO-aware tiling algorithms that compute exact scaled dot-product attention in fast on-chip SRAM. Avoids materializing the $O(N^2)$ attention matrix in high-bandwidth memory (HBM), drastically cutting memory bandwidth bottlenecks.
- **Operator Fusion:** Merges adjacent pointwise, normalization, and linear operations (e.g., RMSNorm + GEMM, SwiGLU, fused AdamW) to eliminate redundant kernel launches and intermediate global-memory round-trips.
- **Deep Learning Compilers:** Frameworks like PyTorch Dynamo/`torch.compile`, Triton, and XLA generate optimized, hardware-specific C++/CUDA kernels with automatic layout transformations and graph fusions.
- **Communication Overlap:** Hides collective communication overhead (`AllReduce`, `ReduceScatter`, `AllGather`) behind independent compute kernels using CUDA streams and double-buffering.

### 4\. Precision & Numerical Formats

- **Mixed Precision (AMP with BF16 / FP16):** Stores weights, activations, and gradients in 16-bit while tracking optimizer states and weight master copies in FP32. BF16 is preferred over FP16 due to its dynamic range matching FP32, eliminating manual loss scaling.
- **FP8 Training:** Utilizes E4M3 (forward activations/weights) and E5M2 (backward gradients) formats supported on newer accelerator architectures (Hopper, Blackwell) to halve HBM footprints and double Tensor Core throughput.
- **Stochastic Rounding:** Mitigates gradient underflow and quantization error when accumulating low-bit gradient updates into low-precision master weights.

### 5\. Architectural & Algorithmic Optimizations

- **Attention Variants:**
	- **Multi-Query Attention (MQA) & Grouped-Query Attention (GQA):** Shares key/value projections across heads, decreasing memory traffic and cache footprint during autoregressive rollout and training.
		- **Sliding Window / Local Attention:** Restricts attention computation to a localized context window.
- **Optimizers:**
	- **Memory-Efficient Optimizers:** Adafactor, Lion, and 8-bit Adam (bitsandbytes) compress or factorize optimizer second-moment accumulators from 8 bytes/param down to 1–2 bytes/param.
		- **Muon / Shampoo:** Second-order or orthogonalized momentum optimizers that accelerate convergence per token compared to standard AdamW.
- **Data Packing:** Concatenates multiple short token sequences into a single fixed-length window separated by attention masks to eliminate padding-token compute waste.
- **Curriculum & Variable-Length Batching:** Schedules sequence lengths progressively (e.g., training on $2\text{k}$ tokens before expanding to $8\text{k}\text{+}$) to minimize compute early in training.

### 6\. Parameter-Efficient Fine-Tuning (PEFT)

When adapting pre-trained models rather than pretraining from scratch:

| Method | Mechanism | Parameter Footprint |
| --- | --- | --- |
| **LoRA / QLoRA** | Injects low-rank decomposition matrices ($W_0+BA$) into attention/MLP layers; QLoRA quantizes the base model to 4-bit NormalFloat (NF4). | $\ll 1\%$ trainable params |
| **DoRA** | Decomposes weight matrices into magnitude and directional components, applying LoRA solely to the direction vector. | $\ll 1\%$ trainable params |
| **Prefix Tuning / Prompt Tuning** | Prepends trainable continuous virtual tokens to key/value or input embeddings. | $<0.1\%$ trainable params |
