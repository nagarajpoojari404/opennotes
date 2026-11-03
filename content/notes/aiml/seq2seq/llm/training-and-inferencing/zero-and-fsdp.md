---
title: "ZeRO and FSDP"
date: "2026-11-01"
---

Zero Redundancy Optimizer (ZeRO) and Fully Sharded Data Parallelism (FSDP) are memory optimization techniques that scale Data Parallelism by slicing a model's memory footprint across multiple GPUs, enabling the training of models far larger than a single GPU's VRAM limits.

![Memory reduction across ZeRO Stages, AI generated](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtI5ccSwrcMsexdxuNE0Lxqegt_DUrLjW_y_guvpnkADI9DlaekXv81AQ&s=10)

Memory reduction across ZeRO Stages. Source: Medium

**ZeRO Stage 1 (Optimizer State Partitioning)** In standard data parallelism, every GPU holds a full copy of the optimizer states (like Adam's momentum and variance), consuming massive memory. Stage 1 partitions these states across the cluster. Each GPU holds and updates only its assigned slice of the optimizer state.

**ZeRO Stage 2 (Gradient Partitioning)** Builds on Stage 1 by also partitioning the gradients. During the backward pass, a GPU computes gradients and retains only the specific slice it is responsible for updating, immediately discarding the rest to free memory.

**ZeRO Stage 3 (Parameter Partitioning)** The most aggressive form of ZeRO, dividing the actual model weights across all GPUs. No single GPU holds the full model. During forward and backward passes, a GPU dynamically fetches the required parameter slices from other GPUs using an *All-Gather* operation, computes the layer, and instantly discards the fetched weights.

**ZeRO-Offload and ZeRO-Infinity** Extensions of the ZeRO architecture that move optimizer states, gradients, or parameters out of GPU VRAM and onto CPU RAM or NVMe storage. This trades communication speed for massive capacity, allowing trillion-parameter model training on limited hardware.

**Fully Sharded Data Parallelism (FSDP)** FSDP is PyTorch's native, highly optimized implementation of ZeRO Stage 3. It shards parameters, gradients, and optimizer states across data-parallel workers. FSDP wraps individual layers (like transformer blocks) and automatically manages the complex *All-Gather* (fetching weights before computation) and *Reduce-Scatter* (synchronizing gradients after computation) operations under the hood, making it the standard choice for modern PyTorch-based LLM training.
