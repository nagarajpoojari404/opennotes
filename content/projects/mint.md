---
title: "Mint: Distributed Compute-Optimal LLM Training Framework"
description: "A distributed LLM pre-training, SFT, and DPO framework compute-optimized for Apple Silicon (Metal) and NVIDIA GPUs using ZeRO-2/3, tensor and sequence parallelism, and MuonAdamW."
date: "2025-06-01T00:00:00+05:30"
categories: ["aiml", "distributed-training"]
tags: ["llm", "distributed-ml", "pytorch", "metal", "dpo", "sft", "zero-2", "zero-3", "tensor-parallel", "muon"]
summary: "Mint is a distributed, compute-optimal training and alignment framework for Large Language Models. Supports pre-training, SFT, and DPO across Apple Metal and CUDA GPUs with ZeRO-2/3 sharding and tensor parallelism."
---

[**View on GitHub**](https://github.com/nagarajpoojari404/mint) · [**Benchmarks**](#benchmarks) · [**Architecture**](#architecture--modules) · [**Quickstart**](#quickstart--cli-commands) · [**Key Innovations**](#what-makes-mint-special)

**Mint** is a high-performance, compute-optimal distributed training, fine-tuning, and alignment framework engineered for Large Language Models. Designed to achieve maximum parameter throughput at lowest hardware cost, Mint runs seamlessly across both Apple Silicon (MPS / Metal) and high-density NVIDIA CUDA GPU clusters.

---

## Benchmarks

Parallelism benchmark on **2 x Tesla T4** (15 GB each, PCIe, FP16 autocast, sequence length 128), comparing a single GPU, DDP with ZeRO-0/1/2/3, tensor parallelism (TP) and TP with sequence parallelism (SP) across six model shapes (8M to 142M params).

- **DDP (ZeRO-0)** was the fastest multi-GPU mode in 12/12 cells, reaching **1.70x to 1.87x** speedup over one GPU, up to **255k tok/s** and **20.6% MFU**.
- **ZeRO-1 / ZeRO-2** cut peak memory by about **3%** versus DDP-0, while **ZeRO-3** used about **30% more** at this small scale.
- Best data-parallel mode averaged **1.52x** the throughput of the best tensor-parallel mode; adding SP changed TP throughput by **0.70x** on average.

{{< iframe src="benchmarks/mint/dashboard.html" title="Mint parallelism benchmark" height="900px" >}}

---

## Architecture & Modules

### Module Breakdown

| Package Module | Purpose |
| :--- | :--- |
| **`mint.nn`** | Transformer architectures, RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU MLP layers |
| **`mint.optim`** | Distributed **MuonAdamW** optimizer with `norMuon` normalization and Polar Express coefficients, alongside ZeRO-2 sharding |
| **`mint.data`** | Zero-memory-bloat streaming chunk dataloaders for pre-training and preference datasets |
| **`mint.trainer`** | Distributed multi-GPU / multi-device training orchestrator supporting PyTorch DDP and Apple Metal (MPS) |
| **`mint.sft`** | Supervised fine-tuning pipeline for custom models and HuggingFace backbones (e.g. Qwen2, Llama) |
| **`mint.eval`** | Distributed evaluation suite covering conversational benchmarks, GSM8K reasoning, and MCQ tasks |
| **`mint.kvcache`** | Low-overhead Key-Value cache implementation for high-throughput inference and validation |

---

## What Makes Mint Special?

### 1. Automated Compute Optimality
Mint automatically calculates scaling-law optimal parameters (dataset token count, layer depth, hidden dimension, and learning rate schedules) based on the user's available FLOP budget and cluster memory constraints.

### 2. Distributed MuonAdamW Optimizer
Integrates the state-of-the-art **Muon** matrix optimizer alongside AdamW, utilizing `norMuon` normalization with Polar Express coefficients for faster convergence and lower loss variance on modern transformer architectures.

### 3. ZeRO-2/3 Sharding, Tensor & Sequence Parallelism
- **ZeRO-2 / ZeRO-3 Sharding**: Partitions optimizer states, gradients (and, with ZeRO-3, parameters) across ranks to train larger models on memory-constrained GPUs.
- **Tensor / Sequence Parallelism**: Column/row-parallel linear layers on a 2-D `dp` x `tp` device mesh, with optional sequence parallelism.
- **Compute-Optimal Configs**: `mint gen-config` derives model size, token budget, batch size and LR from a FLOPs budget.
- **Custom Streaming Dataloaders**: Streams pre-tokenized shards directly from disk or cloud buckets without memory bloat.

### 4. Cross-Platform Acceleration
- **Apple Silicon (MPS)**: Native Metal Performance Shaders integration for local M-series Mac training and fine-tuning.
- **NVIDIA CUDA**: Supports PyTorch Distributed Data Parallel (DDP), FlashAttention-2, and `bfloat16` mixed precision.

---

## Quickstart & CLI Commands

### Installation

```bash
# Clone and install Mint
git clone https://github.com/nagarajpoojari404/mint.git
cd mint
uv pip install -e .
```

### Pre-Training

```bash
# Pre-train a custom transformer model
uv run mint pretrain --config configs/config_d12_pretrain.toml

# Stream training across sharded datasets
uv run mint stream-train \
    --num-train-shards 4 \
    --min-shards 1 \
    --train-script scripts/pretrain.py \
    --train-args "--config configs/config_d12_pretrain.toml"
```

### Supervised Fine-Tuning (SFT)

```bash
# SFT on custom architectures
uv run mint sft --config configs/config_d12_sft.toml

# SFT on HuggingFace models (e.g. Qwen2, Llama) with curated datasets
uv run mint v2-sft \
    --config configs/config_hf_sft.toml \
    --model-name Qwen/Qwen2-0.5B \
    --datasets gsm8k smoltalk
```

### Direct Preference Optimization (DPO)

```bash
# DPO alignment against a reference model
uv run mint dpo \
    --config configs/config_d12_dpo.toml \
    --ref-model checkpoints/checkpoint_step_10.pt

# HuggingFace DPO
uv run mint v2-dpo \
    --model-name Qwen/Qwen2-0.5B \
    --config configs/config_hf_dpo.toml
```

---

## Dataset Utilities Included

- `mint prepare-dpo`: Prepares preference pairs (chosen vs. rejected responses).
- `mint download-climbmix`: Downloads and tokenizes curated reasoning and code pre-training datasets.
- `mint --help`: Lists all CLI commands and configuration options.
