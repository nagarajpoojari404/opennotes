---
title: "Context parallelism"
date: "2026-11-03"
---

**Context Parallelism (CP)**—often referred to as Sequence Parallelism across nodes, or realized via primitives like **RingAttention** and **DeepSpeed-Ulysses**—is an intra-layer distributed training and inference paradigm that partitions the **sequence (context) dimension** $S$ across $P$ accelerators while leaving the model’s weight matrices intact on each rank.

Where Tensor Parallelism (TP) splits the hidden dimension $h$ (or attention heads $h_{\text{heads}}$) and hits an intra-node barrier ($P \le 8$), Context Parallelism enables scaling sequence lengths to hundreds of thousands or millions of tokens ($S \ge 1\text{M}$) across multi-node clusters by distributing activation memory and compute quadratically tied to sequence length.

### 1\. The Scaling Bottleneck CP Resolves

In standard Transformer architectures, self-attention memory and compute scale with sequence length $S$:

- **Compute (FLOPs):** $\mathcal{O}(S^2 \cdot d)$ per head.
- **Activation Memory:** Attention logits require $\mathcal{O}(B \cdot h \cdot S^2)$ in vanilla attention, or $\mathcal{O}(B \cdot h \cdot S \cdot d)$ under online Softmax engines like FlashAttention.
- **KV Cache (Inference):** Scales linearly with $S$ per request: $2 \times 2 \times B \times S \times L \times h_{KV} \times d_k$ bytes.

When scaling $S \to 128\text{k}$ or $1\text{M}$, a single GPU's High-Bandwidth Memory (HBM) is overwhelmed by activations even if the model weights fit completely. TP cannot shard $S$ inside the core attention computation without cross-head communication, and Pipeline Parallelism (PP) requires activations of the entire sequence to be buffered between stage handoffs. Context Parallelism directly addresses this by slicing inputs:

$$
X \in \mathbb{R}^{B \times S \times d} \implies X^{(p)} \in \mathbb{R}^{B \times \frac{S}{P} \times d}, \quad p \in \{0, 1, \dots, P-1\}
$$

Each rank holds a sub-sequence of length $s = S/P$.

### 2\. The Core Challenge: Cross-Rank Attention

Feedforward networks (MLPs), RMSNorm/LayerNorm, and embedding lookups operate **element-wise** or across token features independently. Sharding along $S$ means these layers run locally on rank $p$ with **zero cross-rank communication**:

$$
H_{\text{MLP}}^{(p)} = \text{MLP}\left(X^{(p)}\right)
$$

The fundamental challenge arises strictly at **Self-Attention**: token $i$ on rank $p$ must attend to all tokens $j \le i$ (under causal masking) or all tokens $j \in [0, S-1]$ (under bidirectional attention), many of which reside in the memory of ranks $q \ne p$.

Two primary architectures have emerged to solve this: **RingAttention** and **DeepSpeed-Ulysses**.

### 3\. Paradigm A: RingAttention (P2P Overlapped Rings)

Proposed by Hao Liu et al., **RingAttention** distributes the query, key, and value sequences across a logical ring of $P$ GPUs.

#### Sharding

Each GPU $p$ permanently holds its local chunk of Queries, Keys, and Values:

$$
Q^{(p)}, K^{(p)}, V^{(p)} \in \mathbb{R}^{B \times \frac{S}{P} \times h \times d_k}
$$

#### Execution Loop

Instead of collecting all Keys and Values globally via an `All-Gather` (which would trigger an out-of-memory exception), RingAttention streams $K$ and $V$ in a bucket-brigade ring across $P$ steps:

```
Step 0: GPU p computes Attention(Q^(p), K^(p), V^(p))
  │
  ├── P2P Send K^(p), V^(p) to GPU (p + 1)
  └── P2P Recv K^(p-1), V^(p-1) from GPU (p - 1)
  │
Step 1: GPU p computes Attention(Q^(p), K^(p-1), V^(p-1)) [Overlapped with step 2 transfer]
  │
  └── ... repeat for P - 1 shifts ...
```
```
┌───────────────────────────────┐
       ▼                               │
   [GPU 0] ──► (K, V) ──► [GPU 1] ──► [GPU 2] ──► [GPU 3] ──┐
      ▲                                                      │
      └──────────────────────────────────────────────────────┘
```

#### Online Softmax Rescaling

Because $Q^{(p)}$ attends to blocks of $(K, V)$ arriving asynchronously over $P$ ring steps, the standard Softmax normalization factor $\sum e^{q \cdot k}$ cannot be computed upfront. RingAttention integrates **Online Softmax** (the core mathematical engine of FlashAttention):

For each block step $k \in [0, P-1]$, rank $p$ maintains running statistics:

- Local maximum logit: $m_i^{(k)} \in \mathbb{R}$
- Running denominator sum: $\ell_i^{(k)} \in \mathbb{R}$
- Partial output accumulator: $O_i^{(k)} \in \mathbb{R}^{d_k}$

When a new incoming block $K_{\text{new}}, V_{\text{new}}$ is processed:

1. Compute local raw scores: $S_{\text{new}} = Q^{(p)} K_{\text{new}}^T / \sqrt{d_k}$
2. Update local max: $m_i^{\text{next}} = \max(m_i^{(k)}, \, \max(S_{\text{new}}))$
3. Rescale running sum and output accumulator:
	$$
	\ell_i^{\text{next}} = \ell_i^{(k)} \cdot e^{m_i^{(k)} - m_i^{\text{next}}} + \sum e^{S_{\text{new}} - m_i^{\text{next}}}
	$$
	$$
	O_i^{\text{next}} = O_i^{(k)} \cdot \left(\frac{\ell_i^{(k)} e^{m_i^{(k)} - m_i^{\text{next}}}}{\ell_i^{\text{next}}}\right) + \text{Softmax}\left(S_{\text{new}} - m_i^{\text{next}}\right) V_{\text{new}} \cdot \left(\frac{e^{m_i^{\text{next}}}}{\ell_i^{\text{next}}}\right)
	$$

At step $P-1$, $O_i$ holds the exact mathematical result without any single GPU ever holding the complete sequence.

#### Compute-Communication Overlap

RingAttention completely hides network latency if the time to compute local attention on a block of size $\frac{S}{P}$ exceeds the point-to-point (P2P) transfer time of $(K, V)$ over interconnect links:

$$
T_{\text{compute}} = \frac{4 \cdot B \cdot (S/P)^2 \cdot h \cdot d_k}{\text{FLOPs}_{\text{GPU}}} \quad \ge \quad T_{\text{comm}} = \frac{2 \cdot B \cdot (S/P) \cdot h_{KV} \cdot d_k \times \text{sizeof(FP16)}}{\text{Bandwidth}_{\text{link}}}
$$

As long as the per-GPU token chunk $\frac{S}{P}$ is large enough (typically $\ge 2\text{k} - 4\text{k}$ tokens), communication is **100% hidden behind compute**.

### 4\. Causal Masking Imbalance and Striped RingAttention

In causal auto-regressive models, token $i$ can only attend to token $j$ where $j \le i$.

If sequences are sliced contiguously into linear chunks:

- **GPU 0** gets tokens $[0, \frac{S}{P}-1]$: It only computes self-attention on its own tokens; all subsequent ring chunks from GPUs $1 \dots P-1$ are masked out ($0\%$ compute on those steps).
- **GPU $P-1$** gets tokens $[\frac{(P-1)S}{P}, S-1]$: It must attend to all past tokens ($100\%$ compute across all $P$ steps).

This creates an extreme compute imbalance where GPU 0 sits idle while GPU $P-1$ bottlenecks the cluster, yielding ~50% hardware underutilization.

```
Naive Block Assignment:
GPU 0: [■ □ □ □]  -> 25% compute
GPU 1: [■ ■ □ □]  -> 50% compute
GPU 2: [■ ■ ■ □]  -> 75% compute
GPU 3: [■ ■ ■ ■]  -> 100% compute (Bottleneck)
```

#### Striped Partitioning (Round-Robin Slicing)

To restore perfect load balance, **Striped Attention** maps tokens to GPUs using an interleaved round-robin stride:

$$
\text{Token } t \implies \text{GPU } (t \bmod P)
$$

Each GPU holds an evenly distributed mixture of early, middle, and late tokens. The number of active causal attention pairs is virtually identical across all $P$ ranks throughout the ring rotations, returning hardware compute efficiency to near $100\%$.

### 5\. Paradigm B: DeepSpeed-Ulysses (All-to-All Transposition)

While RingAttention uses circular asynchronous P2P, **DeepSpeed-Ulysses** leverages global `All-to-All` collective communication primitives to swap between sequence partitioning and attention-head partitioning.

```
Input: Slices along Sequence [S/P, H]
               │
               ▼
   [ All-to-All-v Collective ]  <-- Transposes (S/P, H) -> (S, H/P)
               │
               ▼
Compute Attention locally on [S, H/P] via FlashAttention (Full Sequence, Subset of Heads)
               │
               ▼
   [ All-to-All-v Collective ]  <-- Transposes (S, H/P) -> (S/P, H)
               │
               ▼
Output: Slices along Sequence [S/P, H]
```
1. **Before Attention:** Activations are partitioned along sequence length $\frac{S}{P}$, holding all heads $H$.
2. **First `All-to-All`:** Ranks exchange slices so each GPU now holds the **entire sequence length $S$**, but only a subset of heads $\frac{H}{P}$.
3. **Local Compute:** Standard optimized FlashAttention runs locally on full context $S$ for $\frac{H}{P}$ heads with zero custom communication or online softmax stitching.
4. **Second `All-to-All`:** The attention output tensor is transposed back from $[S, \frac{H}{P}]$ to $[\frac{S}{P}, H]$ to resume element-wise MLP computation.

#### DeepSpeed-Ulysses vs. RingAttention Trade-offs

| Dimension | RingAttention | DeepSpeed-Ulysses |
| --- | --- | --- |
| **Communication Type** | Asynchronous Non-blocking P2P (`ISend` / `IRecv`) | Synchronous Collective (`All-to-All`) |
| **Head Constraint** | Works with any head configuration (even MQA where $h_{KV}=1$) | Requires total attention heads divisible by $P$ ($H \bmod P = 0$) |
| **Interconnect Sensitivity** | Tolerant of lower bandwidths (e.g., inter-node RoCE/Infiniband) via overlap | Demands ultra-high bisection bandwidth (NVSwitch / high-tier InfiniBand) |
| **Implementation Complexity** | High (custom Online Softmax loop + CUDA/Triton ring management) | Low (wraps native FlashAttention inside two `All-to-All` collectives) |
| **Memory Ceiling** | Independent of $P$; scales arbitrarily to millions of tokens | Max sequence $S$ constrained by whether a single GPU can hold $[S,\tfrac{H}{P}]$ in memory |

### 6\. Integration in 4D Parallelism (TP + PP + DP/FSDP + CP)

In large-scale production training (e.g., Llama 3, DeepSeek-V3), Context Parallelism forms a dedicated dimension in the parallel process grid:

$$
\text{World Size} = \text{TP} \times \text{PP} \times \text{DP (FSDP)} \times \text{CP}
$$

- **Intra-node:** Tensor Parallelism (TP $\le 8$) operates within a single NVLink domain due to sub-microsecond latency demands.
- **Low-tier Inter-node:** Context Parallelism (CP) operates across nearby nodes, hiding inter-node network latency behind the compute of large sequence chunks.
- **Global Grid:** ZeRO-3 / FSDP shards optimizer states across the data-parallel group, while Pipeline Parallelism (PP) connects distant clusters or stages across minimal communication boundaries.

During backpropagation, CP requires reversing the communication ring or re-executing `All-to-All` to distribute activation gradients $\nabla_Q, \nabla_K, \nabla_V$, maintaining an identical memory footprint reduction throughout the backward pass.
