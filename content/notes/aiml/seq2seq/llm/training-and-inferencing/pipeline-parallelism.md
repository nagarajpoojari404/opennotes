---
title: "Pipeline Parallelism"
date: "2026-11-03"
---
### Pipeline Parallelism Fundamentals

Pipeline Parallelism (PP) partitions a deep neural network sequentially across $p$ accelerators (pipeline stages), where stage $k$ processes layer chunk $k$ and sends boundary activations to stage $k+1$.

In a standard backward pass of deep learning workloads, computation splits into two distinct operations:

1. **$B$ (Activation Gradient / $\nabla_{\mathbf{x}}$):** Computes gradients with respect to input activations $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ and passes them upstream to stage $k-1$.
2. **$W$ (Weight Gradient / $\nabla_{\mathbf{w}}$):** Computes gradients with respect to local parameters $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$. This computation does not block preceding pipeline stages.

The fundamental trade-off in pipeline parallelism balances three competing factors:

- **Bubble Rate ($r_{bubble}$):** The fraction of GPU time spent idle waiting for upstream activations or downstream gradients.
- **Peak Activation Memory:** The maximum number of un-retired micro-batch activation buffers stored concurrently across devices before backward passes consume them.
- **Communication Overhead:** The volume of peer-to-peer (P2P) boundary transfers and synchronization barriers across stages.

### 1\. Naive Model Parallelism (Sequential Execution)

The model is partitioned across $p$ GPUs, but the entire batch $B$ runs sequentially through each GPU without micro-batching.

```
GPU 0: [ F ]--------------------------------------------[ B ]
GPU 1: ----- [ F ]--------------------------------[ B ]------
GPU 2: ----------- [ F ]--------------------[ B ]------------
GPU 3: ----------------- [ F ]--------[ B ]------------------
Time  ------------------------------------------------------>
```
- **Execution:** GPU 0 computes the forward pass on the entire batch, transmits activations to GPU 1, and idles. The data travels forward to GPU $p-1$, which computes loss and starts backpropagation back to GPU 0.
- **Bubble Fraction:** At any given moment, only $1$ of $p$ GPUs is active.
	$$
	r_{bubble} = \frac{p - 1}{p}
	$$
	For $p=8$, $87.5\%$ of all accelerator compute cycles are completely idle.

### 2\. GPipe (Synchronous Micro-Batching)

GPipe splits a mini-batch into $m$ independent micro-batches ($m \gg p$). Micro-batches are pushed sequentially through the stages in an accumulated pipeline.

```
GPU 0: [F1][F2][F3][F4]                        [B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4]                [B4][B3][B2][B1]
GPU 2:         [F1][F2][F3][F4]        [B4][B3][B2][B1]
GPU 3:             [F1][F2][F3][F4][B4][B3][B2][B1]
       |<-- Startup Bubble -->|        |<-- Drain Bubble -->|
```
- **Execution Schedule:** All $m$ forward micro-batches run first (filling the pipe), followed by an accumulation barrier, and then all $m$ backward micro-batches run in reverse order (LIFO or FIFO) before an optimizer step.
- **Bubble Ratio:** With forward micro-batch execution time $t_F$ and backward execution time $t_B \approx 2t_F$:
	$$
	r_{bubble} = \frac{(p - 1)(t_F + t_B)}{m(t_F + t_B)} = \frac{p - 1}{m}
	$$
	To bring bubble overhead under $10\%$, $m \ge 9p$ is required.
- **Memory Bottleneck:** GPU 0 must retain intermediate activations for all $m$ micro-batches until micro-batch 1's backward pass begins:
	$$
	\text{Peak Activation Memory} \propto \mathcal{O}(m)
	$$
	When scaling $m$ to reduce bubbles, device memory exhausts rapidly unless activation recomputation (checkpointing) is aggressively applied.

### 3\. 1F1B (One Forward, One Backward)

Introduced in PipeDream and Megatron-LM, **1F1B** caps activation memory by scheduling alternating execution: once a stage's pipeline is warm, it executes exactly one forward pass followed by one backward pass.

```
Warmup: Stage k executes (p - k) forward passes
Steady State: 1 Forward followed by 1 Backward (1F1B)
Cooldown: Finish remaining in-flight backwards

GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4]...
GPU 1:     [F1][F2][F3][B1][F4][B2][F5][B3][F6][B4]...
GPU 2:         [F1][F2][B1][F3][B2][F4][B3][F5][B4]...
GPU 3:             [F1][B1][F2][B2][F3][B3][F4][B4]...
```
- **Memory Bound:** A stage holds at most $p$ active micro-batch activation buffers at any point, completely decoupling memory consumption from the total number of micro-batches $m$:
	$$
	\text{Peak Activation Memory} \propto \mathcal{O}(p)
	$$
- **Bubble Ratio:** Identical to GPipe:
	$$
	r_{bubble} = \frac{p - 1}{m}
	$$
	1F1B allows large $m$ to lower bubble rates without exhausting VRAM.

### 4\. Interleaved 1F1B

Megatron-LM introduced the **Interleaved 1F1B** schedule to attack the bubble directly without scaling micro-batches ($m$).

- **Mechanism:** Each physical GPU holds $v$ non-continuous "virtual stages" (or chunks) instead of 1. For example, with $p=4$ GPUs and $v=2$ virtual stages per GPU:
	- GPU 0 hosts Chunk 0 (layers 1–2) and Chunk 4 (layers 9–10).
		- GPU 1 hosts Chunk 1 (layers 3–4) and Chunk 5 (layers 11–12).
		- GPU 2 hosts Chunk 2 (layers 5–6) and Chunk 6 (layers 13–14).
		- GPU 3 hosts Chunk 3 (layers 7–8) and Chunk 7 (layers 15–16).
- **Bubble Reduction:** Because each chunk contains $\frac{1}{v}$ of a stage's layers, individual compute slots shrink by $v$. The startup and drain bubbles scale down:
	$$
	r_{bubble} = \frac{p - 1}{v \cdot m}
	$$
	Setting $v=2$ halves the bubble overhead for the same $m$.
- **Trade-off:** Network communication increases by a factor of $v$ because activations and gradients must pass through the physical device ring $v$ times per mini-batch.

### 5\. Zero-Bubble Pipeline Parallelism (ZBPP)

Conventional 1F1B treats the backward pass as a single monolithic block ($B$). **Zero-Bubble PP** exploits the decoupling between activation gradients ($B$) and parameter gradients ($W$):

Sea AI Lab - Sea Limited

- $B$ lies strictly on the **critical latency path** (upstream stages cannot backpropagate until they receive $\nabla_{\mathbf{x}}$).
- $W$ lies **off the critical path** (weights are only needed at the end-of-batch optimizer step).
```
Decomposition:
Standard:  [ F ]  ->  [ Backward (B + W) ]
ZBPP:      [ F ]  ->  [ B (activation grad) ] ... [ W (weight grad) scheduled into idle bubbles ]
```
- **Scheduling Strategy (ZB1P / ZBV):**
	1. Split backward into distinct kernel launches: compute $\nabla_{\mathbf{x}}$ ($B$) first, defer $\nabla_{\mathbf{w}}$ ($W$).
		2. Prioritize tasks: $B > F > W$.
		3. Push $W$ tasks into the startup and teardown phases where GPUs normally sit idle waiting for boundary activations.
- **Performance:** With $t_F \approx t_B \approx t_W$, the bubble is reduced from $\frac{p-1}{m}$ down to $\frac{p-1}{3m}$ (in ZB1P) or near **$0\%$ bubble** (in ZBV with two virtual stages), without inflating peak activation memory relative to 1F1B.

### 6\. DualPipe (Bidirectional Overlapping Pipeline)

Introduced in the **DeepSeek-V3** architecture, **DualPipe** is designed for Mixture-of-Experts (MoE) architectures running on scale-out clusters where inter-stage communication and cross-node MoE All-to-All communication create severe latency bottlenecks.

```
Forward Direction 0:   Micro-batches run Stage 0  -> Stage p-1
Reverse Direction 1:   Micro-batches run Stage p-1 -> Stage 0

Physical GPU execution:
Overlaps [Forward / B_grad of Stream 0] with [All-to-All Dispatch / Combine of Stream 1]
```
- **Bidirectional Workload:** The mini-batch is split across two concurrent pipelines flowing in opposite directions:
	- Stream 0 moves forward from GPU $0 \to p-1$.
		- Stream 1 moves forward from GPU $p-1 \to 0$.
- **Fine-Grained Compute-Communication Overlap:** DualPipe splits operations into fine-grained units:
	- $F$ (Forward compute)
		- $B$ (Activation backward compute)
		- $W$ (Weight backward compute)
		- $C_{disp}$ / $C_{comb}$ (MoE dispatch and combine collective communications)
- **Execution:** While a GPU computes $F$ or $B$ for Stream 0, it overlaps the MoE All-to-All network communication for Stream 1 on independent SM streaming kernels and copy engines.
- **Impact:** It eliminates both pipeline bubble latency and network communication waiting time simultaneously, achieving close to $100\%$ GPU compute saturation during MoE training.

### Comparative Architecture Reference

| Variant | Bubble Fraction ($r_{\\text{bubble}}$) | Peak Activation Memory | Communication Volume | Primary Advantage | Main Drawback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive** | $\\frac{p-1}{p}$ | $\\mathcal{O}(1)$ micro-batch | $1\\times$ | Simple implementation | Severe device underutilization |
| **GPipe** | $\\frac{p-1}{m}$ | $\\mathcal{O}(m)$ micro-batches | $1\\times$ | Synchronous, clean batch updates | High memory footprint for small bubbles |
| **1F1B** | $\\frac{p-1}{m}$ | $\\mathcal{O}(p)$ micro-batches | $1\\times$ | Memory capped regardless of $m$ | Bubble still significant when $m \\approx p$ |
| **Interleaved 1F1B** | $\\frac{p-1}{v \\cdot m}$ | $\\mathcal{O}(p)$ micro-batches | $v\\times$ | Reduces bubble by $v$-factor | Increases P2P communication bandwidth |
| **Zero-Bubble (ZBV)** | $\\approx 0$ | $\\mathcal{O}(p)$ micro-batches | $1\\times$ | Near-zero pipeline idle time | Requires splitting $B$ and $W$ kernel graphs |
| **DualPipe** | $\\approx 0$ (bidirectional) | $\\mathcal{O}(p)$ micro-batches | $1\\times$ (pipelined) | Overlaps MoE All-to-All comms with compute | Complex scheduling and bidirectional routing |
