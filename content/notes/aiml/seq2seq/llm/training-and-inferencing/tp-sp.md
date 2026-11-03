---
title: "TP + SP"
date: "2026-11-03"
---

**Tensor Parallelism (TP)** is an intra-layer model-parallelism technique that splits individual weight matrices, matrix multiplications (GEMMs), and activations across multiple accelerators (GPUs/TPUs).

Unlike Pipeline Parallelism (PP)—which partitions models across sequential layers and introduces execution bubbles—or Data Parallelism (DP/FSDP)—which replicates model weights or shards parameters across batch dimensions—Tensor Parallelism splits individual weight tensors within a layer to compute a single layer cooperatively across multiple ranks.

### 1\. Mathematical Foundations: 1D Matrix Partitioning

Consider a standard fully connected layer computing $Y = \text{Activation}(XW)$, where:

- Input activation $X \in \mathbb{R}^{B \times d_{in}}$ (batch size $B \times$ sequence length $S$, flattened to dimension $B$)
- Weight matrix $W \in \mathbb{R}^{d_{in} \times d_{out}}$
- Output activation $Y \in \mathbb{R}^{B \times d_{out}}$

Across $N$ parallel processing ranks, a matrix multiplication can be sliced along its columns or its rows.

#### Column-Parallel Linear Layer

The weight matrix $W$ is sliced along its column dimension into $N$ equal chunks:

$$
W = \begin{bmatrix} W_1 & W_2 & \dots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{d_{in} \times \frac{d_{out}}{N}}
$$

Each GPU $i$ holds the full input $X$ and its local shard $W_i$:

$$
Y_i = X W_i \in \mathbb{R}^{B \times \frac{d_{out}}{N}}
$$

Because linear matrix multiplication is distributive across columns:

$$
Y = \begin{bmatrix} Y_1 & Y_2 & \dots & Y_N \end{bmatrix}
$$

If an element-wise non-linearity $\sigma$ (e.g., GeLU, SwiGLU) is applied, it commutes with column concatenation:

$$
\sigma(Y) = \begin{bmatrix} \sigma(Y_1) & \sigma(Y_2) & \dots & \sigma(Y_N) \end{bmatrix}
$$

- **Forward pass:** Each rank computes $Y_i$ locally with **zero communication**.
- **Communication required:** None yet, as long as the subsequent operation can consume column-sharded tensors.

#### Row-Parallel Linear Layer

The weight matrix $W$ is sliced along its row (contracting) dimension:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{d_{in}}{N} \times d_{out}}
$$

To match dimensions, the input $X$ must be partitioned along its column dimension: $X = [X_1, X_2, \dots, X_N]$, where $X_i \in \mathbb{R}^{B \times \frac{d_{in}}{N}}$.

Each rank computes a local partial product:

$$
Z_i = X_i W_i \in \mathbb{R}^{B \times d_{out}}
$$

The full mathematical output $Y$ is the summation of these partial products:

$$
Y = \sum_{i=1}^N Z_i = \sum_{i=1}^N X_i W_i = XW
$$

- **Forward pass:** Requires an **All-Reduce (sum)** collective across all $N$ ranks to aggregate $Z_i$ into $Y$.

### 2\. The Megatron-LM Coupling Principle

Naively chaining parallel layers requires an `All-Gather` after each column-parallel layer and an `All-Reduce` after each row-parallel layer, creating substantial collective communication overhead.

The core insight of Megatron-LM is pairing **Column-Parallel** and **Row-Parallel** layers back-to-back to cancel intermediate communications.

```
MLP Forward Pass:
X (Replicated) 
  │
  ├──► GPU 0: [W1_col] ──► GeLU ──► [W2_row] ──┐
  │                                            │ (All-Reduce Sum)
  └──► GPU 1: [W2_col] ──► GeLU ──► [W2_row] ──┴──► Output (Replicated)
```

#### A. Multi-Layer Perceptron (MLP / FFN) Block

A standard Transformer MLP consists of two projections: $Z = \sigma(X W_{gate}) \odot (X W_{up})$ followed by an output projection $W_{down}$.

1. **First Layer (Column-Parallel):**
	- $W_1 \in \mathbb{R}^{h \times 4h}$ is sliced column-wise into $[W_{1,1}, \dots, W_{1,N}]$, each of size $h \times \frac{4h}{N}$.
		- Each rank computes $H_i = \sigma(X W_{1,i})$.
		- *Communication:* **0 bytes.**
2. **Second Layer (Row-Parallel):**
	- $W_2 \in \mathbb{R}^{4h \times h}$ is sliced row-wise into $[W_{2,1}; \dots; W_{2,N}]$, each of size $\frac{4h}{N} \times h$.
		- Each rank multiplies its local intermediate activation $H_i$ with $W_{2,i}$:
		$$
		\widetilde{Y}_i = H_i W_{2,i}
		$$
		- An **All-Reduce** sum collective is executed across the $N$ ranks:
		$$
		Y = \sum_{i=1}^N \widetilde{Y}_i
		$$
		- *Communication:* Exactly **1 All-Reduce** per MLP block.

#### B. Multi-Head Attention (MHA) Block

Attention naturally parallelizes across attention heads ($h$ heads, each with head dimension $d_k$):

1. **$Q, K, V$ Projections (Column-Parallel):**
	- The projection weights $W_Q, W_K, W_V \in \mathbb{R}^{h \times h}$ are sliced along columns such that each GPU receives $\frac{h_{\text{heads}}}{N}$ complete heads.
		- *No communication* is required to split attention heads; each rank gets its local $Q_i, K_i, V_i$.
2. **Local Attention Computation:**
	- Each rank performs scaled dot-product attention independently on its local head partition:
		$$
		\text{Attention}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
		$$
		- *Communication:* **0 bytes.**
3. **Output Projection (Row-Parallel):**
	- The output projection matrix $W_O \in \mathbb{R}^{h \times h}$ is sliced row-wise: $[W_{O,1}; \dots; W_{O,N}]$.
		- Each rank computes $\widetilde{O}_i = \text{Attention}_i \cdot W_{O,i}$.
		- An **All-Reduce** is executed to produce the unified context tensor:
		$$
		O = \sum_{i=1}^N \widetilde{O}_i
		$$
		- *Communication:* Exactly **1 All-Reduce** per Attention block.

### 3\. Forward vs. Backward Conjugate Collectives

Under reverse-mode automatic differentiation, the backward pass of a linear transformation swaps the transpose of the forward communication primitive:

| Layer Type | Forward Communication | Backward Communication (Gradient $\nabla X$) | Backward Parameter Gradient ($\nabla W$) |
| --- | --- | --- | --- |
| **Column Parallel** | None (Identity / Broadcast input) | **All-Reduce** ($\nabla X=\sum_i\nabla Y_iW_i^\top$) | Local: $X^\top\nabla Y_i$ |
| **Row Parallel** | **All-Reduce** ($Y=\sum_i X_iW_i$) | None (Identity / Sliced $\nabla Y$) | Local: $X_i^\top\nabla Y$ |

Each Transformer layer requires:

- **Forward Pass:** 2 All-Reduces (1 for Attention output, 1 for MLP output).
- **Backward Pass:** 2 All-Reduces (1 across the input of Row-Parallel Attention, 1 across the input of Row-Parallel MLP).

### 4\. Communication Overhead and Topology Constraints

A standard Ring All-Reduce transfers $2 \cdot \frac{N-1}{N} \cdot S$ bytes for a tensor of size $S$.

For a Transformer layer with sequence length $L$, batch size $B$, and hidden dimension $h$:

- The tensor transmitted in each All-Reduce is activation $Y \in \mathbb{R}^{B \times L \times h}$.
- In 16-bit precision (FP16/BF16), total bytes per All-Reduce:
	$$
	\text{Bytes} = 2 \times B \times L \times h \times 2 = 4BLh
	$$
- Total forward communication per layer:
	$$
	\text{Volume}_{\text{FWD}} = 2 \times 2 \left(\frac{N-1}{N}\right) (2BLh) = 8BLh \left(\frac{N-1}{N}\right) \text{ bytes}
	$$

Because this communication happens on **every single Transformer layer** in the critical path (blocking further GEMM computation), Tensor Parallelism has strict latency requirements:

- **Bandwidth bottleneck:** Inter-node networking (e.g., 400 Gbps / 50 GB/s HDR/NDR InfiniBand) introduces high latency stalls relative to kernel execution times.
- **Hardware constraint:** Tensor Parallelism is almost exclusively restricted to **intra-node domains** linked via high-speed interconnects (NVLink / NVSwitch at 900 GB/s to 1.8 TB/s per GPU). TP degree is typically set to $N \le 8$ (matching single-node GPU density).

### 5\. Evolution: Sequence Parallelism (Megatron-SP)

In vanilla 1D TP, the operations outside the Attention and MLP blocks—specifically **LayerNorm** and **Dropout**—are replicated identically across all $N$ ranks:

```
Vanilla TP:
Input ──► [LayerNorm (Replicated)] ──► [Col-GEMM] ──► ... ──► [Row-GEMM] ──► [All-Reduce] ──► [Dropout]
```

This causes two problems:

1. Redundant compute for LayerNorm/Dropout.
2. Activation memory for LayerNorm inputs/outputs cannot be sharded, dominating memory consumption at long contexts ($L \ge 8\text{k}$).

**Sequence Parallelism (SP)** solves this by observing that the All-Reduce operation in Row-Parallel linear layers is decomposed into:

$$
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
$$

Instead of running an `All-Reduce` at the end of the row-parallel projection, Megatron-SP runs a **Reduce-Scatter**:

1. The partial sums are reduced and simultaneously scattered along the **sequence dimension** ($L/N$).
2. LayerNorm and Dropout execute on local slices of sequence length $\frac{L}{N}$ with **zero communication**.
3. Before the next Column-Parallel projection, an **All-Gather** reassembles the full sequence across all ranks.
```
Megatron-SP:
[Row-GEMM] ──► [Reduce-Scatter along S] ──► [LayerNorm on S/N] ──► [Dropout on S/N] ──► [All-Gather along S] ──► [Col-GEMM]
```

Communication volume remains identical to vanilla 1D TP (since $\text{Cost}(\text{Reduce-Scatter}) + \text{Cost}(\text{All-Gather}) = \text{Cost}(\text{All-Reduce})$), but activation memory from LayerNorm and Dropout drops by a factor of $N$.
