---
title: "Training mem cost"
date: "2026-11-01"
---

During LLM pretraining, memory overhead is divided into **static state memory** (model weights, gradients, optimizer states), **dynamic activation memory**, and **runtime buffers/fragmentation**.

### 1. Static Memory Breakdown (Per-Parameter)

For standard **16-bit mixed-precision training** (BF16 / FP16 forward and backward passes) using the standard **AdamW** optimizer:

|**Component**|**Precision**|**Bytes / Parameter**|**Formula**|
|---|---|---|---|
|**Model Parameters ($\Phi$)**|BF16 / FP16|$2\text{ bytes}$|$2N$|
|**Gradients ($g$)**|BF16 / FP16|$2\text{ bytes}$|$2N$|
|**Master Weights**|FP32|$4\text{ bytes}$|$4N$|
|**Optimizer Momentum ($m$)**|FP32|$4\text{ bytes}$|$4N$|
|**Optimizer Variance ($v$)**|FP32|$4\text{ bytes}$|$4N$|
|**Total Static Footprint**|Mixed FP16/FP32|**$16\text{ bytes / param}$**|**$16N$**|

_(Note: If using full FP32 gradients, add $2N$ $\rightarrow$ $18N$; if using 8-bit AdamW, optimizer states drop from $12N \rightarrow 6N$, reducing total static footprint to $10N$.)_

### 2. Distributed Sharding Scaling (ZeRO / FSDP)

When distributing across $K$ accelerator devices with Data Parallelism ($DP = K$):

|**Strategy**|**Memory per Device** ($M_{\text{static}}$)|**For 7B on 8 devices** ($N=7\times10^9$, $K=8$)|
|---|---|---|
|**No Sharding (Standard DDP)**|$2N + 2N + 12N = \mathbf{16N}$|$\approx 112\text{ GB}$ (OOM on 80GB GPU)|
|**ZeRO-1 / FSDP Stage 1** (Shard Optimizer)|$2N + 2N + \frac{12N}{K} = \mathbf{4N + \frac{12N}{K}}$|$\approx 28 + 10.5 = \mathbf{38.5\text{ GB}}$|
|**ZeRO-2 / FSDP Stage 2** (Shard Opt + Grads)|$2N + \frac{14N}{K} = \mathbf{2N + \frac{14N}{K}}$|$\approx 14 + 12.25 = \mathbf{26.25\text{ GB}}$|
|**ZeRO-3 / FSDP Full** (Shard All)|$\frac{16N}{K}$|$\approx \mathbf{14.0\text{ GB}}$|

### 3. Dynamic Activation Memory

Activation memory scales with batch size per GPU ($b$), sequence length ($s$), number of transformer layers ($L$), and hidden dimension ($h$).

- **Without Activation Checkpointing**:
    
    $$M_{\text{act}} \approx b \cdot s \cdot L \cdot h \cdot \left(34 + 5 \cdot \frac{a \cdot s}{h}\right) \text{ bytes}$$
    
- **With Selective Activation Recomputation (FlashAttention-2 / 3)**: Retains only intermediate layer inputs and recomputes attention maps on the fly:
    
    $$M_{\text{act}} \approx 2 \cdot b \cdot s \cdot L \cdot h \text{ bytes}$$
    
    _Example_: For $b=2$, $s=4096$, $L=32$, $h=4096$ (typical 7B config):
    
    $$M_{\text{act}} \approx 2 \cdot (2) \cdot (4096) \cdot (32) \cdot (4096) \cdot 2 \approx \mathbf{4.29\text{ GB}}$$
    

### 4. Hardware Compute Budget (Chinchilla $D = 20N$)

Under the compute-optimal Chinchilla scaling law ($D = 20N$ tokens):

- **Total Training FLOPs**:
    
    $$C \approx 6 \cdot N \cdot D = 6 \cdot N \cdot (20N) = \mathbf{120N^2\text{ FLOPs}}$$
    
- **Effective Accelerator Throughput**:
    
    $$\text{Wall-Clock Time} = \frac{C}{\text{Num GPUs} \times \text{Peak TFLOPs} \times \text{MFU}}$$
    
    _(MFU = Model FLOPs Utilization, typically $0.40 - 0.50$ for optimized distributed training)._
    

### 5. Memory & Training Demands Across Model Scales & Hardware

The table below calculates the memory overhead, minimum GPU count (for fitting static state + activations under ZeRO-3/FSDP), and training duration using hardware specs (A100 80GB SXM, H100 80GB SXM, H200 141GB SXM, and B200 180GB SXM, assuming MFU $\approx 45\%$):

|**Model Size (N)**|**Chinchilla Dataset (D)**|**Total Static Memory (16N)**|**Min GPUs to Fit (VRAM only)**|**Total FLOPs (120N2)**|**Cluster Config & Estimated Training Time**|
|---|---|---|---|---|---|
|**1B**|20B tokens|16 GB|**1× A100 / H100**|$1.20 \times 10^{20}$|• **8× A100 (80GB)**: ~11 hours • **8× H100 (80GB)**: ~3.5 hours|
|**3B**|60B tokens|48 GB|**1× H200** or **2× A100**|$1.08 \times 10^{21}$|• **16× A100 (80GB)**: ~2.1 days • **8× H100 (80GB)**: ~1.3 days|
|**7B**|140B tokens|112 GB|**2× H100** or **4× A100**|$5.88 \times 10^{21}$|• **64× A100 (80GB)**: ~2.8 days • **32× H100 (80GB)**: ~1.9 days • **16× B200 (180GB)**: ~1.2 days|
|**13B**|260B tokens|208 GB|**4× H100** or **8× A100**|$2.03 \times 10^{22}$|• **128× A100 (80GB)**: ~4.9 days • **64× H100 (80GB)**: ~3.2 days • **32× B200 (180GB)**: ~2.0 days|
|**70B**|1.4T tokens|1.12 TB|**16× H100** or **8× H200/B200**|$5.88 \times 10^{23}$|• **512× H100 (80GB)**: ~11.5 days • **256× H200 (141GB)**: ~23 days • **256× B200 (180GB)**: ~5.8 days|
|**405B**|8.1T tokens|6.48 TB|**64× H200** or **48× B200**|$1.97 \times 10^{25}$|• **4096× H100 (80GB)**: ~48 days • **2048× B200 (180GB)**: ~24 days|

### 6. Hardware VRAM Headroom & Communication Latency Matrix

| **GPU Model** | **VRAM / Bandwidth**                     | **Usable VRAM Budget (after CUDA context & PyTorch allocator reserve)** | **Interconnect Bandwidth** | **Parallelism Bottleneck**                                                                                                       |
| ------------- | ---------------------------------------- | ----------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **A100 SXM4** | 80 GB HBM2e (2.0 TB/s)  | $\approx \mathbf{72\text{ GB}}$                                         | 600 GB/s NVLink            | AllGather during ZeRO-3 starts saturating bandwidth if batch size is too small.                                                  |
| **H100 SXM5** | 80 GB HBM3 (3.35 TB/s)  | $\approx \mathbf{73\text{ GB}}$                                         | 900 GB/s NVLink 4          | Faster FP8 Tensor Cores; 80 GB VRAM becomes the primary bottleneck before compute.                                               |
| **H200 SXM**  | 141 GB HBM3e (4.8 TB/s) | $\approx \mathbf{132\text{ GB}}$                                        | 900 GB/s NVLink 4          | Can host ZeRO-1/2 with larger micro-batch sizes without cross-node activation checkpointing.                                     |
| **B200 SXM**  | 180 GB HBM3e (8.0 TB/s) | $\approx \mathbf{168\text{ GB}}$                                        | 1.8 TB/s NVLink 5          | FP4/FP8 native paths; dual-die packaging doubles compute density, enabling 70B models to fit on single 8-GPU nodes under ZeRO-3. |
|               |                                          |                                                                         |                            |                                                                                                                                  |


Activation memory is the temporary GPU RAM used to store the intermediate tensors produced during the forward pass so that gradients can be computed via the chain rule during the backward pass.

### 1. Why Activations Are Stored

For any linear or matrix operation $Y = X \cdot W$:

- The forward pass computes $Y$.
    
- The backward pass needs to compute the gradient with respect to weights:
    
    $$\frac{\partial \mathcal{L}}{\partial W} = X^T \cdot \frac{\partial \mathcal{L}}{\partial Y}$$
    
    To evaluate that product, the input tensor $X$ **must remain in memory** until the backward pass reaches that layer.
    

### 2. The Analytical Formula (Without Checkpointing)

For a standard transformer layer, activations are stored at each operation: Attention (Q, K, V projections, attention matrix, softmax, output projection), Feed-Forward Network (FFN/MLP), LayerNorms/RMSNorms, and residual additions.

For a model with:

- $b$ = micro-batch size per GPU
    
- $s$ = sequence length (context window)
    
- $h$ = hidden dimension
    
- $a$ = number of attention heads (where head dimension $d = h / a$)
    
- Precision: 16-bit floating point ($2\text{ bytes per element}$)
    

The per-layer breakdown of stored elements:

|**Component**|**Operation**|**Stored Intermediate Tensors**|**Bytes Stored (16-bit = 2×)**|
|---|---|---|---|
|**Self-Attention**|Inputs to $Q, K, V$ projections|$X$ ($b \times s \times h$)|$2 b s h$|
||Attention score matrix ($Q K^T / \sqrt{d}$)|$a \times (b \times s \times s)$|$2 b s^2 a$|
||Softmax output|$a \times (b \times s \times s)$|$2 b s^2 a$|
||Softmax dropout mask (if used)|$a \times (b \times s \times s)$|$1 b s^2 a$ (or $0$ with FlashAttention)|
||Attention $\times V$ matrix product|$b \times s \times h$|$2 b s h$|
||Projection out ($W_o$)|$b \times s \times h$|$2 b s h$|
|**MLP / FFN**|Up-projection (standard 4h, or gated SwiGLU)|$2 \text{ to } 3 \times (b \times s \times h)$|$16 b s h$ to $20 b s h$|
|**Norms & Residuals**|Pre-norm and post-norm inputs|$2 \times (b \times s \times h)$|$4 b s h$|

Summing across all operations in one layer gives:

$$M_{\text{layer}} \approx b \cdot s \cdot h \cdot \left(34 + 5 \cdot \frac{a \cdot s}{h}\right) \text{ bytes}$$

Multiplying by the total number of layers $L$:

$$M_{\text{act}} \approx b \cdot s \cdot L \cdot h \cdot \left(34 + 5 \cdot \frac{a \cdot s}{h}\right) \text{ bytes}$$

Notice two distinct terms inside the parentheses:

1. **$34$ (Linear scaling with $s$):** Hidden state projections, norms, and FFN expansions ($O(s)$).
    
2. **$5 \cdot \frac{as}{h}$ (Quadratic scaling with $s$):** Storing the raw $s \times s$ attention matrices and softmax maps ($O(s^2)$).
    

### 3. How Checkpointing and FlashAttention Change This

Because storing every intermediate tensor causes memory to grow linearly with layer depth $L$ and quadratically with context length $s^2$, modern training uses two primary optimizations:

#### A. Full Activation Checkpointing (Rematerialization)

- **Mechanism:** Discards all intermediate activations inside each transformer block after the forward pass, retaining **only the single input tensor** entering each layer ($b \times s \times h$).
    
- **During Backward:** Re-executes the forward pass for that block on-the-fly to regenerate the necessary activations.
    
- **Trade-off:** Adds ~30–33% compute overhead (one extra forward pass), but reduces activation memory from $O(L \cdot s^2)$ down to roughly:
    
    $$M_{\text{act}} \approx 2 \cdot b \cdot s \cdot L \cdot h \text{ bytes}$$
    

#### B. FlashAttention (Tiling & Online Softmax)

- **Mechanism:** Fuses the attention calculation into a single GPU SRAM kernel using online softmax. It never writes the large $b \times a \times s \times s$ attention matrix to high-bandwidth memory (HBM).
    
- **Impact:** Eliminates the quadratic $O(s^2)$ memory term entirely, making attention memory scale strictly linear with sequence length $s$.
    

### 4. Concrete Example: 7B Model (LLaMA Architecture)

- Micro-batch size ($b$): $2$
    
- Sequence length ($s$): $4,096$
    
- Layers ($L$): $32$
    
- Hidden dimension ($h$): $4,096$
    
- Heads ($a$): $32$
    

#### Without Checkpointing (Vanilla PyTorch)

$$\frac{a \cdot s}{h} = \frac{32 \cdot 4096}{4096} = 32$$

$$\text{Bracket factor} = 34 + (5 \times 32) = 194$$

$$M_{\text{act}} \approx 2 \cdot 4096 \cdot 32 \cdot 4096 \cdot 194 \text{ bytes} \approx \mathbf{208.3\text{ GB}}$$

_(Exceeds the entire 80 GB VRAM of an A100/H100 from activations alone)._

#### With FlashAttention + Selective Recomputation

Only the input to each layer and minimal non-recomputed buffers are retained:

$$M_{\text{act}} \approx 2 \cdot b \cdot s \cdot L \cdot h \text{ bytes} = 2 \cdot 2 \cdot 4096 \cdot 32 \cdot 4096 \text{ bytes} \approx \mathbf{2.15\text{ GB}}$$

_(Drops from 208 GB to ~2 GB per GPU, leaving ample headroom for weights and optimizer states)._
