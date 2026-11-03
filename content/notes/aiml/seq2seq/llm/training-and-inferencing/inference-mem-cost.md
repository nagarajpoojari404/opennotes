---
title: "Inference mem cost"
date: "2026-11-01"
---


LLM inference memory and throughput fundamentally split across two operating regimes governed by the **Roofline Model**: the **prefill phase** (time-to-first-token, compute-bound) and the **decode phase** (token generation, memory-bandwidth-bound).

### 1. Total Inference Memory Breakdown

Total inference memory consists of static weights, the dynamic Key-Value (KV) cache, and transient runtime activation scratchpads:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{runtime\_buffers}}$$

#### A. Model Weights ($M_{\text{weights}}$)

Unlike training, inference requires **no optimizer states** and **no gradients**:

$$M_{\text{weights}} = N \times P_{\text{weight}}$$

- **FP16 / BF16**: $2\text{ bytes / param}$ $\rightarrow M_{\text{weights}} = 2N$
    
- **FP8 / INT8**: $1\text{ byte / param}$ $\rightarrow M_{\text{weights}} = 1N$
    
- **INT4 (GPTQ / AWQ / GGUF Q4_K_M)**: $\approx 0.5\text{ to } 0.56\text{ bytes / param}$ $\rightarrow M_{\text{weights}} \approx 0.55N$
    

#### B. KV Cache Memory ($M_{\text{KV}}$)

For autoregressive generation, keys and values computed for previous tokens are stored in VRAM to avoid $O(s^2)$ recomputation:

$$M_{\text{KV}} = 2 \cdot L \cdot h_{\text{KV}} \cdot d_{\text{head}} \cdot s \cdot b \cdot P_{\text{KV}}$$

- $2$: Stores both **Key** and **Value** tensors
    
- $L$: Number of transformer layers
    
- $h_{\text{KV}}$: Number of KV heads (in Grouped-Query Attention / GQA, $h_{\text{KV}} \ll h_{\text{Q}}$, typically $8$ vs $32$)
    
- $d_{\text{head}}$: Head dimension (usually $128$)
    
- $s$: Total sequence context length (prompt tokens + generated tokens)
    
- $b$: Batch size (concurrent requests)
    
- $P_{\text{KV}}$: Bytes per element ($2$ for FP16, $1$ for FP8, $0.5$ for INT4)
    

> **Example (Llama 3.1 8B)**: $L=32$, $h_{\text{KV}}=8$, $d_{\text{head}}=128$, FP16 cache ($2\text{ bytes}$):
> 
> $$\text{KV per token} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072\text{ bytes} = 128\text{ KiB / token}$$
> 
> For a batch of $b=16$ requests at context $s=8,192$:
> 
> $$M_{\text{KV}} = 128\text{ KiB} \times 8,192 \times 16 \approx \mathbf{16.78\text{ GB}}$$

### 2. Autoregressive Decode Speed (Tokens / Sec)

In the token generation (decode) phase with batch size $b=1$, the GPU performs roughly $2N$ FLOPs per token. Because every model parameter must be fetched from HBM into SRAM once per token, the arithmetic intensity is:

$$\text{Arithmetic Intensity} = \frac{2N \text{ FLOPs}}{M_{\text{weights}} \text{ bytes}} \approx \frac{2 \text{ FLOPs}}{2 \text{ bytes}} = 1.0 \text{ FLOP/byte}$$

Modern GPUs have an arithmetic intensity ceiling of $100\text{–}300\text{ FLOP/byte}$, meaning compute units sit idle waiting for memory. **Single-sequence generation speed is strictly memory-bandwidth bound**.

#### Analytical Formula:

$$\text{Throughput (Tokens/sec)} = \frac{\text{Memory Bandwidth (GB/s)} \times \eta_{\text{MBU}}}{\text{Bytes Read Per Token (GB)}}$$

- **$\eta_{\text{MBU}}$ (Memory Bandwidth Utilization)**: Real-world kernel efficiency (typically $60\% \text{ to } 75\%$ in engines like vLLM, TensorRT-LLM, or llama.cpp).
    
- **Bytes read per token**: $M_{\text{weights}} + \frac{M_{\text{KV}}}{b}$ (at moderate contexts, $M_{\text{weights}}$ dominates).
    

When concurrent batch size $b$ increases (continuous batching), the static weights are read once and reused across all $b$ sequences. Generation transitions from **memory-bound** to **compute-bound** when:

$$b^* \approx \frac{\text{Peak Tensor TFLOPs}}{\text{Memory Bandwidth (TB/s)} \times \text{Weight Precision Bytes}}$$

### 3. Hardware Comparison Matrix (Batch Size $b=1$, FP16 vs INT4)

Assuming sustained real-world $\eta_{\text{MBU}} \approx 70\%$:

|**Hardware**|**Memory Type & VRAM**|**Theoretical Bandwidth**|**Llama 8B (FP16: 16 GB)** $\text{Speed}\approx\frac{BW\times 0.7}{16}$|**Llama 8B (INT4: 4.5 GB)** $\text{Speed}\approx\frac{BW\times 0.7}{4.5}$|**Llama 70B (INT4: 38 GB)** $\text{Speed}\approx\frac{BW\times 0.7}{38}$|
|---|---|---|---|---|---|
|**MacBook M3/M4 Max**|128 GB Unified LPDDR5X|$400\text{ GB/s}$|~17.5 tok/s|~62 tok/s|~7.4 tok/s|
|**RTX 3090 / 4090**|24 GB GDDR6X|$1,008\text{ GB/s}$|~44.1 tok/s|~156 tok/s|_(Exceeds 24 GB)_|
|**RTX 5090**|32 GB GDDR7|$1,792\text{ GB/s}$|~78.4 tok/s|~278 tok/s|_(Exceeds 32 GB)_|
|**A100 SXM4**|80 GB HBM2e|$2,039\text{ GB/s}$|~89.2 tok/s|~317 tok/s|~37.5 tok/s|
|**H100 SXM5**|80 GB HBM3|$3,350\text{ GB/s}$|~146.5 tok/s|~521 tok/s|~61.7 tok/s|
|**H200 SXM**|141 GB HBM3e|$4,800\text{ GB/s}$|~210.0 tok/s|~746 tok/s|~88.4 tok/s|
|**B200 SXM**|180 GB HBM3e|$8,000\text{ GB/s}$|~350.0 tok/s|~1,244 tok/s|~147.3 tok/s|

### 4. Batched Serving Throughput (Compute-Bound Scaling)

For cloud inference serving (e.g., serving 64 concurrent users simultaneously on an H100):

- **Single-user generation** is latency-optimized ($\sim 146\text{ tok/s}$, GPU compute is mostly idle).
    
- **Batched generation ($b=64$)** is throughput-optimized:
    
    $$\text{Aggregate Tokens/Sec} \approx \frac{\text{Active FP16 Tensor TFLOPS} \times \text{MFU}}{2 \cdot N} \cdot 10^{12}$$
    
    On an H100 SXM (989 TFLOPS BF16 dense, assuming 40% MFU on 8B model):
    
    $$\text{Aggregate Throughput} \approx \frac{989 \times 10^{12} \times 0.40}{2 \times (8 \times 10^9)} \approx \mathbf{24,725\text{ tokens/sec across all users}}$$
    
    _(Each user receives $\frac{24,725}{64} \approx 386\text{ tokens/sec}$, achieving orders-of-magnitude better compute and hardware cost efficiency)._
