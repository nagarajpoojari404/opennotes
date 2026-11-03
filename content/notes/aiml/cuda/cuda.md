---
title: "CUDA"
date: "2026-10-29"
---

CUDA (Compute Unified Device Architecture) bridges a massively parallel hardware substrate with an explicit, scalable software abstraction. Understanding CUDA at an advanced level requires mapping software concepts directly to physical hardware and silicon execution behavior.

## 1\. Hardware Architecture & Execution Substrate

A modern NVIDIA GPU (such as Hopper or Blackwell) consists of a Scalable Processing Array composed of multiple **GPCs (Graphics Processing Clusters)**, each containing multiple **TPCs (Texture Processing Clusters)**, which house the core compute engines: the **Streaming Multiprocessors (SMs)**.

```
+-------------------------------------------------------+
       |                  GPU Global Memory (HBM3e/GDDR6X)     |
       +-------------------------------------------------------+
                                  |
                   Crossbar / NVLink Network On-Chip
                                  |
       +-------------------------------------------------------+
       |                  Shared L2 Cache (e.g. 50–96 MB)      |
       +-------------------------------------------------------+
                |                                    |
     +---------------------+              +---------------------+
     |    SM 0             |              |    SM N             |
     |  +---------------+  |              |  +---------------+  |
     |  | SMSP 0        |  |              |  | SMSP 0        |  |
     |  | - Warp Sched  |  |              |  | - Warp Sched  |  |
     |  | - Reg File    |  |  ...         |  | - Reg File    |  |
     |  | - FP32 / INT32|  |              |  | - FP32 / INT32|  |
     |  | - Tensor Core |  |              |  | - Tensor Core |  |
     |  +---------------+  |              |  +---------------+  |
     |  [Shared Memory/L1] |              |  [Shared Memory/L1] |
     +---------------------+              +---------------------+
```

### The Anatomy of a Streaming Multiprocessor (SM)

Each SM functions as an independent processing unit partitioned into **4 sub-cores (SMSPs - SM Processing Blocks)**:

- **Warp Schedulers & Dispatch Units:** Each sub-core includes 1 warp scheduler and dispatch logic. In every clock cycle, each scheduler inspects its pool of active warps and issues ready instructions without pipeline stalling.
	Scribd
- **Register File (RF):** A massive physical SRAM structure (typically 64K 32-bit registers, or 256 KB per SM). It provides zero-cycle context switching between active threads because all registers for all resident threads stay on-chip simultaneously.
	Nikhil Kuniyil
- **Execution Pipelines:**
	- **ALUs:** FP32 (Single Precision), FP64 (Double Precision), and INT32 units.
		- **SFUs (Special Function Units):** Approximate transcendental math ($\sin, \cos, \log_2, \exp_2$, reciprocal square root) via hardware interpolation.
		- **Tensor Cores:** Mixed-precision matrix-multiply-accumulate (MMA) systolic engines executing operations like $D = A \times B + C$ on matrix fragments.
		- **LD/ST Units (Load/Store):** Dedicated hardware to manage address calculation, memory coalescing, and cache accesses.
- **Unified L1 Data Cache / Shared Memory:** A configurable, high-bandwidth SRAM block (up to 228–256 KB per SM) split dynamically between hardware-managed L1 cache and software-managed shared memory.
	Nikhil Kuniyil
- **TMA (Tensor Memory Accelerator - Hopper/Blackwell):** An asynchronous hardware engine that transfers multi-dimensional tensor tiles directly between Global Memory and Shared Memory without requiring SM arithmetic cores to execute load/store loops.

## 2\. The Programming Model vs. Hardware Mapping

CUDA maps a hierarchical software abstraction onto physical hardware elements:

| Software Abstraction | Execution Granularity | Hardware Mapping | Hardware Scope |
| --- | --- | --- | --- |
| **Thread** | Scalar Instruction stream | CUDA Core / ALU Lane | Single pipeline lane |
| **Warp** | 32 Threads (Lockstep SIMT) | Warp Scheduler & Vector Unit | Sub-core (SMSP) |
| **Thread Block (CTA)** | Cooperative Thread Array (up to 1024 threads) | Streaming Multiprocessor (SM) | 1 SM (non-migrating) |
| **Grid** | Entire kernel invocation | Entire GPU Device | All SMs |
| **Cluster** (Compute 9.0+) | Group of cooperative CTAs | Distributed SMs (Cluster) | Local GPC interconnect |

### SIMT Execution and Warp Divergence

CUDA uses **SIMT (Single Instruction, Multiple Threads)**:

Scribd

- 32 threads form a single **warp**. Instructions are broadcast to all 32 lanes simultaneously.
	Scribd\+ 1
- **Warp Divergence:** If threads in a warp branch down different paths via conditional branches (`if/else`), the warp must serialize execution: it executes the `true` path while masking inactive lanes, then executes the `false` path with the complement mask.
	Medium - Medium
- **Independent Thread Scheduling (Volta+):** Prior to Volta, the entire warp shared a single Program Counter (PC). Volta introduced per-thread call stacks and PCs, enabling sub-warp concurrency and interleaving, while warp divergence is tracked via execution masks.

## 3\. The Memory Hierarchy & Coalescing

CUDA’s memory architecture spans latency and bandwidth tiers:

```
Fastest / Smallest       Registers (0-1 cycle latency, ~256 KB/SM)
        |                Shared Memory / L1 Data Cache (tens of cycles, ~100-256 KB/SM)
        |                L2 Cache (tens of nanoseconds, ~40-96 MB chip-wide)
        v                Global Memory / HBM (hundreds of cycles, 16-192 GB)
Slowest / Largest        Host Memory (PCIe / NVLink, High latency)
```

### Memory Spaces

- **Registers:** Thread-private, single-cycle access. If a thread exceeds its register budget, excess variables spill into **Local Memory** (physically backed by off-chip DRAM, though cached in L1/L2).
- **Shared Memory:** Block-wide software-managed scratchpad SRAM. Divided into 32 banks (4 bytes wide per bank).
- **Constant Memory:** 64 KB cached memory optimized for uniform access, where all threads in a warp read the exact same address simultaneously.
- **Global Memory:** High-bandwidth off-chip DRAM (HBM/GDDR) accessible by all thread blocks and grids.

### Coalescing and Bank Conflicts

- **Global Memory Coalescing:** Memory requests from threads in a warp are merged by hardware into cache-line aligned transactions (typically 32-byte or 128-byte segments). If 32 threads access consecutive 4-byte floats ($32 \times 4 = 128$ bytes), the hardware serves the entire warp with a single 128-byte transaction. Non-sequential or strided access causes the transaction to split into multiple serialized DRAM requests.
- **Shared Memory Bank Conflicts:** Shared memory is split into 32 banks (Bank Index = `(address / 4) % 32`). If two or more threads in a warp request addresses mapping to the **same bank** (with different word addresses), access is serialized (an $N$\-way bank conflict). If all threads read the *exact same* word, a broadcast occurs with no conflict.

## 4\. Advanced Primitives & Optimization Techniques

### Warp-Level Primitives (Intra-Warp Communication)

Rather than writing to shared memory and calling `__syncthreads()`, threads within a warp can exchange registers directly in silicon using warp shuffle instructions:

Code snippet

```cpp
// Intra-warp inclusive sum reduction using shuffle down
inline __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // Lane 0 contains the reduction sum of the warp
}
```
- **Active Mask (`0xffffffff`):** Specifies which lanes participate; inactive lanes are ignored.
- Other primitives include `__ballot_sync`, `__any_sync`, and `__all_sync` for evaluating branch predicates across a warp.

### Asynchronous Data Movement & Pipeline Stages (Ampere / Hopper)

Modern CUDA bypasses intermediate register staging for global-to-shared copies using asynchronous copy instructions:

Code snippet

```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

// Asynchronous copy directly from Global to Shared Memory (Ampere+ cp.async)
__global__ void tiled_async_copy(float* d_in, float* d_out, int N) {
    extern __shared__ float s_tile[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Direct Global -> Shared transfer without touching thread registers
    cuda::memcpy_async(&s_tile[tid], &d_in[idx], sizeof(float), cuda::aligned_size_t{});

    // Asynchronous barrier sync
    cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;
    // Overlap compute on previous tile while waiting for current tile transfer...
}
```

### Cooperative Groups and Clusters

CUDA Cooperative Groups replace brittle raw barriers with composable, flexible thread synchronization:

Code snippet

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperative_kernel() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Synchronize only the warp, not the whole block
    warp.sync();

    // Cluster-level access (Hopper Architecture: Distributed Shared Memory / DSMEM)
    // Allows thread blocks within a cluster to read/write each other's shared memory
}
```

## 5\. Architectural Trade-offs: Occupancy vs. Latency Hiding

GPUs hide memory latency through hardware **concurrency**, not speculative out-of-order execution:

1. **Theoretical Occupancy:** The ratio of active warps per SM to the maximum theoretical warps supported by the SM.
	Medium - Medium
2. **Resource Constraints:** Occupancy is constrained by:
	- Registers per thread (e.g., using 64 registers/thread limits active threads per SM).
		- Shared memory allocated per thread block.
		- Total thread blocks per SM.
3. **The Occupancy Fallacy:** Higher occupancy does not guarantee faster code. Once an SM has enough active warps to completely cover arithmetic and memory pipeline latency (typically 4–8 warps per scheduler), further increasing occupancy reduces the register budget per thread, triggering **register spilling** to local memory and lowering instruction-level parallelism (ILP).
