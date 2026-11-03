---
title: "API"
date: "2026-10-28"
---

The PyTorch core reference API is divided into functional domains spanning tensor manipulation, autograd, neural network modules, runtime optimization, compilation, and distributed execution.

## 1\. Core Tensor Engine (`torch`)

The root `torch` namespace governs tensor instantiation, memory layout, indexing, and mathematical transformations.

| Category | Primary APIs | Mechanics & Details |
| --- | --- | --- |
| **Creation Ops** | `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye` | Allocates uninitialized memory (`empty`) or initialized buffers directly on host or accelerator. |
| **From Existing Data** | `tensor`, `as_tensor`, `from_numpy` | `torch.tensor` performs a defensive copy; `as_tensor` and `from_numpy` share the underlying data pointer if dtype and device match. |
| **Shape & Topology** | `view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`, `contiguous` | `view` alters strides over contiguous storage; `reshape` falls back to a memory copy if strides are non-contiguous; `permute` and `transpose` adjust dimension strides without moving bytes. |
| **Slicing & Joining** | `cat`, `stack`, `split`, `chunk`, `gather`, `scatter_`, `index_select` | `cat` concatenates along an existing dimension; `stack` creates a new dimension; `gather`/`scatter_` perform indexed reads/writes using index tensors. |
| **Math & Linear Algebra** | `matmul`, `bmm`, `einsum`, `addmm`, `linalg.*` (`inv`, `svd`, `qr`, `norm`) | High-performance BLAS/LAPACK backends; `einsum` processes Einstein summation across arbitrary tensor dimensions. |
| **Reductions & Stats** | `sum`, `mean`, `prod`, `std`, `var`, `min`, `max`, `argmax`, `argmin` | Supports multi-dim reductions with `dim` and `keepdim=True` to preserve broadcast compatibility. |

## 2\. Automatic Differentiation (`torch.autograd`)

PyTorch constructs a dynamic Directed Acyclic Graph (DAG) during the forward pass, where leaf nodes are parameters and interior nodes are execution nodes (`Node`/`grad_fn`).

- **Core Execution Primitives:**
	- `torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None)`: Computes sum of gradients of given tensors with respect to graph leaves.
		- `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)`: Computes and returns the sum of gradients of outputs with respect to specific inputs. Setting `create_graph=True` enables higher-order derivatives (e.g., Hessian computations).
- **Context Managers:**
	- `torch.no_grad()`: Disables graph construction, reducing memory consumption during inference.
		- `torch.enable_grad()`: Re-enables gradient computation inside a `no_grad` scope.
		- `torch.inference_mode()`: More aggressive than `no_grad()`; disables both gradient tracking and version counter tracking on tensors.
		- `torch.autograd.set_detect_anomaly(True)`: Traces forward execution to pinpoint operations that generate `NaN` or `Inf` gradients during backward passes.
- **Extensibility:**
	- `torch.autograd.Function`: Base class for writing custom differentiable operations implementing `@staticmethod` `forward(ctx, *args)` and `backward(ctx, *grad_outputs)`. Use `ctx.save_for_backward()` to cache tensors needed for backward calculations.

## 3\. Neural Network Abstractions (`torch.nn` & `torch.nn.functional`)

`torch.nn` provides stateful modules holding parameters and persistent state. `torch.nn.functional` provides stateless mathematical operations.

### Stateful Containers & Layer Classes (`torch.nn`)

- **Base Primitives:** `nn.Module` (tracks child modules, parameters, and buffers), `nn.Parameter` (tensor registered as a trainable weight), `nn.Buffer` (persistent state not updated by optimizers, e.g., running mean/variance).
- **Containers:** `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`.
- **Linear & Recurrent:** `nn.Linear`, `nn.Bilinear`, `nn.RNN`, `nn.LSTM`, `nn.GRU`.
- **Spatial & Volumetric:** `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, `nn.ConvTranspose2d`.
- **Normalization:** `nn.BatchNorm1d/2d`, `nn.LayerNorm`, `nn.GroupNorm`, `nn.RMSNorm`.
- **Attention & Transformers:** `nn.MultiheadAttention`, `nn.TransformerEncoder`, `nn.TransformerDecoder`.
- **Loss Functions:** `nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`, `nn.KLDivLoss`, `nn.CTCLoss`.

### Stateless Operations (`torch.nn.functional` as `F`)

- **Activations:** `F.relu`, `F.gelu`, `F.silu`, `F.softmax`, `F.log_softmax`, `F.sigmoid`, `F.tanh`.
- **Attention Kernels:** `F.scaled_dot_product_attention` (SDPA — selects among FlashAttention, Memory-Efficient Attention, or Math kernels automatically).
- **Sampling & Interpolation:** `F.interpolate`, `F.grid_sample`, `F.pad`.

## 4\. Parameter Optimization (`torch.optim`)

Manages weight updates based on accumulated `.grad` attributes.

- **Algorithms:**
	- First-Order: `SGD` (with momentum and Nesterov), `Adam`, `AdamW` (decoupled weight decay), `RMSprop`, `Adagrad`.
		- Second-Order: `LBFGS`.
- **Optimizer Interfaces:**
	- `optimizer.step()`: Applies gradient updates.
		- `optimizer.zero_grad(set_to_none=True)`: Resets `.grad`. `set_to_none=True` frees memory buffers rather than filling them with zero tensors, improving memory bandwidth.
- **Learning Rate Schedulers (`torch.optim.lr_scheduler`):**
	- Step & Exponential: `StepLR`, `MultiStepLR`, `ExponentialLR`.
		- Dynamic: `ReduceLROnPlateau` (adjusts LR based on validation metric).
		- Smooth Cycles: `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `OneCycleLR`.

## 5\. Data Pipelines (`torch.utils.data`)

Decouples storage and item retrieval from batch sampling, worker multiprocessing, and hardware staging.

- **Datasets:**
	- `Dataset`: Map-style dataset implementing `__len__` and `__getitem__(index)`.
		- `IterableDataset`: Stream-oriented dataset implementing `__iter__` for unbounded or disk-streamed datasets.
- **Sampling & Loading:**
	- `DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last)`:
		- `num_workers > 0`: Uses `multiprocessing` workers to prefetch batches into shared memory (`shm`).
		- `pin_memory=True`: Allocates page-locked host memory to enable non-blocking asynchronous host-to-device transfers (`.to(device, non_blocking=True)`).
		- `collate_fn`: Custom callable converting lists of per-sample dicts/tuples into batch tensors.
		- `Sampler`, `BatchSampler`, `DistributedSampler`: Controls sequence of sample indices generated across single or distributed ranks.

## 6\. Hardware Backends & Precision (`torch.cuda`, `torch.amp`)

Provides low-level execution control over GPUs and mixed-precision arithmetic.

- **CUDA Management (`torch.cuda`):**
	- Stream & Events: `torch.cuda.Stream()`, `torch.cuda.Event()` for asynchronous execution, cross-stream synchronization, and timing.
		- Memory Profiling: `torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`, `torch.cuda.empty_cache()`.
- **Automatic Mixed Precision (`torch.amp`):**
	- `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)`: Casts matrix multiplications and convolutions to FP16/BF16 while preserving FP32 for reductions and softmax.
		- `torch.amp.GradScaler('cuda')`: Dynamically scales loss values to avoid numerical underflow during FP16 backward passes (not required for BF16).

## 7\. Distributed Computation (`torch.distributed`)

Enables scaling across multi-GPU and multi-node clusters via NCCL, GLOO, or MPI backends.

- **Primitives & Collective Communications:**
	- Setup: `dist.init_process_group(backend, rank, world_size)`, `dist.destroy_process_group()`.
		- Collectives: `dist.all_reduce`, `dist.reduce`, `dist.all_gather`, `dist.broadcast`, `dist.scatter`, `dist.barrier`.
- **Distributed Paradigms:**
	- `torch.nn.parallel.DistributedDataParallel` (DDP): Multi-process data parallelism; synchronizes gradients across all ranks via ring-allreduce overlapped with the backward pass.
		- `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP): Zero-redundancy optimizer sharding (ZeRO-3 equivalent); shards parameters, gradients, and optimizer states across ranks, gathering them on-demand during forward/backward steps.
		- `torch.distributed.tensor.parallel` (TP): Tensor-parallel layouts across weight matrices for multi-billion parameter architectures.

## 8\. Compilation, Tracing & Serialization

APIs dedicated to production deployment, intermediate representation (IR) export, and JIT/AOT kernel fusion.

```
+-----------------------------+
                  |  PyTorch Python Source Code |
                  +--------------+--------------+
                                 |
              [torch.compile / TorchDynamo / FX]
                                 |
                                 v
                  +-----------------------------+
                  |   AOTAutograd (Joint Graph) |
                  +--------------+--------------+
                                 |
                        [TorchInductor]
                                 |
               +-----------------+-----------------+
               |                                   |
               v                                   v
      +-----------------+                 +-----------------+
      | C++ / OpenMP    |                 | Triton Kernels  |
      | (CPU Targets)   |                 | (CUDA/ROCm GPUs)|
      +-----------------+                 +-----------------+
```
- **Modern Compiler Stack (`torch.compile`):**
	- `TorchDynamo`: JIT compiler using CPython frame evaluation hooks to intercept Python bytecode and extract clean FX computation graphs.
		- `AOTAutograd`: Traces backward graphs ahead-of-time before code generation.
		- `TorchInductor`: Code generation backend compiling computation graphs into C++ (CPU) or Triton kernels (GPU).
- **Legacy JIT & Export:**
	- `torch.jit.trace` / `torch.jit.script`: Converts code to TorchScript IR for execution in pure C++ runtimes (`libtorch`).
		- `torch.export`: Produces hermetic, sound computation graphs (`ExportedProgram`) targeted at mobile runtimes and external hardware compilers.
- **State Serialization:**
	- `torch.save(obj, f)` / `torch.load(f, weights_only=True)`: Serializes object graphs using pickle archives or raw state dictionaries (`state_dict`).

## 9\. Profiling & Execution Inspection (`torch.profiler`)

Monitors device kernels, CPU overhead, memory leaks, and operator runtimes.

- **Context API:**
	- `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)`
- **Analysis & Visualization:**
	- `prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)`: Prints summary tables.
		- `prof.export_chrome_trace("trace.json")`: Exports traces for analysis in Chrome Tracing (`chrome://tracing`) or Perfetto.
