---
title: "Torch"
date: "2026-10-28"
---

## 1\. Tensors & Tensor Operations

Tensors are $n$\-dimensional arrays equipped with hardware acceleration and automatic differentiation engines.

### Memory Layout & Strides

A PyTorch tensor is a strided view over a contiguous block of memory (`torch.Storage`). The `.stride()` method tells how many elements to skip in physical memory to step one index along a given dimension.


```python
import torch

# Tensor creation
x = torch.empty(3, 4)                # Uninitialized memory
zeros = torch.zeros(2, 3)            # Zeroes
rand = torch.randn(2, 3)             # Standard normal distribution N(0, 1)
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

# Hardware placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# Shape operations: view vs reshape vs contiguous
# .view() requires contiguous storage; .reshape() copies data if non-contiguous
a = torch.arange(6)                  # shape: (6,)
b = a.view(2, 3)                     # shape: (2, 3), shares memory with a
c = b.t()                            # transpose: non-contiguous in memory
# c.view(6) -> RuntimeError: view size is not compatible with input tensor's size and stride
c_flat = c.contiguous().view(-1)     # copies memory to make it contiguous, then views
```

## 2\. Autograd Engine

PyTorch evaluates gradients using a dynamic Directed Acyclic Graph (DAG) built during the forward pass. Leaf nodes are inputs/weights; interior nodes are operations (`grad_fn`).


```python
# requires_grad tracks computation graph
w = torch.tensor([2.0, 3.0], requires_grad=True)
x = torch.tensor([4.0, 5.0])

# Forward pass: y = w^T * x + b
y = torch.dot(w, x) + 10.0  # y = (2*4) + (3*5) + 10 = 33

# Backward pass (Reverse-mode Automatic Differentiation)
y.backward()

# Gradients accumulated in .grad
print(w.grad)  # dy/dw = x = [4.0, 5.0]

# Disable gradient tracking when inferencing or updating parameters manually
with torch.no_grad():
    w -= 0.01 * w.grad
    w.grad.zero_()  # Crucial: PyTorch buffers accumulate gradients by default
```

### Writing a Custom Autograd Function

When you need custom forward/backward logic (e.g., custom CUDA kernel or non-differentiable approximation):


```python
class ScaledReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale=2.0):
        # Save tensors or non-tensor attributes for backward
        ctx.save_for_backward(input_tensor)
        ctx.scale = scale
        return torch.clamp(input_tensor, min=0) * scale

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output.clone()
        grad_input[input_tensor <= 0] = 0
        return grad_input * scale, None  # scale doesn't require grad

custom_relu = ScaledReLU.apply
```

## 3\. Modular Architecture: `torch.nn`

All network architectures inherit from `nn.Module`. Every layer manages its own registered parameters and forward definition.


```python
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_classes: int):
        super().__init__()
        # Layers registered as submodules
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define forward execution path
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

model = MultiLayerPerceptron(in_features=128, hidden_dim=64, out_classes=10)
```

## 4\. Data Pipelines: `Dataset` & `DataLoader`

PyTorch decouples data access (`Dataset`) from multiprocessing and batch sampling (`DataLoader`).


```python
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, size: int = 1000, features: int = 128):
        self.x = torch.randn(size, features)
        self.y = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

dataset = SyntheticDataset()
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # Multiprocess worker processes
    pin_memory=True       # Pins page-locked CPU memory for faster GPU copies
)
```

## 5\. Idiomatic Training Loop

A complete end-to-end training cycle with model evaluation and checkpointing:


```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLayerPerceptron(128, 64, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()  # Activates Dropout & BatchNorm training updates
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_() due to skipping memset
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Gradient clipping prevents gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # Freezes Dropout & uses running stats for BatchNorm
    total_loss = 0.0
    correct = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        total_loss += loss.item()
        
        preds = predictions.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy
```

## 6\. Advanced PyTorch Engineering

### Automatic Mixed Precision (AMP)

Executes computations in FP16/BF16 where safe (convolutions, matrix multiplications) while retaining FP32 for numerically sensitive ops (reductions, softmax):

```python
scaler = torch.amp.GradScaler('cuda')

for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(batch_x)
        loss = criterion(out, batch_y)

    # Scales loss to prevent FP16 underflow, unscales on backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Graph Compilation (`torch.compile`)

Using TorchDynamo and TorchInductor, `torch.compile` intercepts Python bytecode and fuses operations into optimized Triton kernels:

```python
# Modes: "default", "reduce-overhead" (CUDA graphs), "max-autotune"
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### Custom Hooks

Inspect intermediate activations and modify gradients during runtime:

```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register forward hook on fc1
handle = model.fc1.register_forward_hook(get_activation("fc1"))

# Run forward pass
_ = model(torch.randn(1, 128).to(device))
print(activations["fc1"].shape)

# Remove hook when done
handle.remove()
```

### Distributed Data Parallel (DDP)

DDP spawns a process per GPU with synchronized ring-allreduce gradients:


```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Wrapping model inside worker process:
# model = DDP(model.to(rank), device_ids=[rank])
```
