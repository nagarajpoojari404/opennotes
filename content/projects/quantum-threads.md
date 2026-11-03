---
title: "Quantum Threads: Hybrid Quantum-Classical Deep Learning"
description: "An end-to-end underwater species identification system combining Parameterized Quantum Circuits (PQC) with classical CNNs and distributed cloud QPUs."
date: "2024-04-01T00:00:00+05:30"
categories: ["quantum-computing", "aiml"]
tags: ["quantum-ml", "qiskit", "pennylane", "pqc", "deep-learning", "resnet"]
summary: "Quantum Threads is a research framework for underwater animal classification combining Parameterized Quantum Circuits (PQC) as quantum kernels with ResNet and Inception architectures, evaluated on real IBM Quantum QPUs."
---

[**View on GitHub**](https://github.com/nagarajpoojari404/Quantum-Threads) · [**Architecture**](#hybrid-quantum-classical-pipeline) · [**Key Innovations**](#what-makes-it-innovative)

**Quantum Threads** is an applied quantum machine learning research and engineering project that benchmarks hybrid quantum-classical neural network architectures for fine-grained underwater animal classification across 23 marine species (>13,000 images in the UAD-2023 dataset).

The project develops hybrid counterparts to premier classical computer vision models—**Hybrid Quantum ResNet-18**, **Hybrid Quantum ResNet-50**, and **Hybrid Quantum Inception-V3**—incorporating Parameterized Quantum Circuits (PQCs) as high-dimensional non-linear quantum kernels.

---

## Hybrid Quantum-Classical Pipeline

```mermaid
flowchart LR
    Input["Underwater Image (224x224x3)"] --> ClassicalCNN["Pre-trained Backbone (ResNet-50 / InceptionV3)"]
    ClassicalCNN --> FeatureVec["Latent Feature Vector (512-d)"]
    FeatureVec --> AngleEncoding["Quantum Angle Encoding"]
    
    subgraph QPU["Cloud QPU (IBM Quantum)"]
        AngleEncoding --> PQC["Parameterized Quantum Circuit"]
        PQC --> Entanglement["CNOT Entangling & Rotation Gates"]
        Entanglement --> Measurement["Pauli-Z Expectation Measurement"]
    end
    
    Measurement --> ClassicalHead["Classification Head (23 Classes)"]
    ClassicalHead --> Loss["Cross-Entropy Loss"]
    Loss -->|"Adjoint Differentiation"| Optimizer["Adam Optimizer"]
    Optimizer -->|"Update Circuit Angles"| PQC
```

---

## What Makes It Innovative?

### 1. Parameterized Quantum Circuits (PQC) as Quantum Kernels
By embedding classical feature embeddings into high-dimensional Hilbert spaces using rotational gates ($R_x, R_y, R_z$) and multi-qubit CNOT entangling layers, the quantum circuit maps non-linearly separable marine feature vectors into linearly separable quantum states.

### 2. Adjoint Differentiation
Rather than relying on parameter-shift rules that require $2N$ quantum circuit executions per step (where $N$ is parameter count), the system leverages reversible state-vector adjoint differentiation to compute analytical quantum gradients with constant memory and execution overhead.

### 3. Distributed Cloud QPU Orchestration
Distributes parallel quantum kernel evaluation batches across multiple IBM Quantum Cloud QPUs and state-vector simulators, mitigating queue latency during large-scale model training.

---

## Research Results & Comparison

| Model Architecture | Quantum Kernel Type | Top-1 Accuracy | Parameter Count | Training Convergence |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 (Classical)** | Standard Dense | 87.4% | 11.2M | 45 epochs |
| **Hybrid Quantum ResNet-18** | 8-Qubit Entangled PQC | **89.8%** | 11.1M | **32 epochs** |
| **ResNet-50 (Classical)** | Standard Dense | 91.2% | 23.5M | 50 epochs |
| **Hybrid Quantum ResNet-50** | 16-Qubit Multi-Layer PQC | **93.6%** | 23.4M | **38 epochs** |
| **Hybrid Inception-V3** | 16-Qubit PQC | **92.9%** | 21.8M | **36 epochs** |

---

## Running on Real IBM Quantum Hardware

```bash
# Clone the repository
git clone https://github.com/nagarajpoojari404/Quantum-Threads.git
cd Quantum-Threads

# Install dependencies (Qiskit, PennyLane, PyTorch)
pip install -r requirements.txt

# Configure IBM Quantum API credentials in config.toml
export IBM_QUANTUM_TOKEN="your_ibm_quantum_api_token"

# Train hybrid model
python train_hybrid.py --backbone resnet50 --qubits 8 --shots 1024
```
