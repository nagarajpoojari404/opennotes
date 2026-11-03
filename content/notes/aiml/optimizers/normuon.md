---
title: "NorMuon"
date: "2026-10-31"
---

**NorMuon** (**Neuron-wise Normalized Muon**) is an optimizer designed to address an imbalance in standard Muon: while Muon's matrix orthogonalization improves the overall condition number of weight updates, individual neuron update norms vary dramatically across rows.

ResearchGate

NorMuon combines Muon’s **polar/spectral orthogonalization** with **neuron-level (row-wise) adaptive scaling** inspired by Adam and Adam-mini.

arXiv

### 1\. The Motivation: Muon’s "Neuron-Imbalance" Problem

In neural network linear layers ($W \in \mathbb{R}^{m \times n}$ where $m$ is the output dimension/neuron count and $n$ is input dimension), each row $W_{i,:}$ represents the incoming connection weights for a single neuron.

ResearchGate

- **Muon's strength:** It projects the momentum matrix $M$ to an approximately orthogonal matrix $O \approx UV^\top$ via Newton–Schulz iterations. This eliminates ill-conditioned singular values, equalizing step sizes along spectral principal directions.
	ResearchGate\+ 1
- **Muon's hidden flaw:** Although the singular values of $O$ are all approximately $1$, the **Euclidean norm of individual rows** $\Vert{}O_{i,:}\Vert{}_2$ (the per-neuron update magnitude) exhibits high variance. Some neurons receive disproportionately aggressive updates while others barely move, causing uneven feature utilization.
	ResearchGate
- **AdamW's behavior:** AdamW applies coordinate-level second-moment scaling ($1 / \sqrt{v + \epsilon}$), which inadvertently regularizes per-neuron norms to be uniform, but fails to fix whole-matrix conditioning.
	ResearchGate\+ 1

NorMuon proves that **matrix-level orthogonalization** and **neuron-level adaptive normalization** are complementary rather than conflicting.

ResearchGate

### 2\. Mathematical Formulation

Let the parameter matrix be $W_t \in \mathbb{R}^{m \times n}$ and the objective gradient be $G_t = \nabla_W \mathcal{L}(W_t)$.

#### Step 1: First-Order Momentum Accumulation

$$
M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t
$$

#### Step 2: Newton–Schulz Orthogonalization (Muon Step)

The momentum matrix is orthogonalized using quintic Newton–Schulz iterations ($\text{NS}_5$) starting from Frobenius-normalized momentum $X_0 = M_t / (\Vert{}M_t\Vert{}_F + \epsilon)$:

ResearchGate

$$
O_t = \text{NS}_5(M_t) \approx U V^\top
$$

#### Step 3: Neuron-wise Second Moment Tracking

ResearchGate

Instead of tracking a full $m \times n$ second-moment tensor like Adam (which is memory-heavy), NorMuon tracks a **1D vector** $v_t \in \mathbb{R}^m$ representing the average squared energy per row of the orthogonalized update $O_t$:

ResearchGate

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \cdot \operatorname{mean}_{\text{cols}}(O_t \odot O_t)
$$

where $\odot$ denotes element-wise multiplication, and $\operatorname{mean}_{\text{cols}}$ computes the mean along row dimension:

$$
(v_t)_i = \beta_2 (v_{t-1})_i + (1 - \beta_2) \frac{1}{n} \sum_{j=1}^n (O_t)_{ij}^2
$$

#### Step 4: Row-Wise Adaptive Normalization

The vector $v_t \in \mathbb{R}^m$ is broadcast across columns to create $V_t \in \mathbb{R}^{m \times n}$ ($V_{ij} = (v_t)_i$). The orthogonalized update is scaled entrywise by the inverse root of the neuron energy:

$$
\widehat{O}_t = O_t \oslash \left( \sqrt{V_t} + \epsilon \right)
$$

This guarantees that all neuron rows update at equalized relative speeds.

#### Step 5: Global Calibration and Weight Update

To prevent the scaling from drifting the effective global step size across different layer shapes, the update matrix is re-calibrated using the theoretical Frobenius expectation $\sqrt{mn}$ and a standard scaling coefficient (typically $0.2$):

$$
\widehat{\eta}_t = 0.2 \cdot \eta_t \cdot \frac{\sqrt{mn}}{\Vert{}\widehat{O}_t\Vert{}_F}
$$

$$
W_{t+1} = W_t - \eta_t \lambda W_t - \widehat{\eta}_t \widehat{O}_t
$$

*(where $\lambda$ is decoupled weight decay and $\eta_t$ is the base learning rate)*.

### 3\. Comparison Across Optimizers

| Property | AdamW | Standard Muon | NorMuon |
| --- | --- | --- | --- |
| **Geometry** | Coordinate-wise ($\ell_\infty$) | Spectral ($S_\infty$) | Spectral + Neuron-wise $\ell_2$ |
| **Matrix Condition Number** | Poor to Moderate | Well-conditioned ($\approx 1$) | Well-conditioned ($\approx 1$) |
| **Per-Neuron Update Variance** | Low (Uniform) | High (Imbalanced) | Low (Uniform) |
| **Optimizer Memory** | $2\times$ param count ($M,V$) | $1\times$ param count ($M$) | $1\times$ param count $+O(m)$ per matrix |
| **Distributed Communication** | Fully local row/col ops | Requires all-gather for SVD/NS5 | All-gather for NS5; local row normalization in FSDP2 |

### 4\. Why Distributed Scaling (FSDP2) Works Well

ResearchGate

Under modern **Fully Sharded Data Parallelism (FSDP2)**, parameters are partitioned row-wise across GPUs:

ResearchGate\+ 1

- The orthogonalization step ($\text{NS}_5$) requires matrix multiplications across the full matrix, which is assigned round-robin across GPUs.
	arXiv
- **The row-normalization step is purely local**: because FSDP2 shards parameter matrices by whole rows, each rank calculates the row-wise second moment $v_t$ on its own local shard without any cross-device communication overhead
