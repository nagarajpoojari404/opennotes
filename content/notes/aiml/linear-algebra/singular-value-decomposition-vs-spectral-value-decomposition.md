---
title: "Singular Value Decomposition vs Spectral Value Decomposition"
date: "2026-10-26"
---

Both **Spectral Decomposition** (also called the Spectral Theorem or Eigendecomposition of symmetric/normal matrices) and **Singular Value Decomposition (SVD)** factorize a matrix into fundamental geometric operations: rotations, reflections, and scaling along orthogonal axes.

The fundamental distinction lies in scope: **Spectral Decomposition applies to square, normal (typically symmetric/Hermitian) matrices using a single orthogonal basis**, whereas **SVD generalizes this to any $m \times n$ matrix by decoupling input and output coordinate systems**.

### 1\. Spectral Decomposition

If $A$ is an $n \times n$ real symmetric matrix ($A = A^T$), the **Spectral Theorem** guarantees that:

1. All eigenvalues $\lambda_i$ of $A$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
3. There exists an orthonormal basis of $\mathbb{R}^n$ composed entirely of eigenvectors of $A$.

#### Mathematical Formulation

$$
A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i q_i q_i^T
$$

- **$Q$ ($n \times n$):** An orthogonal matrix whose columns $q_1, \dots, q_n$ are the orthonormal eigenvectors ($Q^T Q = Q Q^T = I$).
- **$\Lambda$ ($n \times n$):** A diagonal matrix containing the eigenvalues $\lambda_1, \dots, \lambda_n$ along the main diagonal.
- **Outer product expansion:** $A$ is decomposed into a linear combination of rank-1 orthogonal projection matrices $q_i q_i^T$, each scaled by $\lambda_i$.

#### Geometric Interpretation

Multiplying a vector $x$ by $A$ decomposes into three steps:

1. **$Q^T x$:** Rotates/reflects $x$ into the eigenspace coordinates.
2. **$\Lambda (Q^T x)$:** Scales each coordinate independently by its eigenvalue $\lambda_i$ (can stretch, compress, or reverse direction if $\lambda_i < 0$).
3. **$Q (\dots)$:** Rotates/reflects the vector back to the original standard basis.

Because the domain and codomain are the exact same space ($\mathbb{R}^n \to \mathbb{R}^n$), the input coordinate system and output coordinate system are identical ($Q^T$ on the right, $Q$ on the left).

### 2\. Singular Value Decomposition (SVD)

SVD relaxes the symmetry and square constraints entirely. Any arbitrary real matrix $A \in \mathbb{R}^{m \times n}$ can be factored using two distinct orthogonal coordinate systems.

#### Mathematical Formulation

$$
A = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T
$$

- **$U$ ($m \times m$):** Orthogonal matrix whose columns $u_i$ are the **left-singular vectors** (orthonormal eigenvectors of $A A^T$).
- **$\Sigma$ ($m \times n$):** Rectangular diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, followed by zeros.
- **$V$ ($n \times n$):** Orthogonal matrix whose columns $v_i$ are the **right-singular vectors** (orthonormal eigenvectors of $A^T A$).
- **$r$:** The rank of $A$ ($r \le \min(m, n)$).

![Geometric action of SVD, AI generated](https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcQR3DI5aEBdNRUdtGjEtyauCPLHtOJXXiREz2CN5EonlcaoWRjOLX0V3UR3L3RO5tPV71VdL6WUL8q6q80)

Geometric action of SVD. Source: Made with Google AI

#### Geometric Interpretation

Any linear transformation maps an $n$\-dimensional unit hyper-sphere into an $m$\-dimensional hyper-ellipse:

1. **$V^T$:** Rotates the input space $\mathbb{R}^n$ so that the right-singular vectors $v_i$ align with standard coordinate axes.
2. **$\Sigma$:** Stretches/shrinks the axes by non-negative scale factors $\sigma_i$. If $m \neq n$, it embeds or projects into $\mathbb{R}^m$.
3. **$U$:** Rotates the resulting hyper-ellipse into its final orientation in the output space $\mathbb{R}^m$, aligning its semi-axes with $u_i$.

### 3\. Key Relationships & Differences

| Property | Spectral Decomposition ($Q\Lambda Q^\top$) | Singular Value Decomposition ($U\Sigma V^\top$) |
| --- | --- | --- |
| **Matrix Applicability** | Square ($n\times n$), Symmetric ($A=A^\top$) | Any Matrix ($m\times n$) |
| **Bases Involved** | One orthonormal basis ($Q$) | Two distinct orthonormal bases ($U$ and $V$) |
| **Diagonal Values** | Eigenvalues $\lambda_i\in\mathbb{R}$ (can be negative or zero) | Singular values $\sigma_i\ge 0$ (strictly non-negative) |
| **Subspace Mapping** | Maps $\mathbb{R}^n\to\mathbb{R}^n$ using same axes | Maps $\mathbb{R}^n\to\mathbb{R}^m$ across different dimensions |
| **Connection to Gram Matrices** | $A$ diagonalizes itself directly | $V$ diagonalizes $A^\top A$; $U$ diagonalizes $AA^\top$ |

#### The Exact Connection Between Them

When $A$ is symmetric:

- The singular values are the absolute values of the eigenvalues: $\sigma_i = \vert{}\lambda_i\vert{}$.
- The right-singular vectors $v_i$ correspond to the eigenvectors $q_i$.
- The left-singular vectors satisfy $u_i = \operatorname{sign}(\lambda_i) q_i$.

For an arbitrary matrix $A$, SVD can be viewed as computing the spectral decomposition of the symmetric positive semi-definite matrices $A^T A$ and $A A^T$:

$$
A^T A = (V \Sigma^T U^T)(U \Sigma V^T) = V (\Sigma^T \Sigma) V^T
$$

$$
A A^T = (U \Sigma V^T)(V \Sigma^T U^T) = U (\Sigma \Sigma^T) U^T
$$

Here, the eigenvalues of $A^T A$ (and $A A^T$) are precisely $\sigma_i^2$, and their orthonormal eigenvectors yield $V$ and $U$ respectively.
