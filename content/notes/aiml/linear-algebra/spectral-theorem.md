---
title: "Spectral theorem"
date: "2026-10-26"
references: "https://www.youtube.com/watch?v=mhy-ZKSARxI"
---

The **Spectral Theorem** is a fundamental result in linear algebra and functional analysis that establishes when an operator or matrix can be diagonalized using an **orthonormal basis** of eigenvectors.

Geometrically, it means that under the right conditions, a linear transformation is simply a combination of independent stretches along mutually perpendicular axes.

### Finite-Dimensional Formulations

Depending on whether you work over real numbers or complex numbers, the theorem takes slightly different forms:

#### 1\. Real Symmetric Matrices

If $A \in \mathbb{R}^{n \times n}$ is a real symmetric matrix ($A = A^T$):

- All eigenvalues of $A$ are real numbers.
- Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
- $A$ can be factored as:
	$$
	A = Q \Lambda Q^T
	$$
	where $Q$ is an orthogonal matrix ($Q^T Q = I$) whose columns are orthonormal eigenvectors of $A$, and $\Lambda$ is a diagonal matrix containing the real eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$.

#### 2\. Complex Normal Matrices 

If $A \in \mathbb{C}^{n \times n}$, the theorem generalizes to **normal matrices**, defined by $A A^* = A^* A$ (where $A^*$ is the conjugate transpose). This family includes Hermitian ($A = A^*$), skew-Hermitian ($A = -A^*$), and unitary ($A^* A = I$) matrices:

- $A$ is unitarily diagonalizable:
	$$
	A = U \Lambda U^*
	$$
	where $U$ is a unitary matrix ($U^* U = I$) and $\Lambda$ is diagonal.
- If $A$ is specifically **Hermitian** ($A = A^*$), all its eigenvalues are strictly real, mirroring the real symmetric case.

### Spectral Decomposition (Projection Form) 

Instead of matrix products, the spectral theorem can be written as a sum of one-dimensional orthogonal projections:

$$
A = \sum_{i=1}^{n} \lambda_i P_i
$$

where each $P_i = u_i u_i^*$ is the projection operator onto the eigenspace spanned by the orthonormal eigenvector $u_i$. These projections satisfy:

- $P_i P_j = 0$ for $i \neq j$ (mutually orthogonal)
- $P_i^2 = P_i$ (idempotent)
- $\sum_{i=1}^{n} P_i = I$ (resolution of the identity)

This representation makes it trivial to compute powers, inverses, and functions of matrices:

$$
f(A) = \sum_{i=1}^{n} f(\lambda_i) P_i
$$

### Infinite-Dimensional Extension (Hilbert Spaces)

In quantum mechanics and functional analysis, the theorem extends to bounded (and unbounded) self-adjoint operators $T$ on a Hilbert space $\mathcal{H}$.

Because infinite-dimensional operators may possess continuous spectra (where eigenvalues in the discrete sense do not exist, like the position or momentum operators in quantum physics), the summation becomes an integral against a projection-valued measure $E$:

$$
T = \int_{\sigma(T)} \lambda \, dE(\lambda)
$$

where $\sigma(T)$ denotes the spectrum of $T$.

### Core Applications

- **Principal Component Analysis (PCA):** Diagonalizes the symmetric covariance matrix to find principal directions of maximal variance.
- **Singular Value Decomposition (SVD):** SVD applies the spectral theorem to $A^T A$ and $A A^T$ to decompose arbitrary rectangular matrices.
- **Quantum Mechanics:** Physical observables correspond to self-adjoint operators. The spectral theorem ensures that measurement outcomes (eigenvalues) are real and that the eigenstates form a complete basis for the state space.
- **Graph Theory:** Spectral graph theory uses the eigenvalues and eigenvectors of the graph Laplacian or adjacency matrix to analyze graph connectivity, clustering, and cuts.
