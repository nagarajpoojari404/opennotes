---
title: "Orthogonal vs symmetric matrices"
date: "2026-10-26"
---

### Symmetric Matrices

A real square matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric** if it is equal to its transpose:

$$
A = A^T
$$

In component form, this means entry $A_{ij} = A_{ji}$ for all indices $i, j$. Geometrically, the entries are mirrored across the main diagonal.


#### Key Properties

- **Real Eigenvalues:** All eigenvalues of a real symmetric matrix are guaranteed to be real numbers ($\lambda \in \mathbb{R}$).
- **Orthogonal Eigenspaces:** Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal. If $A v_1 = \lambda_1 v_1$ and $A v_2 = \lambda_2 v_2$ with $\lambda_1 \neq \lambda_2$, then $v_1 \cdot v_2 = 0$.
- **Spectral Theorem:** Every real symmetric matrix can be orthogonally diagonalized:
	$$
	A = Q \Lambda Q^T
	$$
	where $Q$ is an orthogonal matrix composed of normalized eigenvectors, and $\Lambda$ is a diagonal matrix containing the real eigenvalues.
- **Geometric Action:** Acting as a linear transformation $x \mapsto Ax$, a symmetric matrix stretches or compresses space along a set of mutually perpendicular axes (its eigenvectors) without introducing any shear or rotation relative to those axes.

### Orthogonal Matrices

A real square matrix $Q \in \mathbb{R}^{n \times n}$ is **orthogonal** if multiplying it by its transpose yields the identity matrix:

$$
Q^T Q = Q Q^T = I
$$

This directly implies that the transpose is the inverse:

$$
Q^{-1} = Q^T
$$

The columns (and rows) of $Q$ form an **orthonormal basis** of $\mathbb{R}^n$. If the columns are $q_1, q_2, \dots, q_n$:

- **Unit length:** $\Vert{}q_i\Vert{} = 1$ for all $i$.
- **Pairwise perpendicular:** $q_i \cdot q_j = 0$ whenever $i \neq j$.
```
Rotation by θ in 2D:
Q = [  cos θ   -sin θ ]
    [  sin θ    cos θ ]
```

#### Key Properties

- **Isometry (Preserves Geometry):** Orthogonal transformations preserve vector lengths and inner products (angles):
	$$
	\Vert{}Qx\Vert{} = \Vert{}x\Vert{}, \quad (Qx) \cdot (Qy) = x \cdot y
	$$
- **Determinant:** $\det(Q) = \pm 1$.
	- If $\det(Q) = +1$, $Q$ represents a pure **rotation** (orientation-preserving).
		- If $\det(Q) = -1$, $Q$ represents a **reflection** or a roto-reflection (orientation-reversing).
- **Eigenvalues on the Unit Circle:** The complex eigenvalues of $Q$ have an absolute value of 1 ($\vert{}\lambda\vert{} = 1$), taking the form $e^{i\theta}$.
- **Closure:** The product of two orthogonal matrices is also orthogonal ($(Q_1 Q_2)^T (Q_1 Q_2) = I$), forming the orthogonal group $\text{O}(n)$.

### Comparison & Interconnection

| Property | Symmetric Matrix ($A=A^\top$) | Orthogonal Matrix ($Q^\top Q=I$) |
| --- | --- | --- |
| **Inverse** | No simple formula (requires inversion algorithm) | Trivial: $Q^{-1}=Q^\top$ |
| **Eigenvalues** | Always real ($\lambda\in\mathbb{R}$) | Complex numbers on unit circle ($\lvert\lambda\rvert=1$) |
| **Geometric Effect** | Scaling along perpendicular axes | Rigid motion: rotations and reflections |
| **Preserves Norm?** | No (scales vectors by eigenvalues) | Yes ($\lVert Qx\rVert=\lVert x\rVert$) |

#### How They Connect

The **Spectral Theorem** links both concepts directly: any real symmetric matrix $A$ can be decomposed using an orthogonal matrix $Q$:

$$
A = Q \Lambda Q^T
$$

Here, $Q$ rotates coordinate axes to align with the matrix's principal directions, $\Lambda$ stretches or scales along those axes, and $Q^T$ rotates them back.
