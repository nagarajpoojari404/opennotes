---
title: "Tensors"
date: "2026-10-26"
---

At its core, a **tensor** is a geometric object that maps geometric entities (vectors and covectors) linearly to real numbers, retaining its physical identity regardless of the coordinate system chosen to describe it.

While computer science views tensors merely as multi-dimensional arrays, physics and mathematics define them by **how their components transform under a change of basis**.

## 1\. Dual Spaces, Vectors, and Covectors

To understand tensors formally, we start on a vector space $V$ of dimension $n$ over $\mathbb{R}$:

- **Contravariant Vectors (Tangent Vectors, $V$):** Elements of $V$. If $\{e_\mu\}$ forms a basis of $V$, a vector $v$ is represented with an upper index:
	$$
	v = v^\mu e_\mu
	$$
- **Covariant Vectors / 1-Forms / Covectors ($V^*$):** Linear functions mapping $V \to \mathbb{R}$. If $\{\theta^\nu\}$ is the dual basis satisfying $\theta^\nu(e_\mu) = \delta^\nu_\mu$, a 1-form $\omega \in V^*$ is written with a lower index:
	$$
	\omega = \omega_\nu \theta^\nu
	$$

Under a coordinate transformation $x^\mu \to x'^\mu$:

- Tangent vectors transform via the Jacobian matrix:
	$$
	v'^\mu = \frac{\partial x'^\mu}{\partial x^\alpha} v^\alpha
	$$
- Dual vectors (gradients of scalars, 1-forms) transform via the inverse Jacobian:
	$$
	\omega'_\mu = \frac{\partial x^\alpha}{\partial x'^\mu} \omega_\alpha
	$$

## 2\. General Tensor Definition

A tensor of type $(r, s)$ (rank $r+s$) is a multilinear map:

$$
T: \underbrace{V^* \times \dots \times V^*}_{r \text{ copies}} \times \underbrace{V \times \dots \times V}_{s \text{ copies}} \longrightarrow \mathbb{R}
$$

- $r$ is the **contravariant rank** (number of upper indices).
- $s$ is the **covariant rank** (number of lower indices).

### Transformation Law

A tensor $T$ of type $(r, s)$ transforms under coordinate change $x \to x'$ as:

$$
{T'}^{\mu_1 \dots \mu_r}_{\nu_1 \dots \nu_s} = \left(\frac{\partial x'^{\mu_1}}{\partial x^{\alpha_1}} \dots \frac{\partial x'^{\mu_r}}{\partial x^{\alpha_r}}\right) \left(\frac{\partial x^{\beta_1}}{\partial x'^{\nu_1}} \dots \frac{\partial x^{\beta_s}}{\partial x'^{\nu_s}}\right) T^{\alpha_1 \dots \alpha_r}_{\beta_1 \dots \beta_s}
$$

This transformation rule is the operational litmus test: an array of numbers is a tensor if and only if its components transform strictly according to this multilinear law.

## 3\. Essential Tensor Algebra

### Einstein Summation Convention

Repeated upper and lower indices imply summation over the dimension range ($1$ to $n$ or $0$ to $3$):

$$
A^\mu B_\mu \equiv \sum_{\mu=1}^n A^\mu B_\mu
$$

### Tensor Product ($\otimes$)

Combines a tensor of type $(r, s)$ and $(p, q)$ into a tensor of type $(r+p, s+q)$:

$$
(A \otimes B)^{\mu_1 \dots \mu_r \alpha_1 \dots \alpha_p}_{\nu_1 \dots \nu_s \beta_1 \dots \beta_q} = A^{\mu_1 \dots \mu_r}_{\nu_1 \dots \nu_s} B^{\alpha_1 \dots \alpha_p}_{\beta_1 \dots \beta_q}
$$

### Contraction

Setting an upper index equal to a lower index sums over that coordinate, reducing the tensor's rank by 2:

$$
C^\mu_{\nu} \xrightarrow{\text{contraction}} C^\mu_\mu = \operatorname{Tr}(C) \quad (\text{a scalar})
$$

### Metric Tensor ($g_{\mu\nu}$)

A symmetric $(0, 2)$ tensor that defines the inner product on the tangent space and sets the line element:

$$
ds^2 = g_{\mu\nu} dx^\mu dx^\nu
$$

The metric is invertible ($g^{\mu\alpha} g_{\alpha\nu} = \delta^\mu_\nu$) and provides a natural isomorphism to **raise and lower indices**:

$$
v_\mu = g_{\mu\nu} v^\nu, \qquad \omega^\mu = g^{\mu\nu} \omega_\nu
$$

## 4\. Tensor Calculus on Curved Manifolds

On curved spaces (or even flat spaces in curvilinear coordinates), the partial derivative of a tensor component, $\partial_\mu T^\nu \equiv \frac{\partial T^\nu}{\partial x^\mu}$, **fails to transform as a tensor** because the basis vectors themselves vary from point to point:

$$
\frac{\partial e_\alpha}{\partial x^\mu} \neq 0
$$

### The Covariant Derivative ($\nabla$ or $;$)

To construct a genuine directional derivative that transforms tensorially, we introduce an affine connection $\Gamma^\lambda_{\mu\nu}$ (the Christoffel symbols):

For a vector $v^\nu$:

$$
\nabla_\mu v^\nu = \partial_\mu v^\nu + \Gamma^\nu_{\mu\lambda} v^\lambda
$$

For a 1-form $\omega_\nu$:

$$
\nabla_\mu \omega_\nu = \partial_\mu \omega_\nu - \Gamma^\lambda_{\mu\nu} \omega_\lambda
$$

For a general $(r, s)$ tensor, add a $+\Gamma$ term for every contravariant index and a $-\Gamma$ term for every covariant index:

$$
\nabla_\lambda T^\mu_\nu = \partial_\lambda T^\mu_\nu + \Gamma^\mu_{\lambda\sigma} T^\sigma_\nu - \Gamma^\sigma_{\lambda\nu} T^\mu_\sigma
$$

### Christoffel Symbols of the Second Kind

Under the **Levi-Civita connection** (which is metric-compatible, $\nabla_\sigma g_{\mu\nu} = 0$, and torsion-free, $\Gamma^\lambda_{\mu\nu} = \Gamma^\lambda_{\nu\mu}$), the connection coefficients are uniquely determined by the metric:

$$
\Gamma^\lambda_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma} \left( \partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu} \right)
$$

> **Crucial Note:** $\Gamma^\lambda_{\mu\nu}$ is **not** a tensor. Its transformation law contains an inhomogeneous second-derivative term $\frac{\partial^2 x^\alpha}{\partial x'^\mu \partial x'^\nu}$, which precisely cancels the non-tensorial term produced by differentiating components.

## 5\. Curvature Tensors

Because covariant derivatives do not generally commute on curved spaces, their commutator reveals the intrinsic curvature of the manifold.

### Riemann Curvature Tensor

Acting with the commutator on a vector field:

$$
[\nabla_\mu, \nabla_\nu] v^\rho = (\nabla_\mu \nabla_\nu - \nabla_\nu \nabla_\mu) v^\rho = R^\rho_{\ \sigma\mu\nu} v^\sigma
$$

Expressed explicitly in terms of Christoffel symbols:

$$
R^\rho_{\ \sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda} \Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda} \Gamma^\lambda_{\mu\sigma}
$$

### Key Contractions of the Riemann Tensor

1. **Ricci Tensor ($R_{\mu\nu}$):** Contraction over the first and third indices:
	$$
	R_{\mu\nu} = R^\lambda_{\ \mu\lambda\nu}
	$$
2. **Ricci Scalar ($R$):** Contraction with the inverse metric:
	$$
	R = g^{\mu\nu} R_{\mu\nu}
	$$
3. **Einstein Tensor ($G_{\mu\nu}$):** Divergence-free tensor governing spacetime geometry in General Relativity ($\nabla^\mu G_{\mu\nu} = 0$ via the Bianchi identities):
	$$
	G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}
	$$
