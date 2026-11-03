---
title: "PCA"
date: "2026-10-26"
---

**Principal Component Analysis (PCA)** is an unsupervised linear dimensionality reduction technique. It transforms a large set of correlated variables into a smaller set of uncorrelated variables called **principal components**, while retaining as much variance (information) as possible.

### The Core Intuition

Imagine holding a 3D object—like a teapot—in front of a lamp so its shadow falls on a wall. Depending on how you rotate the teapot, the 2D shadow might look like a featureless blob, or it might clearly show the spout, handle, and body.

PCA mathematically rotates your high-dimensional data cloud to find the angle that casts the widest, clearest "shadow." In data terms:

- **Maximum spread equals maximum information:** Where data points are spread out the most (highest variance), the differences between them are preserved.
- **Minimum spread equals noise or redundancy:** Dimensions where points barely vary are discarded with minimal loss of information.

### How PCA Works Step-by-Step

Given a dataset $X$ with $n$ observations and $p$ features:

1. **Standardization (Centering and Scaling)** Features often operate on different scales (e.g., age in tens vs. income in tens of thousands). PCA is sensitive to scale, so each feature is standardized to have a mean of $0$ and a standard deviation of $1$:
	$$
	z = \frac{x - \mu}{\sigma}
	$$
2. **Covariance Matrix Computation** Calculate the $p \times p$ covariance matrix $\Sigma$:
	$$
	\Sigma = \frac{1}{n-1} X^T X
	$$
	This matrix captures how every feature varies relative to every other feature. Off-diagonal elements show whether pairs of variables are positively or negatively correlated.
3. **Eigenvalue Decomposition** Solve for the eigenvalues $\lambda$ and eigenvectors $v$ of the covariance matrix:
	$$
	\Sigma v = \lambda v
	$$
	- **Eigenvectors** define the **directions** of the new feature space (the principal axes). They are mutually perpendicular (orthogonal), meaning zero correlation exists between them.
		- **Eigenvalues** quantify the **amount of variance** carried by each corresponding eigenvector.
4. **Selecting the Principal Components** Rank the eigenvectors by their eigenvalues in descending order:
	- **PC1:** The axis capturing the single largest variance in the data.
		- **PC2:** The axis orthogonal to PC1 capturing the next largest variance, and so on. You select the top $k$ components that explain a target threshold of total variance (commonly 80% to 95%).
5. **Projecting onto the New Subspace** Form a projection matrix $W$ using the top $k$ eigenvectors and transform the original data $X$ into the lower-dimensional representation $Z$:
	$$
	Z = X W
	$$

### Why Use PCA?

- **Visualization:** Reduces 10D, 100D, or 10,000D datasets down to 2D or 3D scatter plots to inspect clusters, classes, and outliers.
- **Mitigating the Curse of Dimensionality:** Models run faster and require less memory when sparse, high-dimensional inputs are condensed.
- **Removing Multicollinearity:** Regressions and linear models struggle with heavily correlated inputs. Principal components are completely orthogonal to each other.
- **Noise Reduction:** Discarding lower-variance components often strips out background noise and measurement jitter.

### Key Limitations

- **Linear Only:** PCA assumes linear relationships between features. For complex manifolds or nonlinear structures, techniques like **t-SNE**, **UMAP**, or **Kernel PCA** are required.
- **Interpretability Trade-off:** While the original features had concrete meanings (e.g., "annual salary", "blood pressure"), principal components are linear combinations of all original features (e.g., $0.42 \times \text{salary} - 0.31 \times \text{pressure}$), making direct attribution harder.
- **Outlier Sensitivity:** Because variance is computed using squared distances from the mean, extreme outliers can distort the direction of the principal axes.
