---
title: "Stats"
date: "2026-10-26"
---

Probability distributions are categorized into **discrete** (countable outcomes) and **continuous** (unbroken range of outcomes).

### Discrete Distributions

#### 1\. Bernoulli Distribution

Models a single trial with only two possible outcomes: success ($x = 1$) with probability $p$, or failure ($x = 0$) with probability $1 - p$.

- **Probability Mass Function (PMF):**
	$$
	P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}
	$$
- **Parameters & Moments:** Mean $\mu = p$, Variance $\sigma^2 = p(1 - p)$
- **Example:** Tossing a biased coin once where landing heads has probability $p = 0.7$.

#### 2\. Binomial Distribution

Models the number of successes $k$ in $n$ independent and identical Bernoulli trials.

- **PMF:**
	$$
	P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}
	$$
	$$
	\binom{n}{k} = \frac{n!}{k!(n - k)!}
	$$
- **Parameters & Moments:** Mean $\mu = np$, Variance $\sigma^2 = np(1 - p)$
- **Example:** Flipping a fair coin ($p = 0.5$) $10$ times ($n = 10$) and calculating the probability of getting exactly $k = 4$ heads:
	$$
	P(X = 4) = \binom{10}{4} (0.5)^4 (0.5)^6 = 210 \cdot (0.5)^{10} \approx 0.205
	$$

#### 3\. Poisson Distribution

Models the number of independent events occurring within a fixed interval of time or space, given a known constant average rate $\lambda$.

- **PMF:**
	$$
	P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \dots\}
	$$
- **Parameters & Moments:** Mean $\mu = \lambda$, Variance $\sigma^2 = \lambda$
- **Example:** A web server receives an average of $\lambda = 3$ requests per millisecond. The probability of handling exactly $1$ request in a given millisecond is:
	$$
	P(X = 1) = \frac{3^1 e^{-3}}{1!} \approx 3 \cdot 0.0498 = 0.149
	$$

#### 4\. Geometric Distribution

Models the number of Bernoulli trials required to get the **first** success.

- **PMF:**
	$$
	P(X = k) = (1 - p)^{k - 1} p, \quad k \in \{1, 2, 3, \dots\}
	$$
- **Parameters & Moments:** Mean $\mu = \frac{1}{p}$, Variance $\sigma^2 = \frac{1 - p}{p^2}$
- **Example:** Rolling a standard six-sided die until rolling a $6$ ($p = 1/6$). The probability that the first $6$ appears on the 3rd roll ($k = 3$) is:
	$$
	P(X = 3) = \left(\frac{5}{6}\right)^2 \left(\frac{1}{6}\right) = \frac{25}{216} \approx 0.116
	$$

### Continuous Distributions

#### 5\. Uniform Distribution (Continuous)

Every sub-interval of equal length within $[a, b]$ is equally likely.

- **Probability Density Function (PDF):**
	$$
	f(x) = \begin{cases} \frac{1}{b - a} & \text{for } a \le x \le b \\ 0 & \text{otherwise} \end{cases}
	$$
- **Parameters & Moments:** Mean $\mu = \frac{a + b}{2}$, Variance $\sigma^2 = \frac{(b - a)^2}{12}$
- **Example:** A bus arrives randomly at any point within a 15-minute window from $0$ to $15$ minutes ($a = 0, b = 15$). The probability of waiting between 5 and 10 minutes is:
	$$
	P(5 \le X \le 10) = \int_5^{10} \frac{1}{15} \, dx = \frac{10 - 5}{15} = \frac{1}{3}
	$$

#### 6\. Normal (Gaussian) Distribution

A symmetric, bell-shaped distribution centered around its mean $\mu$ with spread dictated by variance $\sigma^2$. It naturally arises via the Central Limit Theorem.

- **PDF:**
	$$
	f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right), \quad x \in (-\infty, \infty)
	$$
- **Parameters & Moments:** Mean $\mu$, Variance $\sigma^2$
- **Example:** Adult human heights in a population modeled with $\mu = 175\text{ cm}$ and $\sigma = 7\text{ cm}$. Approximately $68.2\%$ of individuals fall within $[\mu - \sigma, \mu + \sigma] = [168\text{ cm}, 182\text{ cm}]$.

#### 7\. Exponential Distribution

Models the time elapsed between Poisson point process events occurring continuously and independently at a constant average rate $\lambda$. It is memoryless: $P(X > s + t \mid X > s) = P(X > t)$.

- **PDF:**
	$$
	f(x) = \begin{cases} \lambda e^{-\lambda x} & \text{for } x \ge 0 \\ 0 & \text{for } x < 0 \end{cases}
	$$
- **Parameters & Moments:** Mean $\mu = \frac{1}{\lambda}$, Variance $\sigma^2 = \frac{1}{\lambda^2}$
- **Example:** A solid-state drive has an average failure rate of $\lambda = 0.05$ failures per year (mean lifetime $\mu = 20$ years). The probability it fails within the first 2 years ($x \le 2$) is:
	$$
	P(X \le 2) = \int_0^2 0.05 e^{-0.05 x} \, dx = 1 - e^{-0.05 \cdot 2} = 1 - e^{-0.1} \approx 0.095
	$$

### Overview Summary

| Distribution | Type | Support | PMF / PDF | Mean ($\mu$) | Variance ($\sigma^2$) |
| --- | --- | --- | --- | --- | --- |
| **Bernoulli** | Discrete | $x\in\{0,1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| **Binomial** | Discrete | $k\in\{0,\dots,n\}$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| **Poisson** | Discrete | $k\in\{0,1,2,\dots\}$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| **Geometric** | Discrete | $k\in\{1,2,\dots\}$ | $(1-p)^{k-1}p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |
| **Uniform** | Continuous | $x\in[a,b]$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| **Normal** | Continuous | $x\in\mathbb{R}$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| **Exponential** | Continuous | $x\in[0,\infty)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
