# GLM Extensions for Scikit-Learn

[![PyPI version](https://img.shields.io/pypi/v/glmext.svg)](https://pypi.org/project/glmext/)
[![Python versions](https://img.shields.io/pypi/pyversions/glmext.svg)](https://pypi.org/project/glmext/)
[![License](https://img.shields.io/pypi/l/glmext.svg)](https://pypi.org/project/glmext/)
[![Wheel](https://img.shields.io/pypi/wheel/glmext.svg)](https://pypi.org/project/glmext/#files)

This package provides **Generalized Linear Model (GLM)** implementations that are missing from scikit-learn, with a familiar *sklearn-style* API.

Currently supported models include:

- **Negative Binomial GLM (NB2 parameterization)**
- **Binomial GLM**

The goal is to offer a lightweight, numerically stable alternative for users who want GLMs integrated naturally into scikit-learn pipelines, without depending on `statsmodels`.

---

## Motivation

While scikit-learn provides a general GLM framework and logistic regression, it does **not** include support for:

- Negative Binomial regression (for overdispersed count data)
- Binomial regression

This package fills that gap while following scikit-learn conventions:

- `fit() / predict()` 
- Compatibility with pipelines and preprocessing
- Cython-compiled fast inference of loss  and gradient

---

## Installation

```bash
pip install glmext
```

---

## Example
[Example notebook](./notebooks/Example.ipynb)
```python
from glmext.glm import NegativeBinomialRegressor

nb = NegativeBinomialRegressor(
    alpha=0.0,
    k=2,
    max_iter=300,
    tol=1e-7,
)
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
```

---

## Mathematical Reference (Negative Binomial GLM)

We use the **NB2 parameterization**, where the conditional variance is

```
V(μ) = μ + k μ²
```

Here:

- `μ` is the conditional mean
- `k > 0` is the dispersion (overdispersion) parameter

---

### 1. Canonical Parameter

For the Negative Binomial exponential family, the **canonical parameter** is

```
θ_c(μ) = log( k μ / (1 + k μ) )
```

---

### 2. Cumulant Function

The corresponding cumulant function is

```
b(θ_c) = -1 / k · log(1 - exp(θ_c))
```

---

### 3. GLM Loss (Negative Log-Likelihood)

The per-observation negative log-likelihood is

```
ℓ(y, θ_c) = -( y · θ_c - b(θ_c) )
```

Summing over observations gives the total loss.

---

## Derivation of the Closed-Form Loss

Substituting `θ_c(μ)` and `b(θ_c)` into the likelihood:

```
Loss(y, μ) = ∑ [ -( y · log(k μ / (1 + k μ)) - 1/k · log(1 + k μ) ) ]
```

Expanding logarithms:

```
Loss(y, μ) = ∑ [ (y + 1/k) · log(1 + k μ) - y · log(k μ) ]
```

Dropping constant terms independent of `μ` (e.g. `y · log(k)`):

```
Loss(y, μ) = ∑ [ (y + 1/k) · log(1 + k μ) - y · log(μ) ]
```

This is the exact loss optimized by the implementation.

---

## Link Function and Implementation Details

We parameterize the mean using a **log link**:

```
μ = exp(raw_prediction)
log(μ) = raw_prediction
```

**Important:**

- This is **not** the canonical link for the Negative Binomial distribution.
- We intentionally work in log-mean (`log μ`) space for numerical stability and ease of optimization.

### Poisson Limit

As `k → 0`, the Negative Binomial converges to a Poisson distribution, and the loss reduces to the standard Poisson negative log-likelihood:

```
Loss(y, μ) = μ - y · log(μ)
```

### Final Loss Used in Code

Expressed in terms of `raw_prediction`:

```
Loss = ∑ [ (y + 1/k) · log(1 + k · exp(raw)) - y · raw ]
```

---

## Relation to `statsmodels`

This implementation is **numerically equivalent** to

```
statsmodels.genmod.families.NegativeBinomial.loglike_obs()
```

with the following clarifications.

---

### 1. Parameter Naming

- `statsmodels` uses `alpha` for the dispersion parameter
- This package uses `k`

The choice of `k` avoids a naming conflict with `alpha`, which is widely used for regularization strength in scikit-learn.

---

### 2. Canonical Parameter Notation Discrepancy

The `statsmodels` documentation sometimes writes the cumulant function as:

```
b(θ) = -1 / k · log(1 - k · exp(θ))
```

At first glance, this appears inconsistent with the canonical exponential-family form. The discrepancy arises because the symbol `θ` is reused for a **reparameterized quantity**, not the strict canonical parameter.

#### Distinguishing the Parameters

**Canonical parameter (exponential family):**

```
θ_c = log( k μ / (1 + k μ) )
```

**Reparameterized parameter used in practice:**

```
θ_r = log(μ)
```

These are related by:

```
exp(θ_c) = k · exp(θ_r) / (1 + k · exp(θ_r))
θ_c = log(k) + θ_r - log(1 + k · exp(θ_r))
```

Substituting `θ_r` into the canonical cumulant function yields:

```
b(θ_r) = -1 / k · log(1 - k · exp(θ_r))
```

Thus, the appearance of `k · exp(θ)` inside `b(θ)` reflects a **reparameterization**, not a different distribution.

---

### Practical Implication

In practice, both this package and `statsmodels`:

- Operate directly in `μ` or `log μ` space
- Use algebraically equivalent likelihood expressions
- Produce identical numerical results (up to floating-point precision)

The difference is largely one of **notation and documentation clarity** rather than statistical substance.

---

## License

MIT License