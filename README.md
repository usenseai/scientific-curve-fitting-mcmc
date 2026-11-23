# Scientific Curve Fitting and MCMC

A Streamlit app for fitting experimental data with a chosen model and estimating parameter uncertainties with Bayesian MCMC sampling (`emcee`).

## Features

- Upload a CSV dataset.
- Three input modes: existing columns with an optional uncertainty column, replicate columns, or replicate rows.
- Works without measurement uncertainties: the noise level is estimated from the data.
- Automatic mean and standard error of the mean (SEM) from replicates.
- Four models: exponential, linear, Gaussian and sinusoidal.
- Least-squares fit (`scipy.optimize.curve_fit`) with parameter uncertainties.
- Goodness of fit: $\chi^2$ and reduced $\chi^2$ (with uncertainties), or residual standard deviation and $R^2$ (without).
- MCMC sampling with trace plots, posterior histograms and an autocorrelation-time check.
- Download of the processed data and of the fit and MCMC results as CSV.

## Installation

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
streamlit
numpy
pandas
scipy
matplotlib
emcee
```

## Run

```bash
streamlit run Sci_MCMC_Curve_fit.py
```

## Input modes

All columns used by the app must be numeric.

### 1. Existing columns (y and optional sigma)

Select one independent variable $x$ and one dependent variable $y$. The uncertainty column $\sigma$ is optional:

- **Column selected:** a weighted fit and MCMC are performed with the given $\sigma_i$. Rows with non-finite values or $\sigma_i \le 0$ are removed.
- **"None (estimate from data)" selected:** see *Without measurement uncertainties* below.

### 2. Replicate columns (wide format)

Each replicate is its own column, for example:

| Time | Cg_1 | Cg_2 | Cg_3 | YPD_1 | YPD_2 | YPD_3 |
|------|------|------|------|-------|-------|-------|
| 0    | ...  | ...  | ...  | ...   | ...   | ...   |

Select the independent variable, then the replicate columns of one group. Groups such as `Cg` and `YPD` are suggested automatically from column names that differ only by a trailing number (`_1`, `_2`, `_3`). The suggestion can be edited, so any column names work.

### 3. Replicate rows (long format)

Several rows share the same $x$ value. The app groups the rows by $x$ and averages the selected $y$ column.

## Mean and uncertainty from replicates

In both replicate modes, at each $x_i$ with $n_i$ valid replicates $R_{ik}$:

$$\bar{y}_i = \frac{1}{n_i}\sum_{k=1}^{n_i} R_{ik}$$

$$s_i = \sqrt{\frac{1}{n_i-1}\sum_{k=1}^{n_i}\left(R_{ik}-\bar{y}_i\right)^2}$$

$$\sigma_i = \frac{s_i}{\sqrt{n_i}}$$

The fit and the MCMC use $\bar{y}_i$ as $y$ and $\sigma_i$ as the uncertainty.

### Zero or undefined uncertainty

If the replicates are identical ($\sigma_i = 0$) or fewer than two values are available ($n_i < 2$), the weight $1/\sigma_i^2$ is infinite or undefined. The app does not accept such points silently. You choose:

- **drop**: remove those points;
- **floor**: replace $\sigma_i$ with a positive value you set, for example the instrument resolution.

## Without measurement uncertainties

If no uncertainty column is available, the app assumes one constant Gaussian noise level $\sigma$ for all points and estimates it.

Fit: ordinary (unweighted) least squares, minimising

$$\mathrm{RSS} = \sum_{i=1}^{N}\left(y_i - f(x_i;\boldsymbol{\theta})\right)^2$$

The noise level is estimated from the residuals,

$$\hat{\sigma} = \sqrt{\frac{\mathrm{RSS}}{N-p}}$$

and the parameter covariance from `curve_fit` is scaled by this estimate (`absolute_sigma=False`).

MCMC: $\ln\sigma$ is added as an extra free parameter. The log-likelihood is

$$\ln \mathcal{L}(\boldsymbol{\theta},\sigma) = -\frac{1}{2}\sum_{i=1}^{N}\left(\frac{y_i - f(x_i;\boldsymbol{\theta})}{\sigma}\right)^2 - N\ln\sigma + \text{const.}$$

with a flat prior on $\ln\sigma$ over a wide finite range. The MCMC table reports the noise level as `sigma (noise)`.

Goodness of fit: $\chi^2$ cannot be computed without uncertainties, so the app reports $\mathrm{RSS}$, $\hat{\sigma}$ and

$$R^2 = 1 - \frac{\mathrm{RSS}}{\sum_i (y_i-\bar{y})^2}$$

## Models

| Model | Function | Parameters |
|-------|----------|------------|
| Exponential | $y = A e^{-kx} + C$ | $A,\ k,\ C$ |
| Linear | $y = mx + c$ | $m,\ c$ |
| Gaussian | $y = A \exp\!\left[-\dfrac{(x-\mu)^2}{2s^2}\right] + C$ | $A,\ \mu,\ s,\ C$ |
| Sinusoidal | $y = A \sin(\omega x + \phi) + C$ | $A,\ \omega,\ \phi,\ C$ |

Parameters that must be positive ($k$ for the exponential, $s$ for the Gaussian, $\omega$ for the sinusoidal) are bounded in the fit and constrained by the MCMC prior.

## Statistical method (with uncertainties)

Curve fit: weighted least squares with `absolute_sigma=True`, minimising

$$\chi^2 = \sum_{i=1}^{N}\left(\frac{y_i - f(x_i;\boldsymbol{\theta})}{\sigma_i}\right)^2, \qquad \chi^2_\nu = \frac{\chi^2}{N-p}$$

where $N$ is the number of points and $p$ the number of parameters.

MCMC: Gaussian likelihood with a flat prior inside the allowed parameter region,

$$\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2}\sum_{i=1}^{N}\left(\frac{y_i - f(x_i;\boldsymbol{\theta})}{\sigma_i}\right)^2 + \text{const.}$$

Sampling uses the affine-invariant ensemble sampler of `emcee` with at least 32 walkers, 5000 steps and 1000 burn-in steps. Chain length is checked against the rule of thumb of 50 integrated autocorrelation times.

## Output

- Table of fitted parameters with uncertainties.
- Plots of the data and of the fitted curve.
- $\chi^2$ and reduced $\chi^2$, or $\mathrm{RSS}$, $\hat{\sigma}$ and $R^2$ when no uncertainties are given.
- MCMC table (mean and standard deviation of each parameter, including `sigma
