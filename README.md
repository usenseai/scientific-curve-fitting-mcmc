# Scientific Curve Fitting and MCMC

A Streamlit app for fitting experimental data with a chosen model and estimating parameter uncertainties with Bayesian MCMC sampling (`emcee`).

## Features

- Upload a CSV dataset.
- Three input modes: existing `y` and `sigma` columns, replicate columns, or replicate rows.
- Automatic mean and standard error of the mean (SEM) from replicates.
- Four models: exponential, linear, Gaussian and sinusoidal.
- Weighted least-squares fit (`scipy.optimize.curve_fit`) with parameter uncertainties.
- Goodness of fit: $\chi^2$ and reduced $\chi^2$.
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

### 1. Existing columns (y and sigma)

Select one independent variable $x$, one dependent variable $y$ and one uncertainty column $\sigma$. Rows with non-finite values or $\sigma \le 0$ are removed.

### 2. Replicate columns (wide format)

Each replicate is its own column, for example:

| Time | Cg_1 | Cg_2 | Cg_3 | YPD_1 | YPD_2 | YPD_3 |
|------|------|------|------|-------|-------|-------|
| 0    | ...  | ...  | ...  | ...   | ...   | ...   |

Select the independent variable, then the replicate columns of one group. Groups such as `Cg` and `YPD` are suggested automatically from column names that differ only by a trailing number (`_1`, `_2`, `_3`). The suggestion can be edited, so any column names work.

### 3. Replicate rows (long format)

Several rows share the same $x$ value. The app groups the rows by $x$ and averages the selected $y$ column.

## Mean and uncertainty

In both replicate modes, at each $x_i$ with $n_i$ valid replicates $R_{ik}$:

$$\bar{y}_i = \frac{1}{n_i}\sum_{k=1}^{n_i} R_{ik}$$

$$s_i = \sqrt{\frac{1}{n_i-1}\sum_{k=1}^{n_i}\left(R_{ik}-\bar{y}_i\right)^2}$$
