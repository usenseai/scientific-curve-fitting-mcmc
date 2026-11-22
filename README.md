
Scientific parameter estimation using SciPy curve fitting, Bayesian inference and MCMC with emcee.

## Overview

This project performs scientific parameter estimation using experimental data. It combines nonlinear curve fitting with Bayesian parameter estimation using Markov Chain Monte Carlo (MCMC).

The current implementation uses an exponential decay model:

\[
y(t)=Ae^{-kt}+C
\]

where:

- **A:** Amplitude of the exponential component.
- **k:** Decay constant.
- **C:** Baseline or background signal.
- **t:** Time.

The project compares parameter estimates obtained from two approaches:

1. Nonlinear least-squares curve fitting using `scipy.optimize.curve_fit`.
2. Bayesian parameter estimation using the `emcee` MCMC sampler.

## Features

- Load experimental data from CSV files.
- Visualize experimental measurements.
- Perform nonlinear curve fitting.
- Estimate parameter uncertainties using the covariance matrix.
- Evaluate the fit using reduced chi-square.
- Perform Bayesian parameter estimation using MCMC.
- Apply prior constraints to model parameters.
- Analyze posterior samples and parameter uncertainties.
- Inspect MCMC trace plots.
- Plot posterior distributions and compare them with curve-fitting estimates.
- Estimate autocorrelation time to assess sampling efficiency.

## Mathematical Model

The exponential decay model is:

\[
y(t)=Ae^{-kt}+C
\]

The Gaussian log-likelihood used in the analysis is:

\[
\ln L=-\frac{1}{2}\sum_i
\left(\frac{y_i-y_{\mathrm{model},i}}{\sigma_i}\right)^2
\]

The current prior requires:

\[
k>0
\]

The posterior is proportional to:

\[
P(\theta|D)\propto L(D|\theta)P(\theta)
\]

where \(\theta=(A,k,C)\) represents the model parameters and \(D\) represents the experimental data.

## Technologies and Libraries

- Python
- NumPy
- Matplotlib
- SciPy
- emcee

## Project Structure

```text
scientific-curve-fitting-mcmc/
├── ED.csv
├── Sci_MCMC_Curve_fit.py
├── README.md
└── requirements.txt

## Installation

Clone the repository:

```bash
git clone https://github.com/usenseai/scientific-curve-fitting-mcmc.git
## Running the Project

Make sure ED.csv is in the project directory.

Run the program:

```bash
python Sci_MCMC_Curve_fit.py
