
# Scientific Curve Fitting and MCMC

A Python-based interactive application for scientific data analysis using Streamlit, SciPy, NumPy, and emcee.

## Features

- Upload experimental datasets in CSV format.
- Select independent, dependent, and uncertainty columns.
- Visualize experimental data with error bars.
- Perform nonlinear and linear curve fitting.
- Estimate parameter uncertainties using the covariance matrix.
- Calculate chi-square and reduced chi-square.
- Perform Bayesian parameter estimation using MCMC.
- Analyze MCMC chains using trace plots.
- Visualize posterior parameter distributions.
- Compare curve-fitting and MCMC estimates.
- Download parameter estimation results.

## Supported Mathematical Models

1. Exponential:
   y = A exp(-kx) + C

2. Linear:
   y = mx + c

3. Gaussian:
   y = A exp(-(x-mu)^2/(2s^2)) + C

4. Sinusoidal:
   y = A sin(wx + phi) + C

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- emcee
- Streamlit

## Installation

Clone the repository:

    git clone https://github.com/usenseai/scientific-curve-fitting-mcmc.git

Move into the project directory:

    cd scientific-curve-fitting-mcmc

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run Sci_MCMC_Curve_fit.py

## Dataset Format

Upload a CSV file containing at least three numeric columns:

- Independent variable (x)
- Dependent variable (y)
- Measurement uncertainty (sigma)

The uncertainty values must be positive.

## Methodology

1. Experimental data loading and visualization.
2. Mathematical model selection.
3. Parameter estimation using SciPy curve_fit.
4. Uncertainty estimation using the covariance matrix.
5. Goodness-of-fit analysis using reduced chi-square.
6. Bayesian parameter estimation using MCMC.
7. Posterior distribution and convergence diagnostics.

## Author

UsenseAI
