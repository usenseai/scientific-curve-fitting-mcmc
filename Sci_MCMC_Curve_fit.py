from replicates import suggest_groups, replicate_table, long_table
import numpy as np
import matplotlib.pyplot as plt
import scipy
import emcee
import streamlit as st
import pandas as pd

# --------------------------------------------------
# 1. STREAMLIT APPLICATION
# --------------------------------------------------

st.set_page_config(page_title="Scientific Curve Fitting and MCMC",
                   layout="wide")

st.title("Scientific Curve Fitting and MCMC")
st.write("Upload experimental data, select a mathematical model, and perform curve fitting and Bayesian MCMC analysis.")

# --------------------------------------------------
# 2. DATASET UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df)

# Select numeric columns only
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 3:
    st.error("Your dataset must contain at least three numeric columns: x, y and uncertainty.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    x_col = st.selectbox("Independent variable (x)", numeric_columns)

with col2:
    y_col = st.selectbox("Dependent variable (y)", numeric_columns)

with col3:
    sigma_col = st.selectbox("Uncertainty (sigma)", numeric_columns)

if len({x_col, y_col, sigma_col}) != 3:
    st.error("Please select three different columns.")
    st.stop()

x = df[x_col].to_numpy(dtype=float)
y = df[y_col].to_numpy(dtype=float)
sigma = df[sigma_col].to_numpy(dtype=float)

# Remove invalid rows
valid = (
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(sigma) &
    (sigma > 0)
)

x = x[valid]
y = y[valid]
sigma = sigma[valid]

if len(x) == 0:
    st.error("No valid data points found. Check your dataset and uncertainties.")
    st.stop()

if np.ptp(x) == 0:
    st.error("The independent variable must contain more than one distinct value.")
    st.stop()

st.write(f"Number of valid data points: {len(x)}")

# --------------------------------------------------
# 3. MATHEMATICAL MODELS
# --------------------------------------------------

def exponential(x, A, k, C):
    return A * np.exp(-k * x) + C


def linear(x, m, c):
    return m * x + c


def gaussian(x, A, mu, s, C):
    return A * np.exp(-(x - mu)**2 / (2 * s**2)) + C


def sinusoidal(x, A, w, phi, C):
    return A * np.sin(w * x + phi) + C


models = {
    "Exponential": exponential,
    "Linear": linear,
    "Gaussian": gaussian,
    "Sinusoidal": sinusoidal
}

model_name = st.selectbox(
    "Select mathematical model",
    list(models.keys())
)

model = models[model_name]

# Parameter names and initial guesses
x_range = np.ptp(x)
y_range = np.ptp(y)
y_min = np.min(y)
y_max = np.max(y)
y_mean = np.mean(y)

if model_name == "Exponential":

    parameter_names = ["A", "k", "C"]

    p0 = [
        y[0] - y[-1],
        1 / x_range,
        y[-1]
    ]

    lower_bounds = [-np.inf, 1e-10, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    constrained_index = 1

elif model_name == "Linear":

    parameter_names = ["m", "c"]

    p0 = np.polyfit(x, y, 1)

    lower_bounds = [-np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf]

    constrained_index = None

elif model_name == "Gaussian":

    parameter_names = ["A", "mu", "s", "C"]

    p0 = [
        y_max - y_min,
        x[np.argmax(y)],
        x_range / 6,
        y_min
    ]

    lower_bounds = [-np.inf, -np.inf, 1e-10, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    constrained_index = 2

elif model_name == "Sinusoidal":

    parameter_names = ["A", "w", "phi", "C"]

    p0 = [
        y_range / 2,
        2 * np.pi / x_range,
        0,
        y_mean
    ]

    lower_bounds = [-np.inf, 1e-10, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    constrained_index = 1

ndim = len(parameter_names)

if len(x) <= ndim:
    st.error("You need more data points than model parameters.")
    st.stop()

# --------------------------------------------------
# 4. EXPERIMENTAL DATA PLOT
# --------------------------------------------------

st.subheader("Experimental Dataset")

plt.figure(figsize=(9, 5))

plt.errorbar(
    x, y,
    yerr=sigma,
    fmt="ro",
    markersize=3,
    label="Experimental Data"
)

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.grid(True, alpha=0.3)
plt.legend()

st.pyplot(plt.gcf())
plt.close()

# --------------------------------------------------
# 5. CURVE FITTING
# --------------------------------------------------

st.subheader("Curve Fitting")

try:

    popt, pcov = scipy.optimize.curve_fit(
        model,
        x,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000
    )

except (RuntimeError, ValueError) as error:

    st.error(f"Curve fitting failed: {error}")
    st.stop()

ecov = np.sqrt(np.maximum(np.diag(pcov), 0))

y_fit = model(x, *popt)

# Display fitting results
results = pd.DataFrame({
    "Parameter": parameter_names,
    "Best Fit": popt,
    "Uncertainty": ecov
})

st.dataframe(results, hide_index=True)

# --------------------------------------------------
# 6. FITTED CURVE PLOT
# --------------------------------------------------

x_plot = np.linspace(np.min(x), np.max(x), 500)
y_plot = model(x_plot, *popt)

plt.figure(figsize=(9, 5))

plt.errorbar(
    x, y,
    yerr=sigma,
    fmt="c+",
    markersize=4,
    label="Experimental Data"
)

plt.plot(
    x_plot,
    y_plot,
    "b-",
    linewidth=2,
    label="Best Fit"
)

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.grid(True, alpha=0.3)
plt.legend()

st.pyplot(plt.gcf())
plt.close()

# --------------------------------------------------
# 7. REDUCED CHI-SQUARE
# --------------------------------------------------

chi_sq = np.sum(((y - y_fit) / sigma)**2)

N = len(y)
p = ndim

red_chi_sq = chi_sq / (N - p)

st.subheader("Goodness of Fit")

st.write(f"Chi-square = {chi_sq:.5f}")
st.write(f"Reduced Chi-square = {red_chi_sq:.5f}")

# --------------------------------------------------
# 8. MCMC FUNCTIONS
# --------------------------------------------------

def log_likelihood(theta, x, y, sigma, model):

    y_model = model(x, *theta)

    return -0.5 * np.sum(((y - y_model) / sigma)**2)


def log_prior(theta, model_name):

    if not np.all(np.isfinite(theta)):
        return -np.inf

    if model_name == "Exponential":

        if theta[1] <= 0:
            return -np.inf

    elif model_name == "Gaussian":

        if theta[2] <= 0:
            return -np.inf

    elif model_name == "Sinusoidal":

        if theta[1] <= 0:
            return -np.inf

    return 0.0


def log_posterior(theta, x, y, sigma, model, model_name):

    lp = log_prior(theta, model_name)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, sigma, model)


# --------------------------------------------------
# 9. INITIALIZE MCMC WALKERS
# --------------------------------------------------

st.subheader("MCMC Sampling")

nwalkers = max(32, 2 * ndim + 2)
nsteps = 5000
burnin = 1000

rng = np.random.default_rng(42)

# Use covariance uncertainties to initialize walkers
scale = np.where(
    np.isfinite(ecov) & (ecov > 0),
    ecov,
    np.maximum(np.abs(popt) * 1e-4, 1e-6)
)

pos = popt + 0.1 * scale * rng.standard_normal((nwalkers, ndim))

# Ensure all walkers satisfy the prior
if constrained_index is not None:

    attempts = 0

    while np.any(pos[:, constrained_index] <= 0) and attempts < 1000:

        invalid = pos[:, constrained_index] <= 0

        pos[invalid] = (
            popt +
            0.1 * scale * rng.standard_normal((np.sum(invalid), ndim))
        )

        attempts += 1

    if np.any(pos[:, constrained_index] <= 0):
        st.error("Could not initialize valid MCMC walkers. Check your data and model.")
        st.stop()

# --------------------------------------------------
# 10. RUN MCMC
# --------------------------------------------------

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_posterior,
    args=(x, y, sigma, model, model_name)
)

with st.spinner("Running MCMC sampling..."):

    sampler.run_mcmc(
        pos,
        nsteps,
        progress=False
    )

st.success("MCMC sampling completed.")

# --------------------------------------------------
# 11. ANALYZE MCMC SAMPLES
# --------------------------------------------------

samples = sampler.get_chain(
    discard=burnin,
    flat=True
)

average = np.mean(samples, axis=0)
uncertainty = np.std(samples, axis=0)

mcmc_results = pd.DataFrame({
    "Parameter": parameter_names,
    "Curve Fit": popt,
    "MCMC Mean": average,
    "MCMC Uncertainty": uncertainty
})

st.subheader("MCMC Results")
st.dataframe(mcmc_results, hide_index=True)

# --------------------------------------------------
# 12. AUTOCORRELATION TIME
# --------------------------------------------------

try:

    tau = sampler.get_autocorr_time(discard=burnin)

    if nsteps - burnin > 50 * np.max(tau):

        st.success("Chain length is sufficient according to the 50τ rule of thumb.")

    else:

        st.warning("Chain length may be insufficient according to the 50τ rule of thumb.")

    st.write("Autocorrelation times:", tau)

except emcee.autocorr.AutocorrError:

    st.warning("Chain is too short to estimate autocorrelation time reliably.")

# --------------------------------------------------
# 13. TRACE PLOTS
# --------------------------------------------------

st.subheader("MCMC Trace Plots")

chain = sampler.get_chain()

fig, axes = plt.subplots(
    ndim,
    1,
    figsize=(10, 2.5 * ndim),
    sharex=True
)

if ndim == 1:
    axes = [axes]

for i in range(ndim):

    axes[i].plot(
        chain[:, :, i],
        alpha=0.3
    )

    axes[i].set_ylabel(parameter_names[i])

axes[-1].set_xlabel("MCMC Step")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 14. POSTERIOR DISTRIBUTIONS
# --------------------------------------------------

st.subheader("Posterior Distributions")

for i, name in enumerate(parameter_names):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        samples[:, i],
        bins=40,
        density=True,
        alpha=0.7
    )

    ax.axvline(
        popt[i],
        color="red",
        linestyle="--",
        label="Curve Fit"
    )

    ax.axvline(
        average[i],
        color="blue",
        linestyle="--",
        label="MCMC Mean"
    )

    ax.set_xlabel(name)
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Posterior Distribution of {name}")
    ax.legend()
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# 15. DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("Download Results")

csv_results = mcmc_results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Parameter Results",
    data=csv_results,
    file_name="curve_fitting_mcmc_results.csv",
    mime="text/csv"
)

st.success("Scientific curve fitting and MCMC analysis completed.")
