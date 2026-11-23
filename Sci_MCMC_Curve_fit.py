import numpy as np
import matplotlib.pyplot as plt
import scipy
import emcee
import streamlit as st
st.title("Scientific Curve Fitting and MCMC")
st.write("Exponential model: y = A exp(-kt) + C")
ED=np.loadtxt("ED.csv",delimiter=",", skiprows=1, unpack=True)
t,y,sigma=ED
#print(ED)
#ploting the real dataset
plt.figure("Real Dataset Plot")
plt.plot(t,y,"ro", markersize=1, label="Experimental Data")
plt.ylabel("y")
plt.xlabel("Time")
plt.axhline(y=0,linewidth=1,color="b")
plt.axvline(x=0,linewidth=1,color="b")
plt.grid(True,alpha=0.3)
plt.legend()
st.pyplot(plt.gcf())
plt.close()
#Curve Fitting by scipy.optimize.curve_fit
# Model: y=A*exp(-k*t)+C
# A is the amplitude, k is the decay constant, and C is the baseline.
# Initial value at t=0 is A+C.
def model(t,A,k,C):
    return A*np.exp(-k*t)+C
popt,pcov=scipy.optimize.curve_fit(model,t,y,sigma=sigma, absolute_sigma=True)
#Fitted values come from popt
A_f,k_f,C_f=popt
#gives diagonal elements telling parameters error estimation
ecov=np.sqrt(np.diag(pcov))
st.subheader("Curve Fitting Results")
st.write(f"A = {A_f:.5f}")
st.write(f"k = {k_f:.5f}")
st.write(f"C = {C_f:.5f}")
st.write(f"Parameter uncertainties = {ecov}")
#values  of y as per fitted parameters of model
y_f=model(t,A_f,k_f,C_f)
#ploting the fitting vs the errorbar
plt.figure("Fitted Curve", figsize=(10,6)) #Wider and taler fig
plt.errorbar(t,y,yerr=sigma,fmt="c+", markersize=3, label="Experimental Data")
plt.plot(t,y_f,"b-",linewidth=1,label="Data Fit")
plt.ylabel("y")
plt.xlabel("Time")
plt.axhline(y=0,linewidth=1,color="b")
plt.axvline(x=0,linewidth=1,color="b")
plt.grid(True,alpha=0.3)
plt.legend()
st.pyplot(plt.gcf())
plt.close()
#Checking the fit is how much correct by reduced chi_sqaure method which has formula first for chi square and then for reduced
chi_sq=np.sum(((y-y_f)/sigma)**2)
N=len(y)
p=3
red_chi_sq=chi_sq/(N-p)
st.write(f"Reduced Chi Square = {red_chi_sq:.5f}")
# Near 1: residuals are reasonably consistent with assumed uncertainties.
# Much greater than 1: residuals are larger than expected.
# Much less than 1: uncertainties may be overestimated.
# MCMC stated here with emcee library
# we have to make three log functions, 1. log_likelihood, 2. log_prior, 3. log_posterior
def log_likelihood(theta, t,y,sigma):
    A,k,C=theta
    y_m=model(t,A,k,C)
    return -0.5*np.sum(((y-y_m)/sigma)**2) # same formula as of guassian but with log and ingnoring the lnsqrt(2pi)*sigma terms because it will not affect our likelihood
def log_prior(theta): #only theta as we need to check theta the parameters A ,k, C
    A,k,C=theta
    if k>0:
        return 0.0
    else:
        return -np.inf
def log_posterior(theta,t,y,sigma):
    lp=log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return log_likelihood(theta,t,y,sigma)+lp
# Now running walkers
ndim=3
nwalkers=32 #usually twice as the dimensions are taken but taking 32 will provide large ensemble of walkers
#initializing the positions
pos=popt+0.01*ecov*np.random.randn(nwalkers,ndim) # the position is sum of our popt we have found in curve_fit, ecov are the uncertainties we found for parameter
#0.01 is 1% of the uncertainties used and np.random.randn will generate an array of 32 rows and 3 columns to have our 32 walkers
while not np.all(pos[:,1]>0):
    pos = popt + 0.01 * ecov * np.random.randn(nwalkers, ndim)
#we will use emcee.EnsembleSampler class to construct an object called sampler to perform different operations on it
sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior,args= (t,y,sigma))
# we used log_posterior to evaluate the posterior and args are the experimental data
#Running MCMC chain
nsteps=5000
sampler.run_mcmc(pos,nsteps,progress=True)
# Burning in the intial samples becaue they may influence by their starting position
burnin=1000
samples=sampler.get_chain(discard=burnin, flat=True) # method .get_chain will call in the recorded positions and Flattening to get (samples,3) means no. of samples having parameters 3 a 2D array
#Analyzing the samples
average = np.mean(samples, axis=0)
uncertainty = np.std(samples, axis=0)

st.subheader("MCMC Results")

st.write(f"Curve Fit A = {A_f:.5f} | MCMC A = {average[0]:.5f}")
st.write(f"Curve Fit k = {k_f:.5f} | MCMC k = {average[1]:.5f}")
st.write(f"Curve Fit C = {C_f:.5f} | MCMC C = {average[2]:.5f}")

st.write(f"MCMC uncertainties (A, k, C): {uncertainty}")
tau=sampler.get_autocorr_time(discard=burnin)
# Autocorrelation time estimates how many steps are needed for samples
# to become approximately independent.
if nsteps-burnin > 50*np.max(tau):
    st.success("Your chain length is fine as per 50τ rule of thumb")
else:
    st.warning("Your chain length is short")
chain=sampler.get_chain()
# Plotting A, k, C to inspect chain behaviour and mixing.
# Trace plots alone cannot prove convergence.
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(chain[:,:,0],alpha=0.3)
plt.ylabel("A")
plt.subplot(3,1,2)
plt.plot(chain[:,:,1],alpha=0.3)
plt.ylabel("k")
plt.subplot(3,1,3)
plt.plot(chain[:,:,2],alpha=0.3)
plt.ylabel("C")
plt.tight_layout() # auto adjusting the spaces between plots, labels and axis
st.pyplot(plt.gcf())
plt.close()
# Plotting posterior histograms to examine the distributions of A, k and C.

# Posterior distribution of A, comparing curve-fit and MCMC estimates.
plt.figure(figsize=(8,8))
plt.hist(samples[:,0], bins=40, density=True)
plt.axvline(A_f, color="r",linestyle="--", label="Curve Fit")
plt.axvline(average[0], color="b",linestyle="--",label="MCMC Fit")
plt.ylabel("Probability Density")
plt.xlabel("A")
plt.title("Posterior Distribution of A and Comparison of Curve and MCMC Fit")
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# Posterior distribution of k, comparing curve-fit and MCMC estimates.
plt.figure(figsize=(8,8))
plt.hist(samples[:,1], bins=40, density=True)
plt.axvline(k_f, color="r",linestyle="--", label="Curve Fit")
plt.axvline(average[1], color="b",linestyle="--",label="MCMC Fit")
plt.ylabel("Probability Density")
plt.xlabel("k")
plt.title("Posterior Distribution of k and Comparison of Curve and MCMC Fit")
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# Posterior distribution of C, comparing curve-fit and MCMC estimates.
plt.figure(figsize=(8,8))
plt.hist(samples[:,2], bins=40, density=True)
plt.axvline(C_f, color="r",linestyle="--", label="Curve Fit")
plt.axvline(average[2], color="b",linestyle="--",label="MCMC Fit")
plt.ylabel("Probability Density")
plt.xlabel("C")
plt.title("Posterior Distribution of C and Comparison of Curve and MCMC Fit")
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()
