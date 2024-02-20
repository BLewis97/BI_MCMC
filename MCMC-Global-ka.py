#Import required packages
import jax                                                #numpy on CPU/GPU/TPU
import sys                                                #allows for command line arguments to be run with the script
import arviz as az                                        #for saving Bayesian Data
from funcs_ub_globalfit import *                          #physical recombination models (self written)
import jax.numpy as jnp                                   #jnp regularly called (ease of use)
from jax.random import PRNGKey                            #pseudo-random number generator (ease of use)

import numpyro                                            #Bayesian inference package
from numpyro.infer import MCMC, NUTS, Predictive          #MCMC with NUTS to make it Hamiltonian MC
from numpyro.distributions import TruncatedNormal, Normal #To define prior distributions

eidx = 66 #no. points you eventually want (ish) - look at data and make decision
num_chains = 10                       #number of chains to run (number of cores to use) - might need to run this on linux
numpyro.set_host_device_count(12)     #number of cores on computer

# Define step size - defining x from confocal 
start_step = 1
end_step = 250

len_arr = eidx*100

# Calculate subsampling indices
step_sizes = np.asarray(np.linspace(start_step, end_step, eidx).astype(int)) - 1
subsample_indices = np.cumsum(step_sizes) - step_sizes[0]
subsample_indices = np.minimum(subsample_indices, len_arr-1)
subsample_indices = np.asarray(list(set(subsample_indices)))
subsample_indices.sort()

#Set all floats to 64 bit
jax.config.update("jax_enable_x64", True) #Needed for precision in jax - never touch

ydatas = process_exp_data(bin_amount = 1)

#warm up the JIT
TRPL_HERTZ_LOW(jnp.array([0.0, 1e-15, 1e-18, 1e-3, 0.0, 1e14, 0.0]), 10.0)              #These are from the funcs_ub_globalfit file
TRPL_HERTZ_HIGH(jnp.array([0.0, 1e-15, 1e-18, 1e-3, 0.0, 1e14, 0.0]), 10.0)             #standardises confocal data into numpy arrays
standardise(TRPL_HERTZ_LOW(jnp.array([0.0, 1e-15, 1e-18, 1e-3, 0.0, 1e14, 0.0]), 10.0)) #1 D signal - we tell it time elsewhere

#Bayesian model
def model(dev, ydata = None):
    """
    
    Bayesian model for the BTDP model.

    Parameters
    ----------
    y0: float
        Initial counts in the TRPL signal.

    ydata: array
        standardised log10 of the experimental TRPL signal.

    """

    std_dev = dev

    #std_dev2 = 0.5
    #N0 = numpyro.sample("N0s", TruncatedNormal(low = 1.6, high = 1.9, loc = 1.7, scale = std_dev2))
    N0 = 1.75

    # fac = numpyro.sample(
    #     "fac",
    #     TruncatedNormal(
    #         low   = jnp.array([0.850, 1.950, 2.950, 3.500, 4.000]),
    #         high  = jnp.array([1.250, 2.050, 3.150, 5.900, 6.000]),
    #         loc   = jnp.array([1.000, 2.000, 3.000, 4.000, 5.000]),
    #         scale = jnp.array([0.100, 0.100, 0.100, 0.500, 0.500]),
    #     ),
    # )

    fac = numpyro.sample(
        "fac",
        TruncatedNormal(
            low   = jnp.array([0.950, 1.950, 2.950]),
            high  = jnp.array([1.050, 2.050, 3.150]),
            loc   = jnp.array([1.000, 2.000, 3.000]),
            scale = jnp.array([0.001, 0.001, 0.001]),
        ),
    ) # alans experiment

    theta = numpyro.sample(
        "theta",
        TruncatedNormal(
            low   = jnp.array([-7.00, -2.50, -3.95, -3.00, -4.00, N0 - fac[2] * jnp.log10(4.0)]),
            high  = jnp.array([-4.90,  0.00, -2.00, -1.00, -1.00, N0 - fac[0] * jnp.log10(4.0)]),
            loc   = jnp.array([-5.90, -1.20, -2.30, -2.00, -2.70,  1.00]),
            scale = jnp.array([std_dev, std_dev, std_dev, std_dev, std_dev, std_dev]),
        ),
    ) #in log form, not physically understandable - these units = 1e15 cm-3

    # theta = numpyro.sample(
    #     "theta",
    #     TruncatedNormal(
    #         low   = jnp.array([-7.00, -5.50]),
    #         high  = jnp.array([-4.90, -2.40]),
    #         loc   = jnp.array([-5.90, -4.00]),
    #         scale = jnp.array([std_dev, std_dev]),
    #     ),
    # )
    
    std_dev1 = 0.1
    noise = numpyro.sample(
        "noise",
        TruncatedNormal(
            low   = jnp.array([0.01, 0.01, 0.01, 0.01]),
            high  = jnp.array([0.50, 0.50, 0.50, 0.50]),
            loc   = jnp.array([0.10, 0.10, 0.10, 0.10]),
            scale = jnp.array([std_dev1, std_dev1, std_dev1, std_dev1]),
        ),
    )

    ka    = theta[0]
    kt    = theta[1]
    kb    = theta[2]
    kdt   = theta[3]
    kdp   = theta[4]
    NT    = theta[5]

    # ka = theta[0]
    # kb = theta[1]

    #Calculate the TRPL signal and standardise
    #signal0 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[5] * jnp.log10(4.0)))
    #signal1 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[4] * jnp.log10(4.0)))
    #signal2 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[3] * jnp.log10(4.0)))
    signal3 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0 - fac[2] * jnp.log10(4.0)))
    signal4 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0 - fac[1] * jnp.log10(4.0)))
    signal5 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0 - fac[0] * jnp.log10(4.0)))
    #signal4 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 0.0, 10**kb, 0.0, 0.0, 0.0, 0.0]), 10**(N0 - fac[1] * jnp.log10(4.0)))
    #signal5 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 0.0, 10**kb, 0.0, 0.0, 0.0, 0.0]), 10**(N0 - fac[0] * jnp.log10(4.0)))
    #signal6 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 0.0, 10**kb, 0.0, 0.0, 0.0, 0.0]), 10**(N0))
    signal6 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0))
    
    signal = jnp.stack([signal3, signal4, signal5, signal6])
    signal_s = (signal - means)/stds

    #Define the likelihood
    numpyro.sample("ydata", Normal(signal_s, noise[:, None]), obs=ydata)

num_warmup, num_samples = 5000, 10000 #number of steps you want to do (2x samples than warmup)

rndint      = int(sys.argv[1])         #Seeds for picking of starting points in priors
rndint1     = int(sys.argv[2])         #Seeds picking from generated priors
rndint2     = int(sys.argv[3])         #Picking from posteriors
accept_prob = float(sys.argv[4])
std_dev     = float(sys.argv[5])

key1 = PRNGKey(rndint)            #Generating the random numbers from numbers above
key2 = PRNGKey(rndint1)
key3 = PRNGKey(rndint2)

#Define the MCMC - see Barney's notes for an explanation of each argument
mcmc = MCMC(
        NUTS(model, adapt_step_size=True, max_tree_depth=6, find_heuristic_step_size=False, dense_mass=True,target_accept_prob=accept_prob),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=True,
    )

#Run the MCMC with the simulated data and noise
#ydata = np.stack((*np.log10(ydatas[:3, :60]), *np.log10(ydatas[3:, subsample_indices]))) - np.log10(ydatas.max(1))[:, None]

ydata = np.log10(ydatas[3:, subsample_indices]) - np.log10(ydatas.max(1))[3:, None] # this is because after 6ns it was flat and alan didnt want to fit?
s_ydata, means, stds = standardise(ydata) 
mcmc.run(key1, dev = std_dev, ydata = s_ydata, extra_fields=["num_steps", "energy"]) #runs mcmc

mcmc.print_summary() #prints stats to check how well MCMC has worked

posterior_samples = mcmc.get_samples() #gives all the numbers of posteriors resolved from MCMC
prior_predictions = Predictive(model, num_samples=10000)(key2, dev = std_dev) #makes graph of priors

posterior_predictive = Predictive(model, posterior_samples, num_samples=10000)(key3, dev = std_dev) #makes graph of posteriors


idata = az.from_numpyro(mcmc, prior = prior_predictions,
                    posterior_predictive = posterior_predictive) #save data

accept_prob = int(accept_prob * 100) #acceptance probability - accuracy of integrator
std_dev     = int(std_dev * 100)     #stdev of priors

az.to_netcdf(idata, f"fit_global_10tree_lower_ka_fixed_N0_accept{accept_prob}_std_{std_dev}_{rndint}_1")
