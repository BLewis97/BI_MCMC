#Import required packages
import jax                                                #numpy on CPU/GPU/TPU
import sys                                                #allows for command line arguments to be run with the script
import arviz as az                                        #for saving Bayesian Data
from globalfit_functions import *                          #physical recombination models (self written)
import jax.numpy as jnp                                   #jnp regularly called (ease of use)
from jax.random import PRNGKey                            #pseudo-random number generator (ease of use)
import numpyro                                            #Bayesian inference package
from numpyro.infer import MCMC, NUTS, Predictive          #MCMC with NUTS to make it Hamiltonian MC
from numpyro.distributions import TruncatedNormal, Normal #To define prior distributions
from scipy.interpolate import interp1d

num_chains = 10                     #number of chains to run (number of cores to use) - might need to run this on linux
numpyro.set_host_device_count(10)     #number of cores on computer
print(jax.local_device_count(),'Device Cores,' ,num_chains, 'Cores Being Used') #print number of cores being used
#Load Data - format should be a numpy file with 1st row as time, following rows as signal
data = np.load(r'ABC_Simulated.npy') #load the data

time = data[0] #time axis
ydatas = (data[1:])
print('Ydata shape:',ydatas.shape)

#Set all floats to 64 bit
jax.config.update("jax_enable_x64", True) #Needed for precision in jax - never touch


#warm up the JIT
TRPL_ABC(time,0, 1e-15, 1e-18, 1e-3, 0.0, 1e14)              #These are from the funcs_ub_globalfit file - these numbers dont mean anything, 
standardise(TRPL_ABC(time, 0.0, 1e-15, 1e-18, 1e-3, 0.0, 1e14)) #it is literally to get the JIT warmed up with numbers

#Bayesian model
def model(dev, ydata = None):
    """
    
    Bayesian model for the BTDP model.

    Parameters
    ----------
    dev: float
        stdev chosen to be less than the bounds of the truncated guassians of priors (fac and theta, defined below)
    
    y0: float
        Initial counts in the TRPL signal.

    ydata: array
        standardised log10 of the experimental TRPL signal.

    """

    std_dev = dev

    N0 = 2.84

    fac = numpyro.sample(                                 # Creates a truncated normal that we use to create a distribution for the N0s later
        "fac",
        TruncatedNormal(
            low   = jnp.array([0.950, 1.950, 2.950, 3.950, 4.950]),
            high  = jnp.array([1.050, 2.050, 3.150, 4.150, 5.150]),
            loc   = jnp.array([1.000, 2.000, 3.000, 4.000, 5.000]), #changing the number of facs doesnt do anything
            scale = jnp.array([0.001, 0.001, 0.001, 0.001, 0.001]),
        ),
    ) 

    theta = numpyro.sample(                            #Truncated normal distributions of priors in special units. See NOTE for detailed conversion,
        "theta",                                       #but 1 units = 1e15 cm-3. Consider each parameters conversion with this in mind (Auger = cm6s-1)
        TruncatedNormal(
            low   = jnp.array([-8.50, -6.00, -6.00, -6.00, -6.00, -2.00, -3.00]), # changed the fac from 2 to 1
            high  = jnp.array([-5.50, -2.00, -2.00, -2.00, -1.00, 1.00,  -1.00]), # and the log10 from 4 to 10
            loc   = jnp.array([-7.00, -4.00, -4.00, -4.00, -4.00, 0.00, -2.00]),
            scale = jnp.array([std_dev, std_dev, std_dev, std_dev, std_dev, std_dev, std_dev]),
        ),
    ) #in log form, not physically understandable - these units = 1e15 cm-3

    
    std_dev1 = 0.1
    noise = numpyro.sample(
        "noise",
        TruncatedNormal(
            low   = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]),
            high  = jnp.array([0.50, 0.50, 0.50, 0.50, 0.50]),
            loc   = jnp.array([0.10, 0.10, 0.10, 0.10, 0.10]), #this was a large issue - too many noise made a broadcasting error due to the final line of function
            scale = jnp.array([std_dev1, std_dev1, std_dev1, std_dev1, std_dev1]),
        ),
    )
    p_cons = theta[0]
    ka = theta[1]
    kb = theta[2]
    kc = theta[3]
    bkr = theta[4]

    # ka = theta[0]
    # kb = theta[1]

    #Calculate the TRPL signal and standardise
    #signal0 = TRPL_ABC_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[5] * jnp.log10(4.0)))
    #signal1 = TRPL_ABC_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[4] * jnp.log10(4.0)))
    #signal2 = TRPL_ABC_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[3] * jnp.log10(4.0)))
    #signal3 = TRPL_ABC_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0 - fac[2] * jnp.log10(4.0)))
    signal2 = TRPL_ABC(time, 10**(N0- fac[3] * jnp.log10(10.0)),10**p_cons, 10**ka, 10**kb, 10**kc, 10**bkr) #changed factor to 10 from 4
    signal3 = TRPL_ABC(time, 10**(N0- fac[2] * jnp.log10(10.0)),10**p_cons, 10**ka, 10**kb, 10**kc, 10**bkr) #changed to TRPL_ABC from TRPL_ABC_HIGH added time in
    signal4 = TRPL_ABC(time, 10**(N0- fac[1] * jnp.log10(10.0)),10**p_cons, 10**ka, 10**kb, 10**kc, 10**bkr)
    signal5 = TRPL_ABC(time, 10**(N0- fac[0] * jnp.log10(10.0)),10**p_cons, 10**ka, 10**kb, 10**kc, 10**bkr) 
    signal6 = TRPL_ABC(time, 10**(N0),10**p_cons, 10**ka, 10**kb, 10**kc, 10**bkr)
    #signal7 = TRPL_ABC(time,jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0))
    
    signal = jnp.stack([signal6, signal5, signal4, signal3,signal2])
    print('Signal shapes',signal.shape)
    print('Noise shape:',noise.shape)
    signal_s = standardise(signal)[0] #standardise the signal - this is where the incompatible broadcasting came from of [3, 1]
    print('standardised model shape:',signal_s.shape)
    #Define the likelihood
    numpyro.sample("ydata", Normal(signal_s.squeeze(-1), noise[:, None]), obs=ydata)

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
#Adapt step size - goes around gradient of equal probability until you get back to where you started - step size and # steps will never be optimal so adjusts to get granular detail in steps
#
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

#ydata = np.log10(ydatas[3:, subsample_indices]) - np.log10(ydatas.max(1))[3:, None] # this is because after 6ns it was flat and alan didnt want to fit?
s_ydata, means, stds = standardise(np.log10(ydatas))
print('standardised data shape:',s_ydata.shape) 
mcmc.run(key1, dev = std_dev, ydata = s_ydata, extra_fields=["num_steps", "energy"]) #runs mcmc. Betancourt p45

mcmc.print_summary() #prints stats to check how well MCMC has worked

posterior_samples = mcmc.get_samples() #gives all the numbers of posteriors resolved from MCMC
prior_predictions = Predictive(model, num_samples=10000)(key2, dev = std_dev) #makes graph of priors

posterior_predictive = Predictive(model, posterior_samples, num_samples=10000)(key3, dev = std_dev) #makes graph of posteriors


idata = az.from_numpyro(mcmc, prior = prior_predictions,
                    posterior_predictive = posterior_predictive) #save data

accept_prob = int(accept_prob * 100) #acceptance probability - accuracy of integrator
std_dev     = int(std_dev * 100)     #stdev of priors

az.to_netcdf(idata, f"fit_global_10tree_lower_ka_fixed_N0_accept{accept_prob}_std_{std_dev}_{rndint}_1")


# we feed our y data into MCMC, which has initial parameters fed in, which creates a parameter space.
#Each MCMC step calculates a signal, so we see the likelihood that the parameters at that step generate the curve we have fed in