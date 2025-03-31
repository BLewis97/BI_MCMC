import jax                     #numpy for CPU/GPU/TPU
import diffrax                 #jax-based numerical differential eq solver
import numpy as np
import equinox as eqx          #extension of jax
import jax.numpy as jnp        #jax numpy
import numpy as np

jax.config.update("jax_enable_x64", True)

#Rate equations charge carrier dynamics
def Extended_Model(t, y, args):
    dne_dt, dnt_dt, dnh_dt = y
    ka, kt, kb, kdt, kdp, NT, p0 = args
    
    # Add some numerical safeguards
    y0 = jnp.maximum(y[0], 1e-10)  # Prevent negative/zero values
    y1 = jnp.maximum(y[1], 1e-10)
    y2 = jnp.maximum(y[2], 1e-10)
    
    # Clip very large values
    y0 = jnp.minimum(y0, 1e20)
    y1 = jnp.minimum(y1, 1e20)
    y2 = jnp.minimum(y2, 1e20)
    
    A   = ka * (y0*(y2 + p0)**2 + (y2 + p0)*y0**2)
    B   = kb * y0 * (y2 + p0)
    T   = kt * y0 * (NT - y1)
    DT  = kdt * y1
    DP  = kdp * y1 * (y2 + p0)
    
    dne_dt  = - B - A - T + DT
    dnt_dt  =   T - DP - DT
    dnh_dt  = - B - A - DP
    
    # More robust constraint
    dnt_dt = jnp.where(y1 <= NT, dnt_dt, -jnp.abs(dnt_dt))
    
    return jnp.stack([dne_dt, dnt_dt, dnh_dt])


#JIT compiled function to solve the ODE
@jax.jit #JIT = 'Just in time' - takes python code and translates to computer 1s and 0s
def solve_TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0, NTp, N0h):
    """
    Formerly solve_TRPL_Extended_Model_LOW
    Solve the ODEs for the Extended_Model model.
    Solves for electron concentration in conduction band
    Solves for electron concentration in traps
    Solves for hole concentration in valence band

    Parameters
    ----------
    ka: float
        k_A Auger rate constant (cm^6 ns^-1).

    kt: float
        k_T trapping rate constant (cm^3 ns^-1).

    kb: float
        k_B bimolecular rate constant (cm^3 ns^-1).

    kdt: float
        k_Dt detrapping rate constant (ns^-1) (trap to conduction band).

    kdp: float
        k_Dp depopulation rate constant (cm^3 ns^-1) (trap to valence band).
    
    NT: float
        Trap density (cm^-3).

    p0: float
        Doping density (cm^-3).

    N0: float
        Initial electron concentration (cm^-3).

    NTp: float
        Initial density of carriers in traps (cm^-3).

    N0h: float
        Initial hole concentration (cm^-3).
    
    Returns
    -------
    sol: array
        Solution to the ODEs.

    """

    #Define equations
    terms = diffrax.ODETerm(Extended_Model) #Ordinary Differential Term - Input Model here

    #t = jnp.arange(60, dtype=jnp.float64)/10 commented out as i will put time in as an argument
    #Start and end times
    t0 = t[0] #Originally 0
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([N0, NTp, N0h]) 
    dt0 = t[1]-t[0] #Originally 0.0002 - this may be more robust when feeding in new datasets

    #Define solver and times to save at
    solver = diffrax.Kvaerno5() #Choice of numerical solver
    saveat = diffrax.SaveAt(ts=t) #Defining time values to save at - set to t so all times

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6) #PID controller is used to dynamically adapt step sizes to match a desired error tolerance
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([ka, kt, kb, kdt, kdp, NT, p0]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000000000
    )
    return sol
#%%
#Function to calculate the TRPL signal
@jax.jit
def TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0): 
    """
    
    Calculate the TRPL signal for the BTD model with auger, accumulation included.

    Parameters
    ----------
    r: jnp.array
        
        ka: float
            k_A Auger rate constant (cm^6 ns^-1).

        kt: float
            k_T trapping rate constant (cm^3 ns^-1).
        
        kb: float
            k_B bimolecular rate constant (cm^3 ns^-1).
        
        kdt: float
            k_Dt detrapping rate constant (ns^-1) (trap to conduction band).

        kdp: float
            k_Dp depopulation rate constant (cm^3 ns^-1) (trap to valence band).

        NT: float
            Trap density (cm^-3).

        p0: float
            Doping density (cm^-3).
        
        bkr: float
            Background counts (counts).

    y0: float
        Initial TRPL counts (counts).
    
    N0: float
        Initial electron concentration (cm^-3).
    
    bkr: float
        Background counts (counts).


    Returns
    -------
    sig: array
        TRPL signal.
    
    """

    #Solve ODEs
    sol = solve_TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0, 0.0, N0) #formerly solve_TRPL_Extended_Model_LOW(...)

    # def body_fun(_, val):
    #     return solve_TRPL_Extended_Model(*r, N0 + val.ys[-1, 0], val.ys[-1, 1], N0 + val.ys[-1, 2] - r[-1])

    # sol = jax.lax.fori_loop(0, 10, body_fun, sol)

    #Calculate TRPL signal
    sig = (sol.ys[:, 0] * (sol.ys[:, 2] + p0)) #because the signal is in log form, this is n*p + a background

    #Adjust for background
    #sig = sig - sig[0] #normalisation 
    
    return jnp.log10(sig) #- jnp.log10(sig[0]) #normalisation

#%%
#Standardise the data
@jax.jit
def standardise(x):
    """
    Standardise the data to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    x:  numpy.ndarray
        The data to standardise.

    Returns
    -------
    x:  numpy.ndarray
        The standardised data.
    """
    mean =  jnp.mean(x, keepdims=True)
    std = jnp.std(x, keepdims=True)
    print(mean)
    print(std)
    stan_array = (x - mean) / std
    return stan_array, mean, std
@jax.jit
def normalise(x):
    """
    Standardise the data to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    x:  numpy.ndarray
        The data to standardise.

    Returns
    -------
    x:  numpy.ndarray
        The standardised data.
    """
    return x/x.max(-1, keepdims=True)
#%%
def TRPL_AB(t, n_0, k_A, k_B):
    def AB_rate_equations(t, n, args):
        """ Rate equation of the ABC model
        
        :param n_0: Initial concentration of the free electron
        :param k_A: SRH Rate Constant
        :param k_B: Bimolecular Rate Constant
        :param k_C: Auger Rate Constant""" 
        k_A, k_B = args
        dne_dt = - k_A*n - k_B*n**2
        
        return dne_dt
    #Solve the ordinary differential equations for the free electron concentration    
    #Define equations
    terms = diffrax.ODETerm(AB_rate_equations)

    
    #Start and end times
    t0 = t[0]
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([n_0])
    dt0 = 0.0002

    #Define solver and times to save at
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([k_A, k_B]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    
    #Calculate TRPL Signal
    
    signal = sol.ys**2 
    
    signal = jnp.log10(signal)
    
    return signal

#%%
def TRPL_ABC(t, n_0, k_A, k_B, k_C):
    def ABC_rate_equations(t, n, args):
        """ Rate equation of the ABC model
        
        :param n_0: Initial concentration of the free electron
        :param k_A: SRH Rate Constant
        :param k_B: Bimolecular Rate Constant
        :param k_C: Auger Rate Constant""" 
        k_A, k_B, k_C = args
        dne_dt = - k_A*n - k_B*n**2 - k_C*n**3
        
        return dne_dt
    #Solve the ordinary differential equations for the free electron concentration    
    #Define equations
    terms = diffrax.ODETerm(ABC_rate_equations)

    
    #Start and end times
    t0 = t[0]
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([n_0])
    dt0 = 0.0002

    #Define solver and times to save at
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([k_A, k_B, k_C]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    
    #Calculate TRPL Signal
    
    signal = sol.ys**2 
    
    signal = jnp.log10(signal)
    
    return signal

#%% Add noise

def add_noise(x, noise):
    """
    Add noise to the data.

    Parameters
    ----------
    x: numpy.ndarray
        The data to add noise to.

    noise: float
        The standard deviation of the noise to add.

    Returns
    -------
    x: numpy.ndarray
        The data with noise added.
    """
    return x +- np.random.normal(0.5, noise, x.shape)