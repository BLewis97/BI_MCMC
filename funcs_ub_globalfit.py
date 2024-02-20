import jax                     #numpy for CPU/GPU/TPU
import diffrax                 #jax-based numerical differential eq solver
import numpy as np
import equinox as eqx          #extension of jax
import jax.numpy as jnp        #jax numpy

jax.config.update("jax_enable_x64", True)

#Rate equations charge carrier dynamics
def HERTZ(t, y, args):
    """

    Rate equations for a mixture of the models as per 10.1002/adfm.202004312
    p. 2004312 (5 of 12) and 10.1039/d0cp04950f p. 28348.

    Model allows for a non-constant density of traps with depopulation from these
    traps to the valence band and detrapping to the conduction band, included Auger.

    Background doping included.

    Returns
    -------
    [f0, f1, f2]: array
        Array of the rate equations. Dynamics for electron, trap and hole
        concentraions.
    
    """
    dne_dt, dnt_dt, dnh_dt = y #Populations of conduction band electrons, trapped electrons and valence band holes
    ka, kt, kb, kdt, kdp, NT, p0 = args #Input parameters
    # Auger term
    A   = ka * (y[0]*(y[2] + p0)**2 + (y[2] + p0)*y[0]**2) #Auger Recombination term
    B   = kb * y[0] * (y[2] + p0)                     # Bimolecular recombination term
    T   = kt * y[0] * (NT - y[1])                     # Trapping term
    DT  = kdt * y[1]                                  # Detrapping term
    DP  = kdp * y[1] * (y[2] + p0)                    # Depopulation term
    dne_dt  = - B - A - T + DT                        # Change in electron concentration
    dnt_dt  =   T - DP - DT                           # Change in trapped electron concentration
    dnh_dt  = - B - A - DP                            # Change in hole concentration
    
    # Enforce the condition nt <= NT
    dnt_dt = jnp.where(y[1] <= NT, dnt_dt, 0)
    
    return jnp.stack([dne_dt, dnt_dt, dnh_dt])


#JIT compiled function to solve the ODE
@jax.jit #JIT = 'Just in time' - takes python code and translates to computer 1s and 0s
def solve_TRPL_HERTZ_LOW(ka, kt, kb, kdt, kdp, NT, p0, N0, NTp, N0h):
    """
    
    Solve the ODEs for the Hertz model.

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
    terms = diffrax.ODETerm(HERTZ)

    t = jnp.arange(60, dtype=jnp.float64)/10
    #Start and end times
    t0 = 0.0
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([N0, NTp, N0h])
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
        args = jnp.array([ka, kt, kb, kdt, kdp, NT, p0]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

#JIT compiled function to solve the ODE
@jax.jit
def solve_TRPL_HERTZ_HIGH(ka, kt, kb, kdt, kdp, NT, p0, N0, NTp, N0h):
    """
    
    Solve the ODEs for the Hertz model.

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

    eidx = 66

    # Define step size
    start_step = 1
    end_step = 250

    len_arr = eidx*100

    # Calculate subsampling indices
    step_sizes = np.asarray(np.linspace(start_step, end_step, eidx).astype(int)) - 1
    subsample_indices = np.cumsum(step_sizes) - step_sizes[0]
    subsample_indices = np.minimum(subsample_indices, len_arr-1)
    subsample_indices = np.asarray(list(set(subsample_indices)))
    subsample_indices.sort()

    #Define equations
    terms = diffrax.ODETerm(HERTZ)

    t = jnp.arange(32000, dtype=jnp.float64)[subsample_indices]/10
    #Start and end times
    t0 = 0.0
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([N0, NTp, N0h])
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
        args = jnp.array([ka, kt, kb, kdt, kdp, NT, p0]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

#Function to calculate the TRPL signal
@jax.jit
def TRPL_HERTZ_LOW(r, N0):
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
    sol = solve_TRPL_HERTZ_LOW(*r, N0, 0.0, N0)

    # def body_fun(_, val):
    #     return solve_TRPL_HERTZ(*r, N0 + val.ys[-1, 0], val.ys[-1, 1], N0 + val.ys[-1, 2] - r[-1])

    # sol = jax.lax.fori_loop(0, 10, body_fun, sol)

    #Calculate TRPL signal
    sig = jnp.log10(jnp.clip(sol.ys[:, 0], a_min = 0.0)) + jnp.log10(jnp.clip(sol.ys[:, 2] + r[-1], a_min = 0.0))

    #Adjust for background
    sig = sig - sig[0] #normalisation 
    
    return sig

#Function to calculate the TRPL signal
@jax.jit
def TRPL_HERTZ_HIGH(r, N0):
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
    sol = solve_TRPL_HERTZ_HIGH(*r, N0, 0.0, N0)

    # def body_fun(_, val):
    #     return solve_TRPL_HERTZ(*r, N0 + val.ys[-1, 0], val.ys[-1, 1], N0 + val.ys[-1, 2] - r[-1])

    # sol = jax.lax.fori_loop(0, 10, body_fun, sol)

    #Calculate TRPL signal
    sig = jnp.log10(jnp.clip(sol.ys[:, 0], a_min = 0.0)) + jnp.log10(jnp.clip(sol.ys[:, 2] + r[-1], a_min = 0.0))
    #n*p+background
    #Adjust for background
    sig = sig - sig[0]
    
    return sig

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
    mean =  jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)

    return ((x - mean) / std), mean, std

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

def process_exp_data(bin_amount = 100, final_time = 32000):
    """
    Prepare the experimental data for fitting.


    Parameters
    ----------
    bin_amount : int, optional
        The amount of bins to combine. The default is 1000.

    final_time : int, optional
        The final time to use. The default is 32000.

    Returns
    -------
    xdata : numpy.ndarray
        The time axis for the data.
    
    ydata : numpy.ndarray
        The data with the zero set to the max of the signal.
    """
    #Load data and adjust for OD filters
    data = np.load('tcspc.npy')

    xdata, ydata = data[..., 0], data[..., 1]
    #Find the max of the signal and shift the data so that the max is at the end
    args = ydata.argmax(1)
    ydata = np.asarray([np.roll(arr, -arg) for arr, arg in zip(ydata, args)])
    for i in range(len(ydata)):
        ydata[i][-args[0]:] = ydata[i][-args[0]*2:-args[0]]

    #Bin the data
    ydata = ydata[:, :final_time].reshape(7, final_time//bin_amount, bin_amount).sum(-1)
    ydata = ydata + 1 #Adjust the data to make sure no 0 or negative values that will cause np.log10 problems

    return jnp.asarray(ydata)
