"""
PROJECT: AI-Accelerated Global Model for Gridded Ion Thrusters (GIT)

DESCRIPTION:
This project aims to develop a machine learning surrogate model for the plasma 
dynamics inside a Gridded Ion Thruster.

PHASES:
1.  Physics Solver (Ground Truth): 
    Replication of the 0D Global Model described in the paper:
    'Global model of a gridded-ion thruster powered by a radiofrequency inductive coil'
    (Chabert et al., DOI: 10.1063/1.4737114).

2.  Dataset Generation: 
    Creation of a synthetic dataset by sweeping input parameters (RF Power, Mass Flow) 
    and simulating steady-state results using the physics solver.

3.  ML Surrogate Model: 
    Training a Neural Network to approximate the system of differential equations, 
    enabling real-time inference.

4.  Dashboard: 
    Development of a Streamlit interface for interactive performance analysis.



PHASE 2:
1) CONSTANTS AND FUNCTIONS (from Phase 1)
2) WORKER FUNCTION (Executed by each CPU core)
3) MAIN BLOCK (Executed once to manage the parallel pool)
"""
# --- 1. CONSTANTS AND FUNCTIONS (from Phase 1)
# This first part introduces just the functions and the constants used in the first phase. 
# There you can find all the explanatory comments
import numpy as np
from scipy.special import jv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import io
kB_eV = 8.617e-5
e = 1.602e-19
Eexc = 11.6
Eiz = 12.127
u = 1.6605e-27
M = 131.29 * u
R = 0.06
L = 0.1
A = np.pi * R**2 + 2 * np.pi * R * L
V = np.pi * R**2 * L
beta_g = 0.3
beta_i = 0.7
Q0 = 1.2e19
Tg0 = 300
Ag = beta_g * np.pi * R**2
sigma_i = 1e-18
me = 9.109e-31
Kel = 1e-13
k = 0.0057
Delta0 = R/2.405 + L/np.pi
Rcoil = 0.7
epsilon_0 = 8.854e-12
f = 13.56e6
c = 3e8
N = 5
omega = 2 * np.pi * f
k0_ = omega / c
s = 0.001
Vgrid = 1000
v_beam = np.sqrt(2*e*Vgrid/M)
Ai = beta_i * np.pi * R**2
def Ic(PRF_, Rind_, Rcoil_):
    Icoil_ = np.sqrt(2 * PRF_ / (Rind_ + Rcoil_))
    return Icoil_
def derivation_JCL_fixed(t, Y, PRF, Q0_):
    n, ng, Tg, Te = Y
    Kex = 1.2921e-13 * np.exp(- Eexc / (kB_eV * Te))
    Kiz1 = 6.73e-15 * np.sqrt(kB_eV * Te) * (3.97 + 0.643 * (kB_eV * Te) - 0.0368 * (kB_eV * Te)**2) * np.exp(- Eiz / (kB_eV * Te))
    Kiz2 = 6.73e-15 * np.sqrt(kB_eV * Te) * (-0.0001031 * (kB_eV * Te)**2 + 6.386 * np.exp(- Eiz / (kB_eV * Te)))
    Kiz = (Kiz1 + Kiz2) / 2
    uB = np.sqrt(kB_eV * Te * e / (M))
    mfp_i = (ng * sigma_i)**-1
    hL = 0.86 * (3 + L/(2*mfp_i))**-0.5
    hR = 0.8 * (4 + R/mfp_i)**-0.5
    Aeff = 2 * hR * np.pi * R * L + 2 * hL * np.pi * R**2
    dn_dt = n * ng * Kiz - n * uB * Aeff / V
    vg = np.sqrt(8 * kB_eV * e * Tg / (np.pi * M))
    Gamma_g = 0.25 * ng * vg
    Aeff1 = 2 * hR * np.pi * R * L + (2 - beta_i) * hL * np.pi * R**2
    dng_dt = Q0_ / V + n * uB * Aeff1 / V - n * ng * Kiz - Gamma_g * Ag / V
    Tions_ = Tg
    vi = np.sqrt(8 * kB_eV * e * Tions_ / (np.pi * M))
    Kin = sigma_i * vi
    dTg_dt = 2 * me/M * (Te - Tg) * n * Kel + M * uB**2 * n * Kin / (6 * kB_eV * e) - 2 * k * A / (3 * kB_eV * e * V * ng) * (Tg - Tg0) / Delta0 - Tg / ng *dng_dt
    nu_m = ng * Kel
    omega_pe = np.sqrt(n * e**2/(me * epsilon_0))
    epsilon_p = 1 - omega_pe**2 / (omega * (omega - 1j * nu_m))
    k_ = k0_ * np.sqrt(epsilon_p)
    argument = R * k_
    Rind = 2*np.pi*N**2 / (L*omega*epsilon_0) * np.real(1j * argument * jv(1, argument) / (epsilon_p * jv(0, argument)))
    Icoil = Ic(PRF, Rind, Rcoil)
    Pabs = Rind * Icoil**2 / (2*V)
    Ploss = Eiz * e * n * ng * Kiz + Eexc * e * n * ng * Kex + 3 * me/M * kB_eV * e * (Te - Tg) * n * ng * Kel + 7 * kB_eV * e * Te * n * uB * Aeff / V
    dTe_dt = 2/3 * (Pabs - Ploss) / (n * kB_eV * e) - Te / n * dn_dt
    return dn_dt, dng_dt, dTg_dt, dTe_dt
def integration_for_ss_values_JCL_fixed(derivation_JCL_fixed, Y0, PRF, Q0_):
    fun_sim = lambda t, y: derivation_JCL_fixed(t, y, PRF, Q0_)
    sol = solve_ivp(fun_sim, [0, 0.03], Y0, method='LSODA', rtol=1e-6, atol=1e-10)
    if not sol.success:
        print(f'Warning! Integration failed at PRF = {PRF:.2e} W (Q = {Q0_:.2e} 1/s)')
        return np.nan, np.nan, np.nan, np.nan
    plasma_number_density = sol.y[0]
    gas_number_density = sol.y[1]
    gas_temperature = sol.y[2]
    electron_temperature = sol.y[3]
    n_ss = plasma_number_density[-1]
    ng_ss = gas_number_density[-1]
    Tg_ss = gas_temperature[-1]
    Te_ss = electron_temperature[-1]
    return n_ss, ng_ss, Tg_ss, Te_ss

import multiprocessing as mp
import pandas as pd
import time
from scipy.stats import qmc


# --- 2. WORKER FUNCTION (Executed by each CPU core) ---
def simulation_worker(params):
    # Unpack the parameters received from the main process: ID, Power (W), Flow (mg/s)
    sim_id, power_val, flow_rate_mg = params
    
    # To start the stopwatch
    start_time = time.time()


    # Convert Mass Flow Rate from mg/s to particles/second (Physical unit conversion)
    Q_val = (flow_rate_mg * 1e-6) / M
    
    # Define Initial Conditions for the ODE solver [n, ng, Tg, Te]
    # These are "guess" values to help the solver start
    Y0_worker = [5e16, 3e19, 300.0, 40000.0]
    
    # Try to solve the physics equations
    try:
        # Call the integration function (defined globally) with the specific Power and Q
        n, ng, Tg, Te = integration_for_ss_values_JCL_fixed(derivation_JCL_fixed, Y0_worker, power_val, Q_val)
    except Exception:
        # If the mathematical solver crashes (e.g., division by zero), return None to signal failure
        return None
    
    # Check if any result is Not a Number (NaN), indicating the simulation didn't converge
    if np.isnan(n) or np.isnan(ng) or np.isnan(Tg) or np.isnan(Te):
        return None
    
    # --- POST-PROCESSING PHYSICS CALCULATIONS ---
    # Calculate Bohm velocity (speed of ions entering the sheath)
    uB = np.sqrt(kB_eV * e * Te / M)
    
    # Calculate Ion Mean Free Path
    mfp = (ng * sigma_i)**-1
    
    # Calculate hL factor (plasma density drop from center to edge)
    hL = 0.86 * (3 + L/(2*mfp))**-0.5
    
    # Calculate Ion Flux (Gamma_i)
    Gamma_i = hL * n * uB
    
    # Calculate Ion Thrust (Ti) = Flux * Mass * Beam Velocity * Grid Area (Ai must be global)
    Ti = Gamma_i * M * v_beam * Ai
    
    # Calculate Neutral Gas Thermal Velocity
    vg = np.sqrt(8 * kB_eV * e * Tg / (np.pi * M))
    
    # Calculate Neutral Flux
    Gamma_g = 0.25 * ng * vg
    
    # Calculate Neutral Thrust (Tn) = Flux * Mass * Velocity * Grid Area (Ag must be global)
    Tn = Gamma_g * M * vg * Ag
    
    # Total Thrust is the sum of Ion and Neutral components
    Total_Thrust = Ti + Tn
    
    # To stop the stopwatch
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Return a dictionary containing inputs and calculated outputs
    return {
            "id": sim_id,
            "Input_Power_W": power_val,
            "Input_Flow_mg_s": flow_rate_mg,
            "Output_Thrust_mN": Total_Thrust * 1000, # Convert N to mN
            "Success": True,
            "Execution_time_s": elapsed_time
        }

# --- 3. MAIN BLOCK (Executed once to manage the parallel pool) ---
if __name__ == "__main__":
    # Configuration: Define sample size and physical ranges
    NUM_SAMPLES = 5000 
    POWER_MIN, POWER_MAX = 50, 2500   
    FLOW_MIN, FLOW_MAX   = 0.5, 40.0

    # Initialize Latin Hypercube Sampler (LHS) for 2 variables (d=2)
    # LHS ensures the design space is covered more efficiently than random sampling
    sampler = qmc.LatinHypercube(d = 2, seed = 42)
    sample = sampler.random(n = NUM_SAMPLES)

    # Scale the samples (originally 0-1) to the real physical ranges (Power and Flow)
    sample = qmc.scale(sample, [POWER_MIN, FLOW_MIN], [POWER_MAX, FLOW_MAX])

    # Create a list of input tuples to distribute to the workers
    inputs = []
    for i in range(NUM_SAMPLES):
        p_val = sample[i, 0] # Power value for sample i
        f_val = sample[i, 1] # Flow value for sample i
        inputs.append((i, p_val, f_val))
    
    # Detect the number of available CPU cores on the machine
    num_cores = mp.cpu_count()
    print(f'Start data generation with {num_cores} cores')

    # Create a Pool of workers and execute simulations in parallel
    # 'pool.map' automatically distributes the 'inputs' list among the cores
    with mp.Pool(processes = num_cores) as pool:
        raw_results = pool.map(simulation_worker, inputs)

    # Filter out failed simulations (where result is None)
    valid_results = [res for res in raw_results if res is not None]
    
    # Create a Pandas DataFrame from the valid results
    df = pd.DataFrame(valid_results)
    
    # Save the dataset to a CSV file (ready for Phase 3: Machine Learning)
    df.to_csv('GlobalModel_Dataset.csv', index = False)


    # Now it is useful to calculate the mean, min and max of the processing time to compare it to the ML Surrogate
    mean_time = df['Execution_time_s'].mean()
    min_time = df['Execution_time_s'].min()
    max_time = df['Execution_time_s'].max()

    print('\n--- PROCESSING TIME ---')
    print(f'The mean processing time is {mean_time:.2f} s')
    print(f'The min processing time is {min_time:.2f} s')
    print(f'The max processing time is {max_time:.2f} s')

    
