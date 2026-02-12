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



PHASE 4:
1) PAGE CONFIGURATION
2) LOAD ASSETS (Model & Scalers)
3) SAFETY CHECK
4) USER INTERFACE (Inputs)
5) PREDICTION PIPELINE (The "Brain")
6) DISPLAY RESULTS
7) DEBUGGING SECTION

This is the final App. To execute, type in the terminal: streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf 
import joblib

# --- 1. PAGE CONFIGURATION ---
# Sets the browser tab title and centers the layout for a professional look.
st.set_page_config(page_title='GIT AI Surrogate', layout='centered')

# Main title of the web application
st.title('Gridded Ion Thruster: AI Surrogate Model')

# --- 2. LOAD ASSETS (Model & Scalers) ---
# @st.cache_resource is a decorator. 
# It tells Streamlit to run this function only once time and save the result in memory.
# This prevents reloading the heavy AI model every time the user moves a slider.
@st.cache_resource
def load_assets():
    # Load the trained Neural Network (.keras file)
    model = tf.keras.models.load_model('git_surrogate_model.keras')
    # Load the translators for inputs (Watts -> 0-1) and outputs (0-1 -> mN)
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

# --- 3. SAFETY CHECK ---
# We try to load the files. If a file is missing, the app shows a clear error 
# instead of crashing with a "Traceback" code.
try:
    model, scaler_X, scaler_y = load_assets()
    st.success('The system is ready')
except Exception as e:
    st.error(f'Critical error: {e}')
    st.stop() # Stops the execution here if assets are missing

st.divider()

# --- 4. USER INTERFACE (Inputs) ---
st.subheader("Operational Parameters")

# Create two columns to place sliders side-by-side
col_sx, col_dx = st.columns(2)

with col_sx:
    # Slider for Power. Step=5.0 ensures fluid movement with reasonable precision.
    power = st.slider('RF Power (Watt)', 50.0, 2500.0, 1000.0, step=5.0)

with col_dx:
    # Slider for Flow. Step=0.1 is important for small values like 0.5 or 2.3.
    flow = st.slider('Mass Flow (mg/s)', 0.5, 40.0, 2.0, step=0.1)


# --- 5. PREDICTION PIPELINE (The "Brain") ---

# Step A: Format data. 
# The model expects a 2D Matrix (list of lists), not a single number.
input_data = np.array([[power, flow]])

# Step B: Normalize.
# Scaling of the input values
input_scaled = scaler_X.transform(input_data)

# Step C: Predict.
# The model calculates the result based on the training.
# verbose=0 keeps the terminal clean (hides the progress bar).
prediction_scaled = model.predict(input_scaled, verbose=0)

# Step D: Denormalize.
# [0][0] extracts the single number from the matrix.
thrust_mN = scaler_y.inverse_transform(prediction_scaled)[0][0]

# --- 6. DISPLAY RESULTS ---
st.divider()
st.subheader('Simulation Results')

# st.metric displays the value in a large, bold font.
st.metric('Thrust (mN)', f'{thrust_mN:.2f} mN')

# --- 7. DEBUGGING SECTION ---
# This expandable box is hidden by default. 
# It helps the developer check if the inputs are normalized correctly.
with st.expander('See what the ML Surrogate sees (Debug):'):
    st.write('The normalized input value is:', input_scaled)
    st.write('The normalized output value is:', prediction_scaled)


# --- OPTIMIZATION MODE SECTION ---
# This part of the dashboard helps engineers find the best settings for a specific goal.

st.divider() # Adds a visual line to separate this section from the manual sliders above.
st.header('Optimization Mode: Dual Strategy')

# Explanation for the user:
# We explain that the tool offers two solutions: one to save power, one to save fuel.
st.markdown("""
This tool finds the best operating parameters for a specific **Target Thrust**.
It proposes two different strategies:
1.  **Min Power:** Best for satellites with limited solar panels.
2.  **Min Propellant:** Best for long-duration missions to save mass.
Monte Carlo population = 50000
Tolerance = 0.5 mN
""")

# Input field:
# instead of a slider, we use a box where the user types the desired thrust (e.g., 100 mN).
target_thrust = st.number_input('Target Thrust (mN)', min_value=0.0, max_value=255.0, value=100.0, step=5.0)


# --- BUTTON CLICK EVENT ---
# The optimization only runs when the user clicks this button.
if st.button('Find optimized parameters'):
    
    # --- 1. POPULATION GENERATION (Monte Carlo) ---
    # We generate 50,000 random engine configurations to test.
    N_pop = 50000 
    
    # Create random values for Power (50-2500 W) and Flow (0.5-40 mg/s)
    rnd_power = np.random.uniform(50.0, 2500.0, N_pop)
    rnd_flow  = np.random.uniform(0.5, 40.0, N_pop)

    # Combine them into a single matrix (required by the AI model)
    virtual_inputs = np.column_stack((rnd_power, rnd_flow))


    # --- 2. AI PREDICTION ---
    # First, we must normalize the random inputs (scale them to 0-1)
    inputs_scaled = scaler_X.transform(virtual_inputs)
    
    # The AI predicts the thrust for all 50,000 configurations at once
    pred_scaled = model.predict(inputs_scaled, verbose=0)
    
    # Convert the predictions back to real units (mN)
    pred_real = scaler_y.inverse_transform(pred_scaled)


    # --- 3. FILTERING (Finding valid candidates) ---
    # We look for configurations that produce the target thrust within a small error range.
    tolerance = 0.5 # We accept results that are +/- 0.5 mN from the target.
    
    # Create a True/False list: True if the thrust is close to the target
    valid_mask = np.abs(pred_real - target_thrust) < tolerance
    
    # Get the ID numbers (indices) of the valid configurations
    valid_indices = np.where(valid_mask)[0]
    
    
    # --- 4. STRATEGY SELECTION ---
    if len(valid_indices) > 0:
        # Success! We found at least one valid configuration.
        st.success(f'Found {len(valid_indices)} valid configurations for {target_thrust} mN')

        # Extract the Power, Flow, and Thrust for only the valid candidates
        cand_power  = rnd_power[valid_indices]
        cand_flow   = rnd_flow[valid_indices]
        cand_thrust = pred_real[valid_indices]

        # --- STRATEGY A: Minimize Power ---
        # Find the candidate that uses the least amount of Power
        idx_min_power = np.argmin(cand_power)

        opt_A_power  = cand_power[idx_min_power]
        opt_A_flow   = cand_flow[idx_min_power]
        opt_A_thrust = cand_thrust[idx_min_power][0] # [0] extracts the value from the array

        # --- STRATEGY B: Minimize Mass Flow ---
        # Find the candidate that uses the least amount of Propellant
        idx_min_flow = np.argmin(cand_flow)

        opt_B_power  = cand_power[idx_min_flow]
        opt_B_flow   = cand_flow[idx_min_flow]
        opt_B_thrust = cand_thrust[idx_min_flow][0]


        # --- 5. DISPLAY RESULTS ---
        # Create two columns side-by-side
        col_A, col_B = st.columns(2)

        # COLUMN A: Power Saver Results
        with col_A:
            st.info('Option A: Min Power')
            st.write('**Best for limited power supply**')
            # Use metrics to display big, bold numbers
            st.metric('Power', f'{opt_A_power:.1f} W')
            st.metric('Flow',  f'{opt_A_flow:.2f} mg/s')
            st.write('Thrust', f'{opt_A_thrust:.2f} mN')

        # COLUMN B: Fuel Saver Results
        with col_B: # <--- Fixed: This was 'col_A' in your snippet, now it is 'col_B'
            st.info('Option B: Min Mass Flow')
            st.write('**Best for propellant savings**')
            st.metric('Power', f'{opt_B_power:.1f} W')
            st.metric('Flow',  f'{opt_B_flow:.2f} mg/s')
            st.write('Thrust', f'{opt_B_thrust:.2f} mN')

    else:
        # Failure: No configuration produced the desired thrust.
        st.error(f'No configuration found for {target_thrust} mN. Try a more realistic value.')
