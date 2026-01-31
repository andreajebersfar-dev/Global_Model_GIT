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