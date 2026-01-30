import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf 
import joblib

st.set_page_config(page_title = 'GIT AI Surrogate', layout = 'centered')
st.title('Gridded Ion Thruster: AI Surrogate Model')

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('git_surrogate_model.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_assets()
    st.success('The system is ready')
except Exception as e:
    st.error(f'Critical error {e}')
    st.stop()

st.divider()

col_sx, col_dx = st.columns(2)
with col_sx:
    power = st.slider('RF Power (Watt)', 50.0, 2500.0, 1000.0, step = 5.0)
with col_dx:
    flow = st.slider('Mass Flow (mg/s)', 0.5, 40.0, 2.0, step = 0.1)



input_data = np.array([[power, flow]])

input_scaled = scaler_X.transform(input_data)

prediction_scaled = model.predict(input_scaled, verbose = 0)

thrust_mN = scaler_y.inverse_transform(prediction_scaled)[0][0]

st.divider()

st.subheader('Simulation Results')

st.metric('Thrust (mN)',f'{thrust_mN:.2f} mN')

with st.expander('See what the ML Surrogate sees:'):
    st.write('The normalized input value is ', input_scaled)
    st.write('The normalized output value is ', prediction_scaled)
