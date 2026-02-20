import numpy as np
import time
import tensorflow as tf
import joblib

print('start ...')

model = tf.keras.models.load_model('git_surrogate_model.keras')
scaler_X = joblib.load('scaler_X.pkl')


# let's choose a arbitrary input
single_arbitrary_input = np.array([[1000.0, 15.0]])
scaled_arbitrary_input = scaler_X.transform(single_arbitrary_input)

# warm up of the system

_ = model(scaled_arbitrary_input, training = False).numpy()

# benchmark (5000 iterations)
N_Test = 5000
time_array = []

for i in range(N_Test):
    start_time_instance = time.time()
    x = model(scaled_arbitrary_input, training = False).numpy()
    end_time_instance = time.time()
    time_instance = (end_time_instance - start_time_instance) * 1000 # in ms
    time_array.append(time_instance)

mean_time_ML_Surrogate = np.mean(time_array)
print(f'the mean latency for the ML Surrogate is {mean_time_ML_Surrogate:.2f} ms')