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



PHASE 3:
1) DATA PREPROCESSING (from Phase 2)
2) CREATION AND TRAINING OF THE NEURAL NETWORK
"""

import pandas as pd  
import numpy as np   
from sklearn.model_selection import train_test_split # Used to split data into training and testing sets
from sklearn.preprocessing import MinMaxScaler # Used to scale numbers between 0 and 1
import joblib # Used to save the scaler files for later use
import tensorflow as tf 
from tensorflow.keras.models import Sequential # To build the network layer by layer
from tensorflow.keras.layers import Dense # To create standard connected layers of neurons
import matplotlib.pyplot as plt 

# --- 1. DATA PREPROCESSING (from Phase 2)
# Function to load the CSV, clean it, and prepare the data
def log_and_preprocess_data(csv_path = 'GlobalModel_Dataset.csv'):
    # Load the data from the CSV file into a pandas DataFrame
    df = pd.read_csv('GlobalModel_Dataset.csv')
    # Print how many lines of data were loaded
    print(f'uploaded data: {len(df)} lines')

    # Remove any rows that contain missing values (NaN) or errors
    df = df.dropna()
    # Print the new number of lines after cleaning
    print(f'clean data: {len(df)} lines')

    # Show the first 5 rows of the data to check if it looks correct
    print(df.head())

    # Select the INPUT columns (Features): Power and Mass Flow
    # .values converts the pandas table into a numpy array (numbers only)
    X = df[['Input_Power_W', 'Input_Flow_mg_s']].values
    
    # Select the OUTPUT column (Target): Thrust
    y = df[['Output_Thrust_mN']].values

    # Print the list of column names for verification
    print(df.columns.tolist())

    # Split the data into two parts:
    # 1. Train set (80%): Used to teach the AI.
    # 2. Test set (20%): Used to test the AI later.
    # random_state=42 ensures the split is the same every time we run the code.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print("\n--- Nomalization (0-1) ---")

    # Create two scalers (tools to squash numbers between 0 and 1)
    # One for the inputs (X) and one for the output (y)
    # (They are defined as 2 'empty boxes'. That is why they are defined the same way)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Apply scaling to the TRAINING data
    # fit_transform: "Learn the min/max from this data, then scale it."
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Apply scaling to the TEST data
    # transform: "Use the min/max you learned before to scale this new data."
    # We do NOT use fit here to avoid cheating (data leakage).
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Save the scalers to files (.pkl)
    # This is crucial: we need them later for the Dashboard to translate user inputs (Phase 4).
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    # Return the four prepared datasets to be used by the neural network
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

# --- 2. CREATION AND TRAINING OF THE NEURAL NETWORK
# Function to build the Neural Network and start training
def build_and_train_model():
    print('\n--- Start of the NN configuration ---')
    
    # Call the previous function to get the clean, scaled data
    X_train, X_test, y_train, y_test = log_and_preprocess_data()
    
    # Create the structure of the Neural Network
    # Sequential means layers are stacked one after another (like an assembly line)
    model = Sequential([
        
        # --- FIRST HIDDEN LAYER ---
        # 64 neurons: 
        #   Think of this as the "resolution". Too few (e.g., 5) would make the prediction "blocky".
        #   64 is enough to draw smooth curves without making the model too heavy.
        # Activation 'relu' (Rectified Linear Unit): 
        #   ReLU allows the network to bend and twist its predictions to fit complex data and it is more
        #   convenient respect to the linear one (we are working just positive values)
        # input_shape=(2,): 
        #   We have exactly 2 input valves: Power and Mass Flow.
        Dense(64, activation = 'relu', input_shape = (2,)),
        
        # --- SECOND HIDDEN LAYER ---
        # Why a second layer?
        #   One layer understands the inputs separately. 
        #   This second layer combines them to understand "interactions" (e.g., how Power behaves when Flow is low).
        #   It is generally a good trade-off between complexity and accuracy
        Dense(64, activation = 'relu'),
        
        # --- OUTPUT LAYER ---
        # 1 neuron: 
        #   We only want one single answer output: the Thrust.
        # No activation (Linear):
        #   We want the raw number (e.g., 150.5 mN).
        #   We do NOT use 'sigmoid' or 'relu' here because we don't want to limit or cut off the result.
        Dense(1)
    ])
    
    # Compile the model (prepare it for training)
    # - optimizer='adam': The algorithm that adjusts the learning rate according to the situation
    # - loss='mse': Mean Squared Error (the metric we want to minimize considering the nature of our problem is
    # one over an infinite number of values)
    # - metrics=['mae']: Also track Mean Absolute Error for readability
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    
    # Print a summary table of the network structure
    model.summary()
    
    print('\n--- Start of the training phase ---')
    
    # Start the actual training loop
    # - epochs=100: Go through the data 100 times
    # - batch_size=32: Processing the entire dataset at once is computationally expensive.
    #   Instead, we divide the data into smaller groups called "batches" of 32 samples.
    #   The model updates its weights after analyzing each batch, rather than waiting for the end.
    #   This value is a standard trade-off: it ensures stable learning without overloading the memory.
    # - validation_split=0.2: Use 20% of training data to self-check during learning (standard choice).
    # - verbose=1: Controls the output display. 
    #   Setting it to 1 shows a progress bar for each epoch with real-time loss updates.
    history = model.fit(X_train, y_train, epochs = 100,
    batch_size = 32, 
    validation_split = 0.2,
    verbose = 1)
    
    print('\n--- Validation on the test set ---')
    
    # Test the model on the Test Set (data it has never seen before)
    loss, mae = model.evaluate (X_test, y_test)
    
    # Save the trained brain to a file so we can reuse it without retraining
    model.save('git_surrogate_model.keras')
    print('the model has been saved with the name \'git_surrogate_model.keras\'')

    # Create a plot to visualize the learning process
    plt.figure()
    # Plot the Training Error (Blue line)
    plt.plot(history.history['loss'], label = 'training error')
    # Plot the Validation Error (Orange line)
    plt.plot(history.history['val_loss'], label = 'validation error')
    plt.title('Learning Curve (Loss)')
    plt.xlabel('epochs')       # X-axis: Time (number of passes)
    plt.ylabel('error (MSE)')  # Y-axis: How wrong the model is
    plt.legend()               
    plt.grid(True)            
    plt.show()                 

# Standard Python check: if this file is run directly, execute the function
if __name__ == "__main__":
    build_and_train_model()

    

