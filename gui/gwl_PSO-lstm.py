import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

# Load the dataset from CSV
dataset = pd.read_csv('GWL 1993-2021 modified.csv', header=None, names=['Date', 'GroundWaterLevel'])

# Preprocessing
# Assuming your dataset has columns 'Date' and 'GroundWaterLevel'
dates = pd.to_datetime(dataset['Date'], format='%b-%y')
data = dataset['GroundWaterLevel'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Create input-output pairs for training and testing
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

# Define LSTM model
def build_lstm_model(hidden_layers, epochs, batch_size, window_size):
    model = Sequential()
    for _ in range(hidden_layers):
        model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Split into training and testing datasets
train_size = int(len(data_normalized) * 0.7)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

# Create input-output pairs for training and testing
window_size = 10  # Adjust the window size according to your data
X_train, Y_train = create_dataset(train_data, window_size)
X_test, Y_test = create_dataset(test_data, window_size)

# Reshape the input data to have three dimensions
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define hyperparameters to search
param_grid = {
    'hidden_layers': [3, 4, 5],
    'epochs': [50, 100, 150],
    'batch_size': [32, 64, 128],
    'window_size': [10, 15, 20]
}

# Perform grid search
best_rmse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    hidden_layers = params['hidden_layers']
    epochs = params['epochs']
    batch_size = params['batch_size']
    window_size = params['window_size']

    # Build and train the LSTM model
    model = build_lstm_model(hidden_layers, epochs, batch_size, window_size)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Model Prediction
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Transform predictions back to the original scale
    train_predictions = scaler.inverse_transform(train_predictions)
    Y_train = scaler.inverse_transform(Y_train)
    test_predictions = scaler.inverse_transform(test_predictions)
    Y_test = scaler.inverse_transform(Y_test)
    
    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_predictions))
    
    # Update best parameters if RMSE improves
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_params = params

# Print the best parameters and RMSE
print('Best Parameters:', best_params)
print('Best RMSE:', best_rmse)

# Build and train the LSTM model with the best parameters
hidden_layers = best_params['hidden_layers']
epochs = best_params['epochs']
batch_size = best_params['batch_size']
window_size = best_params['window_size']

model = build_lstm_model(hidden_layers, epochs, batch_size, window_size)
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# Model Prediction
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Transform predictions back to the original scale
train_predictions = scaler.inverse_transform(train_predictions)
Y_train = scaler.inverse_transform(Y_train)
test_predictions = scaler.inverse_transform(test_predictions)
Y_test = scaler.inverse_transform(Y_test)

# Performance Evaluation
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predictions))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)
