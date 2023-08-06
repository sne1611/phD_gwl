import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the dataset from CSV
dataset = pd.read_csv('GWL 1993-2021 modified.csv', header=None, names=['Date', 'GroundWaterLevel'])

# Preprocessing
# Assuming your dataset has a column named 'ground_water_level'
data = dataset['GroundWaterLevel'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Split into training and testing datasets
train_size = int(len(data_normalized) * 0.7)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

# Create input-output pairs for training and testing
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)

window_size = 10  # Adjust the window size according to your data
X_train, Y_train = create_dataset(train_data, window_size)
X_test, Y_test = create_dataset(test_data, window_size)

# LSTM Model Setup
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Model Training
model.fit(X_train, Y_train, epochs=100, batch_size=5)

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
