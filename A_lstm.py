import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# Load the CSV file
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, 'GWL 1993-2021 modified _header.csv')
data = pd.read_csv(file_path)

# Extract the groundwater levels (values)
values = data["Value"].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Split the data into training and testing sets
train_size = int(len(scaled_values) * 0.8)
train_data, test_data = scaled_values[:train_size], scaled_values[train_size:]

# Create sequences for training
sequence_length = 12  # You can adjust this value based on your preference
train_sequences = []
for i in range(len(train_data) - sequence_length):
    train_sequences.append(train_data[i : i + sequence_length + 1])

train_sequences = np.array(train_sequences)

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

# Create sequences for testing
test_sequences = []
for i in range(len(test_data) - sequence_length):
    test_sequences.append(test_data[i : i + sequence_length + 1])

test_sequences = np.array(test_sequences)

X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to get original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

train_mse = mean_squared_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

# Calculate Root Mean Squared Error (RMSE)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Calculate Mean Absolute Error (MAE)
train_mae = mean_absolute_error(y_train, train_predict)
test_mae = mean_absolute_error(y_test, test_predict)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

train_r2 = r2_score(y_train, train_predict)
test_r2 = r2_score(y_test, test_predict)
print("Train R-squared:", train_r2)
print("Test R-squared:", test_r2)


# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="True")
plt.plot(test_predict, label="Predicted")
plt.legend()
plt.title("Groundwater Level Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Groundwater Level")
plt.show()
