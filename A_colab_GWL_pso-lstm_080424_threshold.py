import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pyswarm import pso

# Set random seeds for reproducibility
np.random.seed(10)

# Load data
data = pd.read_csv('GWL_data_prep.csv')

# Split data into features and target
X = data.drop('Prev_GWL', axis=1)
y = data['Prev_GWL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Define LSTM model creator function
def create_lstm_model(params):
    model = Sequential()
    model.add(LSTM(params[0], input_shape=(1, X_train_scaled.shape[1]), activation=params[1]))
    model.add(Dense(params[2], activation=params[3]))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

# Define fitness function for PSO
def fitness_function(params):
    lstm_model = create_lstm_model(params)
    history = lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=0)
    y_pred = lstm_model.predict(X_test_reshaped)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Define search space for PSO
lb = [16, 'relu', 8, 'relu']  # Lower bounds for parameters
ub = [128, 'tanh', 64, 'tanh']  # Upper bounds for parameters

# Use PSO to find optimal parameters
best_params, _ = pso(fitness_function, lb, ub, swarmsize=10, maxiter=50)

# Train LSTM model with optimized parameters
optimal_lstm_model = create_lstm_model(best_params)
optimal_lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Evaluate model performance
y_pred = optimal_lstm_model.predict(X_test_reshaped)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
cos_sim = cosine_similarity(y_test.values.reshape(1, -1), y_pred.reshape(1, -1))

# Print evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R^2) Score: {r2}')
print(f'Cosine Similarity: {cos_sim[0][0]}')

# Plot actual vs predicted values
plt.figure(figsize=(14, 8))
plt.plot(np.arange(len(y_test)), y_test, marker='.', linestyle='-', color='blue', label='Actual')
plt.plot(np.arange(len(y_test)), y_pred, marker='.', linestyle='-', color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Groundwater Level')
plt.legend()
plt.grid(True)
plt.show()
