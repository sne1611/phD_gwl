import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from dateutil import parser

# Step 1: Data Preprocessing
# Read the dataset from the CSV file
dataset = pd.read_csv('GWL 1993-2021 modified.csv', header=None)
timestamps = dataset.iloc[:, 0].values
groundwater_levels = dataset.iloc[:, 1].values

# Convert timestamps to numeric representation (e.g., month and year)
numeric_timestamps = []
for timestamp in timestamps:
    try:
        date = parser.parse(timestamp, dayfirst=False)
        numeric_timestamps.append(date.year * 12 + date.month)
    except ValueError:
        numeric_timestamps.append(np.nan)
numeric_timestamps = np.array(numeric_timestamps)

# Remove rows with missing values (NaNs)
valid_indices = np.logical_not(np.isnan(numeric_timestamps))
numeric_timestamps = numeric_timestamps[valid_indices]
groundwater_levels = groundwater_levels[valid_indices]

# Scale the groundwater levels to a suitable range (e.g., between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_groundwater_levels = scaler.fit_transform(groundwater_levels.reshape(-1, 1))

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    numeric_timestamps, scaled_groundwater_levels, test_size=0.2, shuffle=False
)

# Step 2: Reshaping the Data
# Reshape the input data to have a 3D shape
X_train = X_train.reshape((X_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, 1))

# Step 3: Model Training
# Set up the CNN model architecture
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(1, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Step 4: Model Evaluation
# Predict groundwater levels for the testing data
y_pred = model.predict(X_test)

# Reverse the scaling of the predicted values to obtain the actual groundwater levels
predicted_groundwater_levels = scaler.inverse_transform(y_pred)

# Evaluate the model's performance using mean squared error (MSE)
mse = np.mean((predicted_groundwater_levels - groundwater_levels[X_train.shape[0]:]) ** 2)
print("Mean Squared Error (MSE):", mse)

# Visualize the predictions
import matplotlib.pyplot as plt

plt.plot(timestamps[X_train.shape[0]:], groundwater_levels[X_train.shape[0]:], label='Actual')
plt.plot(timestamps[X_train.shape[0]:], predicted_groundwater_levels, label='Predicted')
plt.xlabel('Timestamp')
plt.ylabel('Groundwater Level')
plt.title('Groundwater Level Prediction')
plt.legend()
plt.show()
