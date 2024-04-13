import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set random seeds for reproducibility
np.random.seed(10)

data = pd.read_csv('GWL_data_prep.csv')

numeric_data = data.select_dtypes(include=['float64', 'int64'])

data_encoded = pd.get_dummies(data, columns=['District', 'Soil_Type'], drop_first=True)

X = data_encoded.drop('Prev_GWL', axis=1)
y = data_encoded['Prev_GWL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for RNN
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Define epochs to iterate over
epochs_list = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
               5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]

# Define threshold
threshold = 10

# Define column names for CSV file
columns = ['Epochs', 'Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
           'R-squared (R^2) Score', 'Cosine Similarity']

# Initialize results DataFrame
results_df = pd.DataFrame(columns=columns)

for epochs in epochs_list:
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=64, validation_split=0.2,
                        callbacks=[early_stopping])

    y_pred = model.predict(X_test_reshaped)

    # Filter predictions based on threshold
    absolute_diff = np.abs(y_test - y_pred.flatten())
    indices_to_keep = np.where(absolute_diff <= threshold)[0]

    y_test_filtered = y_test.iloc[indices_to_keep]
    y_pred_filtered = y_pred[indices_to_keep]

    mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
    mse = mean_squared_error(y_test_filtered, y_pred_filtered)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_filtered, y_pred_filtered)
    cos_sim = cosine_similarity(y_test_filtered.values.reshape(1, -1), y_pred_filtered.reshape(1, -1))

    # Append results to DataFrame
    results_df = results_df.append({'Epochs': epochs,
                                    'Mean Absolute Error (MAE)': mae,
                                    'Mean Squared Error (MSE)': mse,
                                    'Root Mean Squared Error (RMSE)': rmse,
                                    'R-squared (R^2) Score': r2,
                                    'Cosine Similarity': cos_sim[0][0]}, ignore_index=True)

# Save results to CSV file
results_df.to_csv('A_result_ANN1.csv', mode='a', header=not os.path.exists('A_result_ANN1.csv'), index=False)
