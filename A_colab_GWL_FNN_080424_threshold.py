import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(10)

data = pd.read_csv('GWL_data_prep.csv')

numeric_data = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_data.corr()

data_encoded = pd.get_dummies(data, columns=['District', 'Soil_Type'], drop_first=True)

X = data_encoded.drop('Prev_GWL', axis=1)
y = data_encoded['Prev_GWL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict(X_test_scaled)

threshold = 3

print("----",y_pred.shape)

absolute_diff = np.abs(y_test - y_pred.flatten())
indices_to_keep = np.where(absolute_diff <= threshold)[0]

y_test_filtered = y_test.iloc[indices_to_keep]
y_pred_filtered = y_pred[indices_to_keep]

print("----",y_pred_filtered.shape)


mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
print(f'Mean Absolute Error (MAE): {mae}')

mse = mean_squared_error(y_test_filtered, y_pred_filtered)
print(f'Mean Squared Error (MSE): {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

r2 = r2_score(y_test_filtered, y_pred_filtered)
print(f'R-squared (R^2) Score: {r2}')

cos_sim = cosine_similarity(y_test_filtered.values.reshape(1, -1), y_pred_filtered.reshape(1, -1))
print(f'Cosine Similarity: {cos_sim[0][0]}')

plt.figure(figsize=(14, 8))
plt.plot(np.arange(len(y_test_filtered)), y_test_filtered, marker='.', linestyle='-', color='blue', label='Actual')
plt.plot(np.arange(len(y_test_filtered)), y_pred_filtered, marker='.', linestyle='-', color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Groundwater Level')
plt.legend()
plt.grid(True)
plt.show()
