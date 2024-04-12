
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os


class GroundwaterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Groundwater Prediction GUI")

        # Number of Epochs
        self.label_epochs = ttk.Label(root, text="Number of Epochs:")
        self.label_epochs.pack()

        self.entry_epochs = ttk.Entry(root)
        self.entry_epochs.pack()

        # Learning Rate
        self.label_learning_rate = ttk.Label(root, text="Learning Rate:")
        self.label_learning_rate.pack()

        self.entry_learning_rate = ttk.Entry(root)
        self.entry_learning_rate.pack()

        # Batch Size
        self.label_batch_size = ttk.Label(root, text="Batch Size:")
        self.label_batch_size.pack()

        self.entry_batch_size = ttk.Entry(root)
        self.entry_batch_size.pack()

        # Dropout Rate
        self.label_dropout_rate = ttk.Label(root, text="Dropout Rate:")
        self.label_dropout_rate.pack()

        self.entry_dropout_rate = ttk.Entry(root)
        self.entry_dropout_rate.pack()

        self.button_predict = ttk.Button(root, text="Predict and Show Results", command=self.predict_and_show)
        self.button_predict.pack()

        self.button_append_csv = ttk.Button(root, text="Append CSV", command=self.append_to_csv)
        self.button_append_csv.pack()

        self.results_text = tk.Text(root, height=10, width=40)
        self.results_text.pack()

    def train_and_predict(self, epochs, learning_rate, batch_size, dropout_rate):
        # Load the data
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'parametric_data.csv')
        data = pd.read_csv(file_path)

        # Select relevant features for prediction
        selected_features = ['Month', 'Year', 'Temperature (degree centigrate)', 'diurnal temp range (degree centigrate)',
                             'Precipitation (mm/month)', 'Pressure (Hpa)', 'Previous GWL']
        data = data[selected_features]

        # Normalize the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Split data into features and target
        X = data_scaled[:, :-1]
        y = data_scaled[:, -1]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshape data for LSTM input (samples, time steps, features)
        time_steps = 1  # Adjust this for time windowing
        X_train = X_train.reshape(X_train.shape[0], time_steps, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], time_steps, X_test.shape[1])

        # Build the advanced LSTM model
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        bi_lstm = Bidirectional(LSTM(100, activation='relu', return_sequences=True))(input_layer)
        attention = Attention()([bi_lstm, bi_lstm])
        concatenated = Concatenate()([bi_lstm, attention])
        dropout = Dropout(dropout_rate)(concatenated)
        lstm_out = LSTM(50, activation='relu')(dropout)
        output_layer = Dense(1)(lstm_out)

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # Set up callbacks for early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                            callbacks=[early_stopping, reduce_lr], verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and actual values
        y_pred_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred.reshape(-1, 1))))[:, -1]
        y_test_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))[:, -1]

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        return epochs, mse, rmse, mae, r2, y_test_inv, y_pred_inv

    def predict_and_show(self):
        epochs = int(self.entry_epochs.get())
        learning_rate = float(self.entry_learning_rate.get())
        batch_size = int(self.entry_batch_size.get())
        dropout_rate = float(self.entry_dropout_rate.get())

        results = self.train_and_predict(epochs, learning_rate, batch_size, dropout_rate)

        results_text = f"Epochs: {results[0]}\nMSE: {results[1]}\nRMSE: {results[2]}\nMAE: {results[3]}\nR2: {results[4]}"
        self.results_text.delete(1.0, tk.END)  # Clear previous results
        self.results_text.insert(tk.END, results_text)

        # Plot predictions vs. actual values
        plt.figure(figsize=(10, 6))
        plt.plot(results[5], label='Actual')
        plt.plot(results[6], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Groundwater Level')
        plt.title('Groundwater Level Prediction using Advanced LSTM')
        plt.legend()
        plt.show()

    def append_to_csv(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if filename:
            learning_rate = float(self.entry_learning_rate.get())
            batch_size = int(self.entry_batch_size.get())
            dropout_rate = float(self.entry_dropout_rate.get())

            results = self.train_and_predict(int(self.entry_epochs.get()), learning_rate, batch_size, dropout_rate)
            results_text = f"Epochs, Learning Rate, Batch Size, Dropout Rate, MSE, RMSE, MAE, R2\n{results[0]},{learning_rate}, {batch_size}, {dropout_rate}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}, {results[7]}"

            # Append results to CSV
            with open(filename, "a") as f:
                f.write(results_text + "\n")

            self.results_text.insert(tk.END, "\n Results appended to CSV.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = GroundwaterPredictionApp(root)
    root.mainloop()