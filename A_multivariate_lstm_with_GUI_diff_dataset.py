import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

        self.label_epochs = ttk.Label(root, text="Number of Epochs:")
        self.label_epochs.pack()

        self.entry_epochs = ttk.Entry(root)
        self.entry_epochs.pack()

        self.button_set_parameters = ttk.Button(root, text="Set Parameters", command=self.open_parameter_window)
        self.button_set_parameters.pack()

        self.button_predict = ttk.Button(root, text="Predict and Show Results", command=self.predict_and_show)
        self.button_predict.pack()

        self.button_append_csv = ttk.Button(root, text="Append Result to CSV", command=self.append_to_csv)
        self.button_append_csv.pack()

        self.results_text = tk.Text(root, height=10, width=40)
        self.results_text.pack()

        self.parameter_window = None
        self.entry_units = None
        self.combo_activation_parameter = None
        self.entry_dropout = None
        self.hidden_layers = None

    def open_parameter_window(self):
        self.parameter_window = tk.Toplevel(self.root)
        self.parameter_window.title("Set Parameters")

        self.label_activation_parameter = ttk.Label(self.parameter_window, text="Activation Function:")
        self.label_activation_parameter.pack()

        self.combo_activation_parameter = ttk.Combobox(self.parameter_window, values=["relu", "tanh", "sigmoid"])
        self.combo_activation_parameter.pack()

        self.label_lstm_units_parameter = ttk.Label(self.parameter_window, text="Number of LSTM Units:")
        self.label_lstm_units_parameter.pack()

        self.combo_lstm_units_parameter = ttk.Combobox(self.parameter_window, values=["32", "64", "128", "256"])
        self.combo_lstm_units_parameter.pack()

        self.label_hidden_layers_parameter = ttk.Label(self.parameter_window, text="Number of Hidden Layers:")
        self.label_hidden_layers_parameter.pack()

        self.combo_hidden_layers_parameter = ttk.Combobox(self.parameter_window, values=["0", "1", "2", "3"])
        self.combo_hidden_layers_parameter.pack()

        self.label_dropout_parameter = ttk.Label(self.parameter_window, text="Dropout Rate:")
        self.label_dropout_parameter.pack()

        self.combo_dropout_parameter = ttk.Combobox(self.parameter_window, values=["0.0", "0.2", "0.5", "0.7"])
        self.combo_dropout_parameter.pack()

        self.button_save_parameters = ttk.Button(self.parameter_window, text="Save Parameters", command=self.save_parameters)
        self.button_save_parameters.pack()

    def save_parameters(self):
        self.lstm_units = int(self.combo_lstm_units_parameter.get())
        self.activation_function = self.combo_activation_parameter.get()
        self.dropout_rate = float(self.combo_dropout_parameter.get())
        self.hidden_layers = int(self.combo_hidden_layers_parameter.get())
        self.parameter_window.destroy()

    def train_and_predict(self, epochs):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'HEART_PLOT.csv')
        data = pd.read_csv(file_path)

        selected_features = ['age','sex','chestpain','bloodpressure','cholestoral','bloodsugar','ECG','Heartrate','Outcome']
        data = data[selected_features]

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X = data_scaled[:, :-1]
        y = data_scaled[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        time_steps = 1
        X_train = X_train.reshape(X_train.shape[0], time_steps, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], time_steps, X_test.shape[1])

        model = Sequential()
        model.add(LSTM(self.lstm_units, activation=self.activation_function, input_shape=(X_train.shape[1], X_train.shape[2])))

        for _ in range(self.hidden_layers):
            model.add(Dense(128, activation='relu'))

        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_test, y_test),
                            callbacks=[early_stopping, reduce_lr], verbose=1)

        y_pred = model.predict(X_test)

        y_pred_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred.reshape(-1, 1))))[:, -1]
        y_test_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))[:, -1]

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        # Calculate accuracy with a tolerance of ±5 units
        tolerance = 0.75
        accuracy = self.calculate_accuracy(y_test_inv, y_pred_inv, tolerance)

        return epochs, mse, rmse, mae, r2, accuracy, y_test_inv, y_pred_inv

    def calculate_accuracy(self, y_true, y_pred, tolerance):
        correct = 0
        total = len(y_true)
        for true, pred in zip(y_true, y_pred):
            if abs(true - pred) <= tolerance:
                correct += 1
        return (correct / total) * 100

    def predict_and_show(self):
        epochs = int(self.entry_epochs.get())

        if self.lstm_units is None or self.activation_function is None or self.dropout_rate is None:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please set parameters first.")
            return

        results = self.train_and_predict(epochs)

        results_text = f"Epochs: {results[0]}\nMSE: {results[1]}\nRMSE: {results[2]}\nMAE: {results[3]}\nR2: {results[4]}\nAccuracy : {results[5]:.2f}%"
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)

        plt.figure(figsize=(10, 6))
        plt.plot(results[6], label='Actual')
        plt.plot(results[7], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Groundwater Level')
        plt.title('Groundwater Level Prediction using LSTM')
        plt.legend()
        plt.show()

    def append_to_csv(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if filename:
            results = self.train_and_predict(int(self.entry_epochs.get()))
            results_text = f"{results[0]},{self.activation_function},{self.lstm_units},{self.hidden_layers},{self.dropout_rate},{results[1]},{results[2]},{results[3]},{results[4]},{results[5]}\n"

            if not os.path.exists(filename):
                header = "Epochs,Activation Function,Number of LSTM Units,Number of Hidden Layers,Dropout Rate,MSE,RMSE,MAE,R2,Accuracy\n"
                with open(filename, "w") as f:
                    f.write(header)

            with open(filename, "a") as f:
                f.write(results_text)

            self.results_text.insert(tk.END, "Results appended to CSV.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GroundwaterPredictionApp(root)
    root.mainloop()
