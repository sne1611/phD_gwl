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

        # self.label_epochs = ttk.Label(root, text="Number of Epochs:")
        # self.label_epochs.pack()

        # self.entry_epochs = ttk.Entry(root)
        # self.entry_epochs.pack()

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
        # self.activation_function = None
        self.entry_dropout = None
        self.hidden_layers = None  # Initialize the number of hidden layers parameter

    def open_parameter_window(self):
        self.parameter_window = tk.Toplevel(self.root)
        self.parameter_window.title("Set Parameters")

        self.label_activation_parameter = ttk.Label(self.parameter_window, text="Activation Function:")
        self.label_activation_parameter.pack()

        self.combo_activation_parameter = ttk.Combobox(self.parameter_window, values=["relu", "tanh", "sigmoid"])
        self.combo_activation_parameter.pack()

        self.label_lstm_units_parameter = ttk.Label(self.parameter_window, text="Number of LSTM Units:")
        self.label_lstm_units_parameter.pack()

        self.combo_lstm_units_parameter = ttk.Combobox(self.parameter_window, values=["32", "64", "128", "256"])  # Example values, adjust as needed
        self.combo_lstm_units_parameter.pack()

        self.label_hidden_layers_parameter = ttk.Label(self.parameter_window, text="Number of Hidden Layers:")
        self.label_hidden_layers_parameter.pack()

        self.combo_hidden_layers_parameter = ttk.Combobox(self.parameter_window, values=["0", "1", "2", "3"])  # Example values, adjust as needed
        self.combo_hidden_layers_parameter.pack()

        self.label_dropout_parameter = ttk.Label(self.parameter_window, text="Dropout Rate:")
        self.label_dropout_parameter.pack()

        self.combo_dropout_parameter = ttk.Combobox(self.parameter_window, values=["0.0", "0.2", "0.5", "0.7"])  # Example values, adjust as needed
        self.combo_dropout_parameter.pack()

        
        self.button_save_parameters = ttk.Button(self.parameter_window, text="Save Parameters", command=self.save_parameters)
        self.button_save_parameters.pack()

    def save_parameters(self):
        self.lstm_units = int(self.combo_lstm_units_parameter.get())
        self.activation_function = self.combo_activation_parameter.get()  # Update the activation function attribute
        self.dropout_rate = float(self.combo_dropout_parameter.get())
        self.hidden_layers = int(self.combo_hidden_layers_parameter.get())
        self.parameter_window.destroy()
        # self.parameter_window.destroy()


    def train_and_predict(self):
        epoch = [50,100,150,200,250,300,350,400,450,500,550,600,	650	,700,	750,	800,	850,	900,	950,	1000,	1100	,1200,	1300,	1400	,1500,	1600	,1700	,1800,	1900	,2000	,2100,	2200,	2300,	2400	,2500	,2600,	2700	,2800	,2900	,3000	,3500	,3600	,3700,	3800	,3900	,4000,	4100,	4200	,4300	,4400,	4500,	4600,	4700,	4800,	4900	,5000	,5500,	6000,	6500	,7000]
        for x in range(len(epoch)):
            epochs = epoch[x]
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


            # Build the LSTM model with specified parameters
            model = Sequential()
            model.add(LSTM(self.lstm_units, activation=self.activation_function, input_shape=(X_train.shape[1], X_train.shape[2])))

            # Add hidden layers based on user input
            for _ in range(self.hidden_layers):
                model.add(Dense(128, activation='relu'))  # You can adjust the number of units as needed

            model.add(Dropout(self.dropout_rate))
            model.add(Dense(1))

            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse')

            # Set up callbacks for early stopping and learning rate reduction
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_test, y_test),
                                callbacks=[early_stopping, reduce_lr], verbose=1)

            # Make predictions
            y_pred = model.predict(X_test)

            # Inverse transform predictions and actual values
            y_pred_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred.reshape(-1, 1))))[:, -1]
            y_test_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))[:, -1]

            # Calculate evaluation metrics                                                                                       mse, rmse, mae, r2
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            r2 = r2_score(y_test_inv, y_pred_inv)

            results_text = f"{epochs},{self.activation_function},{self.lstm_units},{self.hidden_layers},{self.dropout_rate},{mse},{rmse},{mae},{r2}\n"

            filename = "A_result copy.csv"

            with open(filename, "a") as f:
                f.write(results_text)

            self.results_text.insert(tk.END, "Results appended to CSV.\n")

        return epochs, mse, rmse, mae, r2, y_test_inv, y_pred_inv


    def predict_and_show(self):
        epochs = int(self.entry_epochs.get())

        if self.lstm_units is None or self.activation_function is None or self.dropout_rate is None:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please set parameters first.")
            return

        results = self.train_and_predict()
        
        results_text = f"Epochs: {results[0]}\nMSE: {results[1]}\nRMSE: {results[2]}\nMAE: {results[3]}\nR2: {results[4]}"
        self.results_text.delete(1.0, tk.END)  # Clear previous results
        self.results_text.insert(tk.END, results_text)

        # Plot predictions vs. actual values
        plt.figure(figsize=(10, 6))
        plt.plot(results[5], label='Actual')
        plt.plot(results[6], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Groundwater Level')
        plt.title('Groundwater Level Prediction using LSTM')
        plt.legend()
        plt.show()


    def append_to_csv(self):
        result = self.train_and_predict()
        # filename = open("A_result copy.csv", "a")

        # with open(filename, "a") as f:
        #     f.write(results_text)

        # self.results_text.insert(tk.END, "Results appended to CSV.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GroundwaterPredictionApp(root)
    root.mainloop()


