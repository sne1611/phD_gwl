#  .\gwlenv\Scripts\activate

from lib2to3.pytree import LeafPattern
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization
from pandas.plotting import register_matplotlib_converters
from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import sklearn.metrics as metrics




from sklearn import preprocessing 

from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r'GWL 1993-2021 modified.csv', header=None)

df_close = pd.DataFrame(df[1])
df_close.describe()

register_matplotlib_converters()

plt.figure(figsize=(8, 6))
plt.plot(df_close, color='g')
#plt.title('GWL', weight='bold', fontsize=16)
plt.xlabel('Time', weight='bold', fontsize=14)
plt.ylabel('GWL', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth = 0.5)
plt.show()


input_layer = Input(shape=(15), dtype='float32')
dense1 = Dense(60, activation='relu')(input_layer)
dense2 = Dense(60, activation='relu')(dense1)
dropout_layer = Dropout(0.2)(dense2)
output_layer = Dense(1, activation='linear')(dropout_layer)


model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()



def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :
    
    """
    Ensure that the index is of datetime type
    Creates features with previous time instant values
    """
        
    list_of_prev_t_instants.sort()
    start = list_of_prev_t_instants[-1] 
    end = len(df)
    df['datetime'] = df.index
    df.reset_index(drop=True)

    df_copy = df[start:end]
    df_copy.reset_index(inplace=True, drop=True)

    for attribute in attribute :
            foobar = pd.DataFrame()

            for prev_t in list_of_prev_t_instants :
                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])
                new_col.reset_index(drop=True, inplace=True)
                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)
                foobar = pd.concat([foobar, new_col], sort=False, axis=1)

            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)
            
    df_copy.set_index(['datetime'], drop=True, inplace=True)
    return df_copy




list_of_attributes = [1]
list_of_prev_t_instants = []
for i in range(1,16):
    list_of_prev_t_instants.append(i)
list_of_prev_t_instants



df_new = create_regressor_attributes(df_close, list_of_attributes, list_of_prev_t_instants)
df_new.head()



df_new.shape


input_layer = Input(shape=(15), dtype='float32')
dense1 = Dense(60, activation='relu')(input_layer)
dense2 = Dense(60, activation='relu')(dense1)
dropout_layer = Dropout(0.2)(dense2)
output_layer = Dense(1, activation='linear')(dropout_layer)



model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.summary()



test_set_size = 0.05
valid_set_size= 0.05

df_copy = df_new.reset_index(drop=True)

df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]
df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]

df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]
df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]

X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

print('Shape of training inputs, training target:', X_train.shape, y_train.shape)
print('Shape of validation inputs, validation target:', X_valid.shape, y_valid.shape)
print('Shape of test inputs, test target:', X_test.shape, y_test.shape)



Target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
Feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))
X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))
X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))

y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1,1))
y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1,1))
y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1,1))


epoch = 10
history = model.fit(x=X_train_scaled, y=y_train_scaled, batch_size=5, epochs=epoch, verbose=1, validation_data=(X_valid_scaled, y_valid_scaled), shuffle=True)


y_pred = model.predict(X_test_scaled)


y_pred_rescaled = Target_scaler.inverse_transform(y_pred)


y_test_rescaled =  Target_scaler.inverse_transform(y_test_scaled)
score = r2_score(y_test_rescaled, y_pred_rescaled)
print('R-squared score for the test set:', round(score,4))



y_actual = pd.DataFrame(y_test_rescaled, columns=['Actual gwl'])
y_hat = pd.DataFrame(y_pred_rescaled, columns=['Predicted gwl'])


plt.figure(figsize=(11, 6))
plt.plot([1, 2, 3, 4], [1, 6, 12, 18],color='w')

plt.plot(y_actual, linestyle='solid', color='b', label = 'Actual')
plt.plot(y_hat, linestyle='dashed', color='r', label = 'Predicted')

plt.legend(loc='best', prop={'size': 14})
plt.title('Ground Water level Forecasting', weight='bold', fontsize=16)
plt.ylabel('GWL', weight='bold', fontsize=14)
plt.xlabel('', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth='0.1')
plt.show()



rms = sqrt(mean_squared_error(y_actual, y_hat))
print("rms = ",rms)


mae = metrics.mean_absolute_error(y_actual, y_hat)
mse = metrics.mean_squared_error(y_actual, y_hat)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(y_actual, y_hat)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)