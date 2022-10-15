#  .\gwlenv\Scripts\activate

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization
from pandas.plotting import register_matplotlib_converters
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r'GWL 1993-2021 modified.csv', header=None)
df_close = pd.DataFrame(df[1])
df_close.describe()



