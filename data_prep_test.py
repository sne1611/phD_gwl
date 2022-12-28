'''
To fill the null values we have
1.  replace     
2.  dropna  
3.  fill with previous values - df.fillna(method="pad")
4.  fill with next value - df.fillna(method="bfill")
5.  with mean of a column df.fillna(value = df['Current Level'].mean())
6.  with max or min - df.fillna(value = df['Current Level'].max()  or df.fillna(value = df['Current Level'].min()
7.  interpolate
'''
import pandas as pd

df = pd.read_csv('GWL1993-2021_norm_uni.csv')

df1 = df.dropna()    
df1.insert(2,'norm','null')             

print(df1.head())

df2 = df
df2['Current Level'] = df2['Current Level'].fillna(df2['Current Level'].mean())

df2.to_csv('gwl_fillna.csv')
print(df)
print("-------------------")
print(df2)