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
from numpy import mean

df = pd.read_csv('GWL1993-2021_norm_uni.csv')

# df1 = df.dropna()    
# df1.insert(2,'norm','null')             

# print(df1.head())

# df2 = df
# df2['Current Level'] = df2['Current Level'].fillna(df2['Current Level'].mean())

# df2.to_csv('gwl_fillna.csv')
# print(df)
print("##########---------------##########")
# print(df2['Current Level'])

lst_null = df['Current Level'].isnull()

for i in range(0,len(df['Current Level'])):
    print(df['Current Level'][i])
    if lst_null[i]==True:
        df['Current Level'][i] = df['Current Level'][1:(i)].mean()
        print("index:",i+1,"      df['Current Level'][i]:",df['Current Level'][i])

df.to_csv("gwl_preproc.csv",index=False)
# numbers = [i for i in range(1, ) if lst_null[i]==True] 

# print(lst_null[2])
# print(df['Current Level'][1])


