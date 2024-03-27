
#Importing the libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data

data=pd.read_csv('/content/Car details v3.csv')

data.head()

data.info()

#Preprocessing

data.isnull().sum()

data=data.dropna(axis=0)

def convertToNumber(s:str):
  d=""
  for i in list(s):
    if (i.isdigit() or i=='.'):
      d+=i
  return eval(d)

data['mileage']=data['mileage'].apply(convertToNumber)
data['engine']=data['engine'].apply(convertToNumber)
data['max_power']=data['max_power'].apply(convertToNumber)

data.head()

data=data.drop(['torque','name'], axis=1)
data.head()

data.fuel.unique()

data.owner.unique()

from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
data['fuel']=label_encoder.fit_transform(data['fuel'])
data['owner']=label_encoder.fit_transform(data['owner'])
data['transmission']=label_encoder.fit_transform(data['transmission'])
data['seller_type']=label_encoder.fit_transform(data['seller_type'])

data.head(10)

#Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler

mmScaler=MinMaxScaler()
mmScaler_y=MinMaxScaler()

x=data.drop(columns=['selling_price']).values
y=data[['selling_price']].values

print(data.shape)
print(x.shape)
print(y.shape)

x=mmScaler.fit_transform(x)
y=mmScaler_y.fit_transform(y)

#EDA (exploratory data Analysis)

import plotly.graph_objects as go
correlation_matrix = data.corr()

corr = go.Heatmap(
    z = correlation_matrix.values,
    x = correlation_matrix.columns,
    y = correlation_matrix.columns,
    colorscale='RdYlBu',
    colorbar=dict(title='Correlation')
)

layout = go.Layout(
    title='Heatmap of Correlation',
    xaxis=dict(title='Columns'),
    yaxis=dict(title='Columns'),
    height= 800
)

fig = go.Figure(data=corr, layout=layout)
fig.show()

for column in data.columns:
  sns.scatterplot(data=data,x=column, y='selling_price')
  plt.title(f"Scatter Plot between {column} and selling price")
  plt.show()

#Split the Data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

print(f"size train : x: {X_train.shape}   ->  y: {y_train.shape}")
print(f"size test : x: {X_test.shape}   ->  y: {y_test.shape}")

def rmse(pred,test):
  return np.sqrt(((pred-test)**2).mean())

#Random Forest Regression 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,max_error,mean_squared_error

rf=RandomForestRegressor(n_estimators=10,random_state=45)
rf.fit(X_train,y_train)

y_pred_rf=rf.predict(X_test)

#Calculating various scores/metrics

rmse_rf = rmse(y_test, y_pred_rf)
r2_score_rf = r2_score(y_test, y_pred_rf)
max_error_rf = max_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"rmse : {rmse_rf}")
print(f"r2_score : {r2_score_rf}")
print(f"max_error : {max_error_rf}")
print(f"mean_squared_error : {mse_rf}")
