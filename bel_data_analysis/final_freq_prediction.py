#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[23]:


def pre_process(filename):
    data=pd.read_csv(filename)
    column_values=['netID', 'frequency', 'date', 'time', 'unitName', 'unitType','emitterID', 'callsign', 'latitude', 'longitude', 'country', 'language','fleettype', 'platform', 'pennantnumber', 'shiptype', 'firing','defects', 'groupval', 'lfid', 'azimuth', 'quality', 'elevation']
    data=data[column_values]
    data=data.replace('unknown',np.nan)
    columns = data.columns
    for i in columns:
        if data[i].isna().sum() > data.shape[0]/2:
            data.drop(i,axis=1,inplace=True)
        else:
            data[i].fillna(method='backfill',inplace=True)
    data_unit1=pd.get_dummies(data['unitName'],prefix='unitName')
    data_pre=pd.concat([data,data_unit1],axis=1)
    data_pre.drop(['unitName'],axis=1,inplace=True)
    data_unit2=pd.get_dummies(data_pre['unitType'],prefix='unitType')
    data_pre=pd.concat([data_pre,data_unit2],axis=1)
    data_pre.drop(['unitType'],axis=1,inplace=True)
    data_pre.index=data_pre['date'] + ' ' + data_pre['time']
    #data_pre.index=data_pre['date']
    data_pre['newdate'] = pd.to_datetime(data_pre['date']).map(dt.datetime.toordinal)
    time_new = pd.to_datetime(data_pre['time'], format='%H:%M:%S')
    data_pre['newtime'] = time_new.dt.hour*3600+time_new.dt.minute*3600+time_new.dt.second
    data_pre.drop(['lfid','emitterID', 'unitName_Goa','unitType_VHF','date','time'],axis=1,inplace=True)
    data_pre.dropna(inplace=True)
    data_pre.to_csv('preprocess.csv')
    return data_pre


# In[24]:


def train_test(data):
    x = data.loc[:,data.columns!='frequency']
    y = data[['frequency']]
    scaler = MinMaxScaler()
    scaler.fit(x)
    scaled_x = scaler.fit_transform(x)
    scaled_data_pre_x = pd.DataFrame(scaled_x, columns=x.columns, index=x.index)
    #scaler1 = MinMaxScaler()
    #scaler1.fit(y)
    #scaled_y = scaler1.fit_transform(y)
    #scaled_data_pre_y = pd.DataFrame(scaled_y, columns=y.columns,index=y.index)
    scaled_data_pre_y = y['frequency'].div(1000000).round(2)
    x_train, x_test, y_train, y_test = train_test_split(scaled_data_pre_x, scaled_data_pre_y, test_size = 0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    #from sklearn.svm import SVR
    #model = SVR(C=.1, gamma=1, kernel="linear")
    predictions = model.predict(x_test)
    y_test = pd.DataFrame(y_test, columns=['frequency'])
    predictions = pd.DataFrame(predictions, columns=['prediction'], index=y_test.index)
    predictions=predictions.sort_index()
    y_test = y_test.sort_index()
    final_csv = pd.concat([y_test,predictions],axis=1)
    final_csv['date'] = final_csv.index
    final_csv = final_csv.reset_index(drop=True)
    final_csv.to_csv('prediction.csv', index=False)
    #plot_df(predictions,x=y_test.index, y=predictions.prediction)
    #plot_df(y_test,x=y_test.index, y=y_test.frequency,colour='blue')
    return final_csv


# In[25]:


#def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100,colour='red'):
 #   plt.figure(figsize=(16,5), dpi=dpi)
  #  plt.plot(x, y, color=colour)
   # plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    #plt.show()


# In[26]:


data = pre_process('Dummydata.csv')
train_test(data)


# In[ ]:




