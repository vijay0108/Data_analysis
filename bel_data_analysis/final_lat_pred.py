import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn import linear_model, metrics, tree, neighbors
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats

# importing dataset
data = pd.read_csv("Dummydata_latitude_pred.csv")
column_values = ['netID', 'frequency', 'date', 'time', 'unitName', 'unitType', 'emitterID', 'callsign', 'latitude',
                 'longitude', 'country', 'language', 'fleettype', 'platform', 'pennantnumber', 'shiptype', 'firing',
                 'defects', 'groupval', 'lfid', 'azimuth', 'quality', 'elevation']

#data = data[column_values]

# Replacing unknown with none
data = data.replace('unknown', np.nan)
#columns = data.columns

#meanVal = data.mean(numeric_only=True)
#print(meanVal)

# sum of null values
for i in data:
    if data[i].isna().sum() > data.shape[0] / 2:
        data.drop(i, axis=1, inplace=True)
    else:
        data[i].fillna(method='ffill', inplace=True)

# dropping column with null values
data.dropna(axis=1)

# dropping duplicates
data.drop_duplicates()

# Finding outliers
q1 = data["netID"].quantile(0.25)
q3 = data["netID"].quantile(0.75)
iqr = q3-q1 #Interquartile range
low = q1-1.5*iqr
high = q3+1.5*iqr
data = data.loc[(data["netID"] > low) & (data["netID"] < high)]

# converting character values to integer values
data_unit1 = pd.get_dummies(data['unitName'], prefix='unitName')
data_pre = pd.concat([data, data_unit1], axis=1)
data_pre.drop(['unitName'], axis=1, inplace=True)
data_unit2 = pd.get_dummies(data_pre['unitType'], prefix='unitType')
data_pre = pd.concat([data_pre, data_unit2], axis=1)
data_pre.drop(['unitType'], axis=1, inplace=True)
#data_pre.index = data_pre['date'] + ' ' + data_pre['time']
data_pre["datetime"] = data_pre["date"] + ' ' + data_pre["time"]
data_pre["datetime"] = pd.to_datetime(data_pre["datetime"]).astype("int64")

# after combining deleting the separated columns
data_pre.drop(['lfid', 'emitterID', 'unitName_Goa', 'unitType_VHF', 'date', 'time'], axis=1, inplace=True)
data_pre.dropna(inplace=True)

# converting the columns to float
data_pre["frequency"].astype(float)
data_pre["latitude"].astype(float)
data_pre["longitude"].astype(float)
data_pre["azimuth"].astype(float)
data_pre["elevation"].astype(float)
#print(data_pre.describe())

# Scaling the dataset
data_pre = MinMaxScaler().fit_transform(data_pre.to_numpy())
data_pre = pd.DataFrame(data_pre, columns=['netID', 'frequency', 'latitude', 'longitude', 'azimuth', 'quality', 'elevation','datetime'])
#print((data_pre.to_string()))

# Dividing the dataset into X and y
#independent = ['netID', 'frequency', 'longitude', 'azimuth', 'quality', 'elevation', 'datetime']
X = data_pre.drop('latitude', axis = 1)
y = data_pre['latitude']

# Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# linear regresion
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print("linear_regression")
y_pred = reg.predict(X_test)
print(y_pred)
#print(y_test)
#y_test = np.asarray(y_test).reshape(-1,1)
print("The accuracy of the Regression_model is: ", reg.score(X_test, y_test)*100)
print("The Mean Squared Error for Linear Regression model is: ", mean_squared_error(y_test, y_pred))
print("  ")

# SVM:
clf = SVR()
clf = clf.fit(X_train, y_train)
print("Simple vector machine")
pred = clf.predict(X_test)
print(pred)
# check the accuracy on the training set
#print(svc_model.score(X_train, y_train))
print("The accuracy of the SVR_model is: ", clf.score(X_test, y_test)*100)
#print(y_test)
#print(data_pre.to_string())
print("The Mean Squared Error for SVR model is: ", mean_squared_error(y_test, pred))