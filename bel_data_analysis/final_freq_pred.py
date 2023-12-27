import datetime
import numpy as np
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn import linear_model, metrics, tree, neighbors
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df = pd.read_csv("Dummy.csv")
df.head()

columns = list(df.columns)
print(columns)

df = pd.read_csv('Dummy.csv', na_values=['unknown'])
#df = df.replace('unknown', np.nan)

# replacing null with mean
meanVal = df.mean(numeric_only=True)
df.fillna(value=meanVal, inplace=True)

df = pd.DataFrame(df)
# percentage of null
null_percentage = df.isnull().sum()/df.shape[0]*100
# list of columns having more than 60% null
col_to_drop = null_percentage[null_percentage > 60].keys()
df = df.drop(col_to_drop, axis=1)

# dropping column with null values
df.dropna(axis=1)

# dropping duplicates
df.drop_duplicates()

#print(df.describe())
# dropping outlier
df.drop(90, inplace=True)
df.drop(91, inplace=True)

#pd.get_dummies(df['unitName'],prefix='unitName')
#pd.get_dummies(df['unitType'],prefix='unitType')

# label encoding
label = LabelEncoder().fit_transform(df['unitName'])
label1 = LabelEncoder().fit_transform(df['unitType'])
# dropping columns after encoding
df.drop('unitName', axis=1, inplace=True)
df.drop('unitType', axis=1, inplace=True)
# appending array to dataframe
df['unitName'] = label
df['unitType'] = label1

# converting column to int datatype
df = df.astype({"netID": "int", "frequency": "int","latitude": "int", "longitude": "int", "lfid": "int", "azimuth":"int", "quality":"int", "elevation":"int", 'pennantnumber':'int','defects':'int'})

# combining date and time together in one column
df["datetime"] = df["date"] + ' ' + df["time"]
df["datetime"] = pd.to_datetime(df["datetime"]).astype('int64')
# after combining deleting the separated columns
del df['date']
del df['time']

#scaled_df = (df-df.min())/(df.max()-df.min())
scaled_df = MinMaxScaler().fit_transform(df.to_numpy())
scaled_df = pd.DataFrame(scaled_df, columns=['netID', 'frequency', 'emitterID', 'latitude', 'longitude', 'pennantnumber', 'defects', 'lfid', 'azimuth', 'quality', 'elevation', 'unitName', 'unitType', 'datetime'])
#print(scaled_df.describe().to_string())

independent = ['netID', 'emitterID', 'longitude', 'latitude', 'pennantnumber', 'defects', 'lfid', 'azimuth', 'quality', 'elevation', 'unitName', 'unitType', 'datetime']
X = scaled_df[independent]
y = scaled_df['frequency']
#print(X.info(), y.info())
#print(X.shape)
#print(type(y))
#print(y.shape)
#X=np.asarray((X))
#y=np.asarray(y)

#print(scaled_df.to_string())
#x_train = df.sample(frac=0.8, random_state=42)
#y_train = df.drop(x_train.index)
#print(x_train.shape, y_train.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# linear regresion
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print("linear_regression")
y_pred = reg.predict(X_test)
print(y_pred)
#print(type(y_pred))
#y_test = np.asarray(y_test).reshape(-1,1)
print("The accuracy of the Regression_model is: ", reg.score(X_test, y_test)*100)
print()

# SVM:
clf = SVR(C=.1, kernel="poly")  #SVR(C=.1, kernel="rbf", gamma=1)
clf.fit(X_train, y_train)
print("Simple vector machine")
print(clf.predict(X_test))
# check the accuracy on the training set
#print(svc_model.score(X_train, y_train))
print("The accuracy of the SVR_model is: ", clf.score(X_test, y_test)*100)

y_train = LabelEncoder().fit_transform(y_train)
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
print("LogisticRegression")
print(clf.predict(X_test))
#print("The accuracy of the logistic_regression is: ", clf.score(X_test, y_test)*100)
print(metrics.mean_absolute_error(y_test, y_pred))

#clf =tree.DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#print("DecisionTreeClassifier")
#print(clf.predict(X_test)))

#clf = KNeighborsClassifier()
#clf.fit(X_train, y_train)
#print("KNeighborsClassifier")
#print(clf.predict(clf.predict(X_test)))
















'''def plot_df(scaled_df, X, y, title="", xlabel='Date', ylabel='Value', dpi=100,colour='red'):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(X, y, color=colour)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(scaled_df, X, y)'''








#print(reg.predict(x_test))
#print(metrics.mean_absolute_error(y_test, y_pred))
#eighty_pct = 0.8 * df.shape[0]
#x = df.loc[:eighty_pct - 1, :]  # train
#y = df.loc[eighty_pct:, :]      # test

#x, y = train_test_split(df,random_state=42,test_size=0.2)



'''train_set = df.sample(frac=0.8, random_state=42)
# Dropping all those indexes from the dataframe that exists in the train_set
test_set = df.drop(train_set.index)
print(train_set.shape, test_set.shape)'''''







#cols = ['callsign', 'country', 'language', 'fleettype','platform','pennantnumber', 'shiptype','firing','defects','groupval']
#df = df.drop(cols, axis=1)

#df = df.replace(np.nan, None)

#df = df.set_index('netID')
#df.head()

#print(df.isnull())

#column_latitude
#unique_latitude = df['latitude'].unique()
#print(unique_latitude)
#print(len(unique_latitude))

#df['unitName'] = df['unitName'].str.lower()
#df['unitType'] = df['unitType'].str.lower()

# dropping outlier
#print(df['netID'].describe())
#print(np.where(df['netID']>13015))

#print(df['netID'].describe())

#print(df['latitude'].describe())
#print(df['longitude'].describe())
#print(df['lfid'].describe())
#print(df['azimuth'].describe())
#print(df['quality'].describe())
#print(df['elevation'].describe())

#plt.plot(X=["date"], Y=["latitude"])
#plt.show()

#whole dataset
#print(df.to_string())

#print(df.describe())

#unitName = {"Goa": 1}
#df.unitName = [unitName[item] for item in df.unitName]

#df['unitName'].replace(['Goa'], [1], inplace=True)
#df['unitType'].replace(['VHF'], [1], inplace=True)

#pd.to_datetime(df['date'] + df['time'], format='%m-%d-%Y%H:%M:%S')
#new_df = pd.to_datetime(df.dates.astype(str) + ' ' + df.time.astype(str))
# add column to dataframe
#df.insert('datetime', new_df)

#unitType = {"VHF": 1}
#df.unitType = [unitType[item] for item in df.unitType]
#changing datatype

#print(df.groupby('netID').size())

#x = pd.to_numeric(df['latitude']) #converting latitude to numeric
#cols = ['time', 'date']
#df.drop(cols, axis=1)
#df["date"] = pd.to_datetime(df["date"])
#df["time"] = pd.to_datetime(df["time"])

#df['time'] = df['time'].astype('datetime64[ns]')
#df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')#date format
#df["date"]= pd.to_datetime(df['date'] + df['time'], format='%m-%d-%Y%H:%M:%S')

#print("the length of dataset is:", len(df))



#df_norm = (df-df.min())/(df.max()-df.min())
#print(df_norm)

'''df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values

df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values

cols_other_than_id = list(df.columns)[:13]
df.drop_duplicates(inplace=True)

values_list = list()
cols_list = list()
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())*100
    cols_list.append(col)
    values_list.append(pct_missing)
pct_missing_df = pd.DataFrame()
pct_missing_df['col'] = cols_list
pct_missing_df['pct_missing'] = values_list


#missing values
less_missing_values_cols_list = list(pct_missing_df.loc[(pct_missing_df.pct_missing < 0.5) & (pct_missing_df.pct_missing > 0), 'col'].values)
df.dropna(subset=less_missing_values_cols_list, inplace=True)

print(less_missing_values_cols_list)
#less missing values
_40_pct_missing_cols_list = list(pct_missing_df.loc[pct_missing_df.pct_missing > 40, 'col'].values)
df.drop(columns=_40_pct_missing_cols_list, inplace=True)

cols_other_than_id = list(df.columns)[:13]
df.drop_duplicates(inplace=True)
print(df)
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:                   # impute values only for columns that have missing values
        med = df[col].median()            #impute with the median
        df[col] = df[col].fillna(med)

df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
for col in non_numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:                        # impute values only for columns that have missing values
        mod = df[col].describe()['top']        # impute with the most frequently occuring value
        df[col] = df[col].fillna(mod)'''

