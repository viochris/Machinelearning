from statistics import linear_regression
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
from sklearn.metrics import r2_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings('ignore')

house = pd.read_csv('ML/youtube/lat3-housing/Housing.csv')
print(house.head())
print(house.describe())
print(house.info())
print(house.shape)
print(house.isnull().sum())
print(house.isnull().sum() / len(house) * 100)
print(house.isnull().sum() / house.shape[0] * 100)

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
sns.scatterplot(x=house.index, y=house['price'], ax=ax[0, 0], color='blue')
sns.scatterplot(x=house.index, y=house['area'], ax=ax[0, 1], color='green')
sns.scatterplot(x=house.index, y=house['bedrooms'], ax=ax[0, 2], color='orange')
sns.scatterplot(x=house.index, y=house['bathrooms'], ax=ax[1, 0], color='red')
sns.scatterplot(x=house.index, y=house['stories'], ax=ax[1, 1], color='purple')
sns.scatterplot(x=house.index, y=house['parking'], ax=ax[1, 2], color='brown')
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(2,3, figsize=(10,5))
# plt1 = sns.boxplot(house['price'], ax = ax[0,0])
# plt2 = sns.boxplot(house['area'], ax = ax[0,1])
# plt3 = sns.boxplot(house['bedrooms'], ax = ax[0,2])
# plt4 = sns.boxplot(house['bathrooms'], ax = ax[1,0])
# plt5 = sns.boxplot(house['stories'], ax = ax[1,1])
# plt6 = sns.boxplot(house['parking'], ax = ax[1,2])
# plt.tight_layout()
# plt.show()

q1 = house['price'].quantile(0.25)
q3 = house['price'].quantile(0.75)
iqr = q3 - q1
house = house[(house['price'] >= q1 - 1.5*iqr) & (house['price'] <= q3 + 1.5*iqr )]


q1 = np.percentile(house['area'], 25)
q3 = np.percentile(house['area'], 75)
iqr = q3 - q1
house = house[(house['area'] >= q1 - 1.5*iqr) & (house['area'] <= q3 + 1.5*iqr )]

# plt.boxplot(house['area'])
# plt.show()

# fig, ax = plt.subplots(2,3, figsize=(10,5))
# plt1 = sns.boxplot(house['price'], ax = ax[0,0])
# plt2 = sns.boxplot(house['area'], ax = ax[0,1])
# plt3 = sns.boxplot(house['bedrooms'], ax = ax[0,2])
# plt4 = sns.boxplot(house['bathrooms'], ax = ax[1,0])
# plt5 = sns.boxplot(house['stories'], ax = ax[1,1])
# plt6 = sns.boxplot(house['parking'], ax = ax[1,2])
# plt.tight_layout()
# plt.show()




varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
# for i in varlist:
#     house[i] = house[i].map({'yes': 1, 'no': 0})
# def ubah(x):
#     return x.map({'yes': 1, 'no': 0})
# house[varlist] = house[varlist].apply(ubah)
def ubah(df, col):
    df[col] = df[col].map({'yes': 1, 'no': 0})
    return df
# for col in varlist:
#     ubah(house, col)
ubah(house, 'mainroad')
ubah(house, 'guestroom')
ubah(house, 'basement', )
ubah(house, 'hotwaterheating')
ubah(house, 'airconditioning')
ubah(house, 'prefarea')
print(house.head())

status = pd.get_dummies(house['furnishingstatus'], drop_first = True)
print(status)

# house = pd.concat([house, status], axis = 1)
house = pd.merge(house.reset_index(), status.reset_index())
# house = house.drop('furnishingstatus', axis = 1)
house = house.drop(['index','furnishingstatus'], axis = 1)
print(house.head())


np.random.seed(0)
df_train, df_test = train_test_split(house, train_size = 0.7, test_size=0.3, random_state=100)
print(df_train.head())
print(df_test.head())


scaler = MinMaxScaler()
numvars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_train[numvars] = scaler.fit_transform(df_train[numvars])
print(df_train.head())

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()

x_train = df_train
y_train = df_train.pop('price')

k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
lm = LinearRegression()
scoring = 'r2'
score = cross_val_score(lm, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
lm.fit(x_train, y_train)

rfe = RFE(estimator=lm, n_features_to_select=6)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

hasil = pd.DataFrame({
    'Features': x_train.columns,
    'support': rfe.support_,
    'rank': rfe.ranking_
})
print(hasil)

support = x_train.columns[rfe.support_]
not_support = x_train.columns[~rfe.support_]
print(support)
print(not_support)

x_train_rfe = x_train[support]
x_train_rfe = sm.add_constant(x_train_rfe)
print(x_train_rfe.head())
print(x_train_rfe.shape)




lm = sm.OLS(y_train, x_train_rfe).fit()
print(lm.summary())

vif = pd.DataFrame()
x = x_train_rfe
print(x.shape)

vif['features'] = x.columns
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif = vif.sort_values('VIF', ascending=False)
print(vif)

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values('VIF', ascending=False)
print(vif)

y_train_price = lm.predict(x_train_rfe)
res = y_train_price - y_train

df = pd.DataFrame({
    'y_train': y_train.tolist(),
    'y_train_price': y_train_price.tolist(),
    'sisa': res.tolist()
    # 'y_train': y_train.values.tolist(),
    # 'y_train_price': y_train_price.values.tolist(),
    # 'sisa': res.values.tolist()
})
print(df.head())

# fig = plt.figure()
# sns.histplot((y_train - y_train_price), bins = 20, kde=True)
# fig.suptitle('Error Terms', fontsize = 20)
# plt.xlabel('Errors', fontsize = 18)   
# plt.show()



# plt.scatter(y_train, res)
# plt.ylabel('res')
# plt.xlabel('y_train')
# plt.show()

df_test[numvars] = scaler.transform(df_test[numvars])


y_test = df_test.pop('price')
x_test = df_test
x_test = sm.add_constant(x_test)

x_test_rfe = x_test[x_train_rfe.columns]
print(x_test_rfe.head())

y_pred = lm.predict(x_test_rfe)


# df_test = pd.read_csv('ML/youtube/lat3-housing/Housing.csv')
df = pd.DataFrame({
    'Address': x_test['area'],
    'predict': y_pred
})
print(df.head())

r2 = r2_score(y_test, y_pred)
print(r2)


fig = plt.figure()
plt.scatter(y_test, y_pred)
# fig.suptitle('y_test vs y_pred', fontsize=20)
plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)   
plt.show()