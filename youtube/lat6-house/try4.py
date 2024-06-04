import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score 
from sklearn.neural_network import MLPRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

train = pd.read_csv('ML/youtube/lat6-house/train.csv')
test = pd.read_csv('ML/youtube/lat6-house/test.csv')
print(train.head())
print(test.head())
print(train.shape, test.shape)
print(train.isnull().sum())
print(test.isnull().sum())


# fig, ax = plt.subplots(2,3,figsize=(15, 10))
# sns.scatterplot(x=train.index, y=train['SQUARE_FT'], ax=ax[0,0], color='blue')
# sns.scatterplot(x=train.index, y=train['BHK_NO.'], ax=ax[0, 1], color='blue')
# sns.scatterplot(x=test.index, y=test['SQUARE_FT'], ax=ax[0, 2], color='blue')
# sns.scatterplot(x=test.index, y=test['BHK_NO.'], ax=ax[1,0], color='blue')
# sns.scatterplot(x=train.index, y=train['TARGET(PRICE_IN_LACS)'], ax=ax[1,1], color='blue')
# plt.tight_layout()
# plt.show()

q1 = train['SQUARE_FT'].quantile(0.25)
q3 = train['SQUARE_FT'].quantile(0.75)
iqr = q3 - q1
train = train[(train['SQUARE_FT'] >= q1 - 1.5*iqr) & (train['SQUARE_FT'] <= q3 + 1.5*iqr )]


q1 = np.percentile(test['SQUARE_FT'], 25)
q3 = np.percentile(test['SQUARE_FT'], 75)
iqr = q3 - q1
test = test[(test['SQUARE_FT'] >= q1 - 1.5*iqr) & (test['SQUARE_FT'] <= q3 + 1.5*iqr )]


q1 = train['BHK_NO.'].quantile(0.25)
q3 = train['BHK_NO.'].quantile(0.75)
iqr = q3 - q1
train = train[(train['BHK_NO.'] >= q1 - 1.5*iqr) & (train['BHK_NO.'] <= q3 + 1.5*iqr )]


q1 = np.percentile(test['BHK_NO.'], 25)
q3 = np.percentile(test['BHK_NO.'], 75)
iqr = q3 - q1
test = test[(test['BHK_NO.'] >= q1 - 1.5*iqr) & (test['BHK_NO.'] <= q3 + 1.5*iqr )]


q1 = np.percentile(train['TARGET(PRICE_IN_LACS)'], 25)
q3 = np.percentile(train['TARGET(PRICE_IN_LACS)'], 75)
iqr = q3 - q1
train = train[(train['TARGET(PRICE_IN_LACS)'] >= q1 - 1.5*iqr) & (train['TARGET(PRICE_IN_LACS)'] <= q3 + 1.5*iqr )]

# fig, ax = plt.subplots(2,3,figsize=(15, 10))
# sns.scatterplot(x=train.index, y=train['SQUARE_FT'], ax=ax[0,0], color='blue')
# sns.scatterplot(x=train.index, y=train['BHK_NO.'], ax=ax[0, 1], color='blue')
# sns.scatterplot(x=test.index, y=test['SQUARE_FT'], ax=ax[0, 2], color='blue')
# sns.scatterplot(x=test.index, y=test['BHK_NO.'], ax=ax[1,0], color='blue')
# sns.scatterplot(x=train.index, y=train['TARGET(PRICE_IN_LACS)'], ax=ax[1,1], color='blue')
# plt.tight_layout()
# plt.show()


for col in train:
    print(col)
    print(train[col].unique())
    print()

post_train = pd.get_dummies(train['POSTED_BY'])
post_test = pd.get_dummies(test['POSTED_BY'])



train = pd.concat([train, post_train], axis = 1)
del train['POSTED_BY']
test = pd.merge(test.reset_index(), post_test.reset_index())
test = test.drop(['index', 'POSTED_BY'], axis = 1)

train['BHK_OR_RK'] = train['BHK_OR_RK'].map({'BHK': 0, 'RK': 1})
test['BHK_OR_RK'] = test['BHK_OR_RK'].map({'BHK': 0, 'RK': 1})

train['alamat'] = train['ADDRESS'].str.split(',').str.get(-1)
test['alamat'] = test['ADDRESS'].str.split(',').str.get(-1)

def city_group(city):
    if city in ['Bangalore', 'Mysore', 'Chennai', 'Mumbai', 'Delhi', 'Kolkata']:
        return 0
    elif city in ['Pune', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur']:
        return 1
    else:
        return 2 
train['alamat'] = train['alamat'].apply(city_group)
del train['ADDRESS']
test['alamat'] = test['alamat'].apply(city_group)
test = test.drop('ADDRESS', axis = 1)


print(train.head())
print(test.head())

df_train, df_test = train_test_split(train, train_size = 0.7, test_size=0.3, random_state=100)

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()

x_train = df_train
y_train = df_train.pop('TARGET(PRICE_IN_LACS)')


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
modelGB = LinearRegression()
scoring = 'r2'
score = cross_val_score(modelGB, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelGB.fit(x_train, y_train)

rfe = RFE(estimator=modelGB, n_features_to_select=6)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

hasil = pd.DataFrame({
    'features': x_train.columns,
    'support': rfe.support_,
    'rank': rfe.ranking_,
})
print(hasil)

support = x_train.columns[rfe.support_]
not_support = x_train.columns[~rfe.support_]
print(support)
print(not_support)

x_train_rfe = x_train[support]
x_train_rfe['Builder'] = x_train_rfe['Builder'].astype(int)
x_train_rfe['Dealer'] = x_train_rfe['Dealer'].astype(int)
x_train_rfe['Owner'] = x_train_rfe['Owner'].astype(int)


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
modelGB = GradientBoostingRegressor()
scoring = 'r2'
score = cross_val_score(modelGB, x_train_rfe, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelGB.fit(x_train_rfe, y_train)


vif = pd.DataFrame()
x = x_train_rfe
print(x.shape)
print(x.head())

vif['features'] = x.columns
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif = vif.sort_values('VIF', ascending=False)
print(vif)

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values('VIF', ascending=False)
print(vif)


y_train_price = modelGB.predict(x_train_rfe)
res = y_train_price - y_train
print(res)

df = pd.DataFrame({
    'y_train': y_train.values,
    'y_train_price': y_train_price,
    'sisa': res
})
print(df)
r2 = r2_score(y_train, y_train_price)
print(r2)

# fig = plt.figure()
# sns.histplot((y_train - y_train_price), bins = 20, kde=True)
# fig.suptitle('Error Terms', fontsize = 20)
# plt.xlabel('Errors', fontsize = 18)   
# plt.show()

# plt.scatter(y_train, res)
# plt.ylabel('res')
# plt.xlabel('y_train')
# plt.show()

x_test = df_test
y_test = df_test.pop('TARGET(PRICE_IN_LACS)')
x_test_rfe = x_test[support]

y_pred = modelGB.predict(x_test_rfe)
res = y_pred - y_test
print(res)

df = pd.DataFrame({
    'y_train': y_test.values,
    'y_train_price': y_pred,
    'sisa': res
})
print(df)

r2 = r2_score(y_test, y_pred)
print(r2)

fig = plt.figure()
plt.scatter(y_test, y_pred)
# fig.suptitle('y_test vs y_pred', fontsize=20)
plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)   
plt.show()



test = test[support]
prediction = modelGB.predict(test)
print(prediction)

# df_test = pd.read_csv('ML/youtube/lat6-house/test.csv')
# df = pd.DataFrame({
#     'Address': df_test['ADDRESS'],
#     'predict': prediction
# })
# print(df.head())