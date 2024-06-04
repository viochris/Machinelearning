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
from sklearn.neural_network import MLPRegressor

train = pd.read_csv('ML/youtube/lat6-house/train.csv')
test = pd.read_csv('ML/youtube/lat6-house/test.csv')
print(train.head())
print(test.head())
print(train.shape, test.shape)
print(train.isnull().sum())
print(test.isnull().sum())

# fig, ax = plt.subplots(1,2, figsize=(10,5))
# plt1 = sns.boxplot(train['SQUARE_FT'], ax = ax[0])
# plt2 = sns.boxplot(test['SQUARE_FT'], ax = ax[1])
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(2,3,figsize=(15, 10))
# sns.scatterplot(x=train.index, y=train['SQUARE_FT'], ax=ax[0,0], color='blue')
# sns.scatterplot(x=train.index, y=train['BHK_NO.'], ax=ax[0, 1], color='blue')
# sns.scatterplot(x=test.index, y=test['SQUARE_FT'], ax=ax[0, 2], color='blue')
# sns.scatterplot(x=test.index, y=test['BHK_NO.'], ax=ax[1,0], color='blue')
# sns.scatterplot(x=train.index, y=train['TARGET(PRICE_IN_LACS)'], ax=ax[1,1], color='blue')
# plt.tight_layout()
# plt.show()

# q1 = train['SQUARE_FT'].quantile(0.25)
# q3 = train['SQUARE_FT'].quantile(0.75)
# iqr = q3 - q1
# train = train[(train['SQUARE_FT'] >= q1 - 1.5*iqr) & (train['SQUARE_FT'] <= q3 + 1.5*iqr )]


# q1 = np.percentile(test['SQUARE_FT'], 25)
# q3 = np.percentile(test['SQUARE_FT'], 75)
# iqr = q3 - q1
# test = test[(test['SQUARE_FT'] >= q1 - 1.5*iqr) & (test['SQUARE_FT'] <= q3 + 1.5*iqr )]


# q1 = train['BHK_NO.'].quantile(0.25)
# q3 = train['BHK_NO.'].quantile(0.75)
# iqr = q3 - q1
# train = train[(train['BHK_NO.'] >= q1 - 1.5*iqr) & (train['BHK_NO.'] <= q3 + 1.5*iqr )]


# q1 = np.percentile(test['BHK_NO.'], 25)
# q3 = np.percentile(test['BHK_NO.'], 75)
# iqr = q3 - q1
# test = test[(test['BHK_NO.'] >= q1 - 1.5*iqr) & (test['BHK_NO.'] <= q3 + 1.5*iqr )]


# q1 = np.percentile(train['TARGET(PRICE_IN_LACS)'], 25)
# q3 = np.percentile(train['TARGET(PRICE_IN_LACS)'], 75)
# iqr = q3 - q1
# train = train[(train['TARGET(PRICE_IN_LACS)'] >= q1 - 1.5*iqr) & (train['TARGET(PRICE_IN_LACS)'] <= q3 + 1.5*iqr )]

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

# scaler = MinMaxScaler()
# numvars = ['SQUARE_FT', 'TARGET(PRICE_IN_LACS)']
# train[numvars] = scaler.fit_transform(train[numvars])

# # tidak bisa krena hanya ada satu yang diubah
# # test[['SQUARE_FT']] = scaler.transform(test[['SQUARE_FT']])

# print(train.head())
# print(test.head())

# train_data = train.drop('TARGET(PRICE_IN_LACS)', axis = 1)
# target = train['TARGET(PRICE_IN_LACS)']

train_data = train
target = train.pop('TARGET(PRICE_IN_LACS)')

k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
modelGB = GradientBoostingRegressor()
scoring = 'r2'
score = cross_val_score(modelGB, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
lm = LinearRegression()
scoring = 'r2'
score = cross_val_score(lm, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

modelGB.fit(train_data, target)

y_pred = modelGB.predict(test)

print(y_pred)


