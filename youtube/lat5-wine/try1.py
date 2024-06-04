import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import statsmodels.api as sm 
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

df = pd.read_csv('ML/youtube/lat5-wine/WineQT.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# fig, ax = plt.subplots(3,4, figsize=(10,5))
# plt1 = sns.boxplot(df['fixed acidity'], ax = ax[0,0])
# plt2 = sns.boxplot(df['volatile acidity'], ax = ax[0,1])
# plt3 = sns.boxplot(df['citric acid'], ax = ax[0,2])
# plt4 = sns.boxplot(df['residual sugar'], ax = ax[0,3])
# plt5 = sns.boxplot(df['chlorides'], ax = ax[1,0])
# plt6 = sns.boxplot(df['free sulfur dioxide'], ax = ax[1,1])
# plt7 = sns.boxplot(df['total sulfur dioxide'], ax = ax[1,2])
# plt8 = sns.boxplot(df['density'], ax = ax[1,3])
# plt9 = sns.boxplot(df['pH'], ax = ax[2,0])
# plt10 = sns.boxplot(df['sulphates'], ax = ax[2,1])
# plt11 = sns.boxplot(df['alcohol'], ax = ax[2,2])
# plt12 = sns.boxplot(df['quality'], ax = ax[2,3])
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(3, 4, figsize=(15, 10))
# sns.scatterplot(x=df.index, y=df['fixed acidity'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['volatile acidity'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=df.index, y=df['citric acid'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['residual sugar'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=df.index, y=df['chlorides'], ax=ax[1, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['free sulfur dioxide'], ax=ax[1, 1], color='green')
# sns.scatterplot(x=df.index, y=df['total sulfur dioxide'], ax=ax[1, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['density'], ax=ax[1, 3], color='red')
# sns.scatterplot(x=df.index, y=df['pH'], ax=ax[2, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['sulphates'], ax=ax[2, 1], color='green')
# sns.scatterplot(x=df.index, y=df['alcohol'], ax=ax[2, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['quality'], ax=ax[2, 3], color='red')
# plt.tight_layout()
# plt.show()

q1 = df['fixed acidity'].quantile(0.25)
q3 = df['fixed acidity'].quantile(0.75)
iqr = q3 - q1
df = df[(df['fixed acidity'] >= q1 - 1.5*iqr) & (df['fixed acidity'] <= q3 + 1.5*iqr )]


q1 = np.percentile(df['volatile acidity'], 25)
q3 = np.percentile(df['volatile acidity'], 75)
iqr = q3 - q1
df = df[(df['volatile acidity'] >= q1 - 1.5*iqr) & (df['volatile acidity'] <= q3 + 1.5*iqr )]

q1 = df['citric acid'].quantile(0.25)
q3 = df['citric acid'].quantile(0.75)
iqr = q3 - q1
df = df[(df['citric acid'] >= q1 - 1.5*iqr) & (df['citric acid'] <= q3 + 1.5*iqr )]


q1 = np.percentile(df['residual sugar'], 25)
q3 = np.percentile(df['residual sugar'], 75)
iqr = q3 - q1
df = df[(df['residual sugar'] >= q1 - 1.5*iqr) & (df['residual sugar'] <= q3 + 1.5*iqr )]

q1 = df['chlorides'].quantile(0.25)
q3 = df['chlorides'].quantile(0.75)
iqr = q3 - q1
df = df[(df['chlorides'] >= q1 - 1.5*iqr) & (df['chlorides'] <= q3 + 1.5*iqr )]


q1 = np.percentile(df['free sulfur dioxide'], 25)
q3 = np.percentile(df['free sulfur dioxide'], 75)
iqr = q3 - q1
df = df[(df['free sulfur dioxide'] >= q1 - 1.5*iqr) & (df['free sulfur dioxide'] <= q3 + 1.5*iqr )]


q1 = np.percentile(df['total sulfur dioxide'], 25)
q3 = np.percentile(df['total sulfur dioxide'], 75)
iqr = q3 - q1
df = df[(df['total sulfur dioxide'] >= q1 - 1.5*iqr) & (df['total sulfur dioxide'] <= q3 + 1.5*iqr )]


q1 = df['density'].quantile(0.25)
q3 = df['density'].quantile(0.75)
iqr = q3 - q1
df = df[(df['density'] >= q1 - 1.5*iqr) & (df['density'] <= q3 + 1.5*iqr )]


q1 = df['pH'].quantile(0.25)
q3 = df['pH'].quantile(0.75)
iqr = q3 - q1
df = df[(df['pH'] >= q1 - 1.5*iqr) & (df['pH'] <= q3 + 1.5*iqr )]


q1 = df['sulphates'].quantile(0.25)
q3 = df['sulphates'].quantile(0.75)
iqr = q3 - q1
df = df[(df['sulphates'] >= q1 - 1.5*iqr) & (df['sulphates'] <= q3 + 1.5*iqr )]


q1 = df['alcohol'].quantile(0.25)
q3 = df['alcohol'].quantile(0.75)
iqr = q3 - q1
df = df[(df['alcohol'] >= q1 - 1.5*iqr) & (df['alcohol'] <= q3 + 1.5*iqr )]


# fig, ax = plt.subplots(3, 4, figsize=(15, 10))
# sns.scatterplot(x=df.index, y=df['fixed acidity'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['volatile acidity'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=df.index, y=df['citric acid'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['residual sugar'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=df.index, y=df['chlorides'], ax=ax[1, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['free sulfur dioxide'], ax=ax[1, 1], color='green')
# sns.scatterplot(x=df.index, y=df['total sulfur dioxide'], ax=ax[1, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['density'], ax=ax[1, 3], color='red')
# sns.scatterplot(x=df.index, y=df['pH'], ax=ax[2, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['sulphates'], ax=ax[2, 1], color='green')
# sns.scatterplot(x=df.index, y=df['alcohol'], ax=ax[2, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['quality'], ax=ax[2, 3], color='red')
# plt.tight_layout()
# plt.show()

del df['Id']

print(df.head())

scaler = MinMaxScaler()
numvars = df.drop('quality', axis = 1).columns
print(numvars)
df[numvars] = scaler.fit_transform(df[numvars])
print(df.head())

df_train, df_test = train_test_split(df, train_size = 0.7, test_size=0.3, random_state=100)

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()


x_train = df_train
y_train = df_train.pop('quality')

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelNB  = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(modelNB, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelNB.fit(x_train, y_train)


rfe = RFE(estimator=modelNB, n_features_to_select=5)
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


k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelNB  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelNB, x_train_rfe, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelNB.fit(x_train_rfe, y_train)

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


y_pred = modelNB.predict(x_train_rfe)

df = pd.DataFrame({
    'y_train': y_train,
    'y_pred': y_pred,
})
print(df.head())

akurasi = accuracy_score(y_train, y_pred)
print(akurasi)
cm = confusion_matrix(y_train, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()

x_test = df_test
y_test = df_test.pop('quality')
x_test_rfe = x_test[support]
y_pred = modelNB.predict(x_test_rfe)

df = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred,
})
print(df.head())

akurasi = accuracy_score(y_test, y_pred)
print(akurasi)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()