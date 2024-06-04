import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

df = pd.read_csv('ML/youtube/lat4-iris/iris_data.csv', header = None, names = ['Sepal Length (cm)', 'Sepal Width (cm)',  'Petal Length (cm)',  'Petal Width (cm)',  'Species'])
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# fig, ax = plt.subplots(2,2, figsize=(10,5))
# plt1 = sns.boxplot(df['Sepal Length (cm)'], ax = ax[0,0])
# plt2 = sns.boxplot(df['Sepal Width (cm)'], ax = ax[0,1])
# plt3 = sns.boxplot(df['Petal Length (cm)'], ax = ax[1,0])
# plt4 = sns.boxplot(df['Petal Width (cm)'], ax = ax[1,1])
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# sns.scatterplot(x=df.index, y=df['Sepal Length (cm)'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['Sepal Width (cm)'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=df.index, y=df['Petal Length (cm)'], ax=ax[1, 0], color='orange')
# sns.scatterplot(x=df.index, y=df['Petal Width (cm)'], ax=ax[1, 1], color='red')
# plt.tight_layout()
# plt.show()

print(df['Species'].unique())
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':2})
print(df.head())

numvars = ['Sepal Length (cm)', 'Sepal Width (cm)',  'Petal Length (cm)',  'Petal Width (cm)']
scaler = MinMaxScaler()
df[numvars] = scaler.fit_transform(df[numvars])

df_train, df_test = train_test_split(df, train_size = 0.7, test_size=0.3, random_state=100)
print(df_train.head())
print(df_test.head())

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()

x_train = df_train
y_train = df_train.pop('Species')

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelNB  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelNB, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelNB.fit(x_train, y_train)

# sebenarnya tidak dibutuhkan karena jumlah kolomnya sendiri hanya 4
rfe = RFE(estimator=modelNB, n_features_to_select=3)
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
# x_train_rfe = sm.add_constant(x_train_rfe)
print(x_train_rfe.head())
print(x_train_rfe.shape)


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
y_test = df_test.pop('Species')

x_test_rfe = x_test[support]
# x_test_rfe = sm.add_constant(x_test_rfe)
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