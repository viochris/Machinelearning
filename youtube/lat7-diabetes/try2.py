import pandas as pd
import numpy as np
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


diabet = pd.read_csv('ML/youtube/lat7-diabetes/diabetes.csv')
print(diabet)
print(diabet.info())
print(diabet.describe())
print(diabet.isnull().sum())

for col in diabet:
    print(col)
    print(diabet[col].unique())
    print('\n\n')


fig, ax = plt.subplots(2, 5, figsize=(15, 10))
sns.scatterplot(x=diabet.index, y=diabet['Pregnancies'], ax=ax[0, 0], color='blue')
sns.scatterplot(x=diabet.index, y=diabet['Glucose'], ax=ax[0, 1], color='green')
sns.scatterplot(x=diabet.index, y=diabet['BloodPressure'], ax=ax[0, 2], color='orange')
sns.scatterplot(x=diabet.index, y=diabet['SkinThickness'], ax=ax[0, 3], color='red')
sns.scatterplot(x=diabet.index, y=diabet['Insulin'], ax=ax[1, 0], color='blue')
sns.scatterplot(x=diabet.index, y=diabet['BMI'], ax=ax[1, 1], color='green')
sns.scatterplot(x=diabet.index, y=diabet['DiabetesPedigreeFunction'], ax=ax[1, 2], color='orange')
sns.scatterplot(x=diabet.index, y=diabet['Age'], ax=ax[1, 3], color='red')
sns.scatterplot(x=diabet.index, y=diabet['Outcome'], ax=ax[1, 4], color='red')
plt.tight_layout()
plt.show()

q1 = diabet['Pregnancies'].quantile(0.25)
q3 = diabet['Pregnancies'].quantile(0.75)
iqr = q3 - q1
diabet = diabet[(diabet['Pregnancies'] >= q1 - 1.5*iqr) & (diabet['Pregnancies'] <= q3 + 1.5*iqr )]

q1 = np.percentile(diabet['Glucose'], 25)
q3 = np.percentile(diabet['Glucose'], 75)
iqr = q3 - q1
diabet = diabet[(diabet['Glucose'] >= q1 - 1.5*iqr) & (diabet['Glucose'] <= q3 + 1.5*iqr )]

q1 = diabet['BloodPressure'].quantile(0.25)
q3 = diabet['BloodPressure'].quantile(0.75)
iqr = q3 - q1
diabet = diabet[(diabet['BloodPressure'] >= q1 - 1.5*iqr) & (diabet['BloodPressure'] <= q3 + 1.5*iqr )]

q1 = np.percentile(diabet['SkinThickness'], 25)
q3 = np.percentile(diabet['SkinThickness'], 75)
iqr = q3 - q1
diabet = diabet[(diabet['SkinThickness'] >= q1 - 1.5*iqr) & (diabet['SkinThickness'] <= q3 + 1.5*iqr )]

q1 = diabet['Insulin'].quantile(0.25)
q3 = diabet['Insulin'].quantile(0.75)
iqr = q3 - q1
diabet = diabet[(diabet['Insulin'] >= q1 - 1.5*iqr) & (diabet['Insulin'] <= q3 + 1.5*iqr )]

q1 = np.percentile(diabet['BMI'], 25)
q3 = np.percentile(diabet['BMI'], 75)
iqr = q3 - q1
diabet = diabet[(diabet['BMI'] >= q1 - 1.5*iqr) & (diabet['BMI'] <= q3 + 1.5*iqr )]

q1 = diabet['DiabetesPedigreeFunction'].quantile(0.25)
q3 = diabet['DiabetesPedigreeFunction'].quantile(0.75)
iqr = q3 - q1
diabet = diabet[(diabet['DiabetesPedigreeFunction'] >= q1 - 1.5*iqr) & (diabet['DiabetesPedigreeFunction'] <= q3 + 1.5*iqr )]

q1 = np.percentile(diabet['Age'], 25)
q3 = np.percentile(diabet['Age'], 75)
iqr = q3 - q1
diabet = diabet[(diabet['Age'] >= q1 - 1.5*iqr) & (diabet['Age'] <= q3 + 1.5*iqr )]

q1 = np.percentile(diabet['Outcome'], 25)
q3 = np.percentile(diabet['Outcome'], 75)
iqr = q3 - q1
diabet = diabet[(diabet['Outcome'] >= q1 - 1.5*iqr) & (diabet['Outcome'] <= q3 + 1.5*iqr )]


fig, ax = plt.subplots(2, 5, figsize=(15, 10))
sns.scatterplot(x=diabet.index, y=diabet['Pregnancies'], ax=ax[0, 0], color='blue')
sns.scatterplot(x=diabet.index, y=diabet['Glucose'], ax=ax[0, 1], color='green')
sns.scatterplot(x=diabet.index, y=diabet['BloodPressure'], ax=ax[0, 2], color='orange')
sns.scatterplot(x=diabet.index, y=diabet['SkinThickness'], ax=ax[0, 3], color='red')
sns.scatterplot(x=diabet.index, y=diabet['Insulin'], ax=ax[1, 0], color='blue')
sns.scatterplot(x=diabet.index, y=diabet['BMI'], ax=ax[1, 1], color='green')
sns.scatterplot(x=diabet.index, y=diabet['DiabetesPedigreeFunction'], ax=ax[1, 2], color='orange')
sns.scatterplot(x=diabet.index, y=diabet['Age'], ax=ax[1, 3], color='red')
sns.scatterplot(x=diabet.index, y=diabet['Outcome'], ax=ax[1, 4], color='red')
plt.tight_layout()
plt.show()


numvars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = MinMaxScaler()
diabet[numvars] = scaler.fit_transform(diabet[numvars])

print(diabet)

df_train, df_test = train_test_split(diabet, train_size = 0.7, test_size=0.3, random_state=100)
print(df_train.head())
print(df_test.head())

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()

x_train = df_train
y_train = df_train.pop('Outcome')

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelLR  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelLR, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelLR.fit(x_train, y_train)


# sebenarnya tidak dibutuhkan karena jumlah kolomnya sendiri hanya 4
rfe = RFE(estimator=modelLR, n_features_to_select=5)
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

# k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
# modelLR  = LogisticRegression()
# scoring = 'accuracy'
# score = cross_val_score(modelLR, x_train_rfe, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
# print(score)
# print(round(score.mean(), 2))
# modelLR.fit(x_train_rfe, y_train)

# vif = pd.DataFrame()
# x = x_train_rfe
# print(x.shape)

# vif['features'] = x.columns
# vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
# vif = vif.sort_values('VIF', ascending=False)
# print(vif)

# vif['VIF'] = round(vif['VIF'], 2)
# vif = vif.sort_values('VIF', ascending=False)
# print(vif)

# y_pred = modelLR.predict(x_train_rfe)

# df = pd.DataFrame({
#     'y_train': y_train,
#     'y_pred': y_pred,
# })
# print(df.head())
# print(df[df['y_train'] != df['y_pred']].head())

# akurasi = accuracy_score(y_train, y_pred)
# print(akurasi)
# cm = confusion_matrix(y_train, y_pred)
# print(cm)
# sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
# plt.xlabel('y_pred')
# plt.ylabel('y_test')
# plt.title('confusion matrix')
# plt.show()



# x_test = df_test
# x_test_rfe = df_test[support]
# y_test = df_test.pop('Outcome')
# y_pred = modelLR.predict(x_test_rfe)

# df = pd.DataFrame({
#     'y_test': y_test,
#     'y_pred': y_pred,
# })
# print(df.head())
# print(df[df['y_test'] != df['y_pred']])

# akurasi = accuracy_score(y_test, y_pred)
# print(akurasi)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
# plt.xlabel('y_pred')
# plt.ylabel('y_test')
# plt.title('confusion matrix')
# plt.show()