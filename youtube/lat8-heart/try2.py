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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor


heart = pd.read_csv('ML/youtube/lat8-heart/heart.csv')
print(heart)
print(heart.info())
print(heart.describe())
print(heart.isnull().sum())

for col in heart:
    print(col)
    print(heart[col].unique())
    print('\n\n')
    
# fig, ax = plt.subplots(3, 5, figsize=(15, 10))
# sns.scatterplot(x=heart.index, y=heart['age'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=heart.index, y=heart['sex'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=heart.index, y=heart['cp'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=heart.index, y=heart['trestbps'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=heart.index, y=heart['chol'], ax=ax[0, 4], color='blue')
# sns.scatterplot(x=heart.index, y=heart['fbs'], ax=ax[1, 0], color='green')
# sns.scatterplot(x=heart.index, y=heart['restecg'], ax=ax[1, 1], color='orange')
# sns.scatterplot(x=heart.index, y=heart['thalach'], ax=ax[1, 2], color='red')
# sns.scatterplot(x=heart.index, y=heart['exang'], ax=ax[1, 3], color='blue')
# sns.scatterplot(x=heart.index, y=heart['oldpeak'], ax=ax[1, 4], color='green')
# sns.scatterplot(x=heart.index, y=heart['slope'], ax=ax[2, 0], color='orange')
# sns.scatterplot(x=heart.index, y=heart['ca'], ax=ax[2, 1], color='red')
# sns.scatterplot(x=heart.index, y=heart['thal'], ax=ax[2, 2], color='red')
# sns.scatterplot(x=heart.index, y=heart['target'], ax=ax[2, 3], color='red')
# plt.tight_layout()
# plt.show()

q1 = heart['age'].quantile(0.25)
q3 = heart['age'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['age'] >= q1 - 1.5*iqr) & (heart['age'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['sex'], 25)
q3 = np.percentile(heart['sex'], 75)
iqr = q3 - q1
heart = heart[(heart['sex'] >= q1 - 1.5*iqr) & (heart['sex'] <= q3 + 1.5*iqr )]

q1 = heart['cp'].quantile(0.25)
q3 = heart['cp'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['cp'] >= q1 - 1.5*iqr) & (heart['cp'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['trestbps'], 25)
q3 = np.percentile(heart['trestbps'], 75)
iqr = q3 - q1
heart = heart[(heart['trestbps'] >= q1 - 1.5*iqr) & (heart['trestbps'] <= q3 + 1.5*iqr )]

q1 = heart['chol'].quantile(0.25)
q3 = heart['chol'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['chol'] >= q1 - 1.5*iqr) & (heart['chol'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['fbs'], 25)
q3 = np.percentile(heart['fbs'], 75)
iqr = q3 - q1
heart = heart[(heart['fbs'] >= q1 - 1.5*iqr) & (heart['fbs'] <= q3 + 1.5*iqr )]

q1 = heart['restecg'].quantile(0.25)
q3 = heart['restecg'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['restecg'] >= q1 - 1.5*iqr) & (heart['restecg'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['thalach'], 25)
q3 = np.percentile(heart['thalach'], 75)
iqr = q3 - q1
heart = heart[(heart['thalach'] >= q1 - 1.5*iqr) & (heart['thalach'] <= q3 + 1.5*iqr )]

q1 = heart['exang'].quantile(0.25)
q3 = heart['exang'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['exang'] >= q1 - 1.5*iqr) & (heart['exang'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['oldpeak'], 25)
q3 = np.percentile(heart['oldpeak'], 75)
iqr = q3 - q1
heart = heart[(heart['oldpeak'] >= q1 - 1.5*iqr) & (heart['oldpeak'] <= q3 + 1.5*iqr )]

q1 = heart['slope'].quantile(0.25)
q3 = heart['slope'].quantile(0.75)
iqr = q3 - q1
heart = heart[(heart['slope'] >= q1 - 1.5*iqr) & (heart['slope'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['ca'], 25)
q3 = np.percentile(heart['ca'], 75)
iqr = q3 - q1
heart = heart[(heart['ca'] >= q1 - 1.5*iqr) & (heart['ca'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['thal'], 25)
q3 = np.percentile(heart['thal'], 75)
iqr = q3 - q1
heart = heart[(heart['thal'] >= q1 - 1.5*iqr) & (heart['thal'] <= q3 + 1.5*iqr )]

q1 = np.percentile(heart['target'], 25)
q3 = np.percentile(heart['target'], 75)
iqr = q3 - q1
heart = heart[(heart['target'] >= q1 - 1.5*iqr) & (heart['target'] <= q3 + 1.5*iqr )]

# fig, ax = plt.subplots(3, 5, figsize=(15, 10))
# sns.scatterplot(x=heart.index, y=heart['age'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=heart.index, y=heart['sex'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=heart.index, y=heart['cp'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=heart.index, y=heart['trestbps'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=heart.index, y=heart['chol'], ax=ax[0, 4], color='blue')
# sns.scatterplot(x=heart.index, y=heart['fbs'], ax=ax[1, 0], color='green')
# sns.scatterplot(x=heart.index, y=heart['restecg'], ax=ax[1, 1], color='orange')
# sns.scatterplot(x=heart.index, y=heart['thalach'], ax=ax[1, 2], color='red')
# sns.scatterplot(x=heart.index, y=heart['exang'], ax=ax[1, 3], color='blue')
# sns.scatterplot(x=heart.index, y=heart['oldpeak'], ax=ax[1, 4], color='green')
# sns.scatterplot(x=heart.index, y=heart['slope'], ax=ax[2, 0], color='orange')
# sns.scatterplot(x=heart.index, y=heart['ca'], ax=ax[2, 1], color='red')
# sns.scatterplot(x=heart.index, y=heart['thal'], ax=ax[2, 2], color='red')
# sns.scatterplot(x=heart.index, y=heart['target'], ax=ax[2, 3], color='red')
# plt.tight_layout()
# plt.show()

numvars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = MinMaxScaler()
heart[numvars] = scaler.fit_transform(heart[numvars])

print(heart)


df_train, df_test = train_test_split(heart, train_size = 0.7, test_size=0.3, random_state=100)
print(df_train.head())
print(df_test.head())

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()


x_train = df_train
y_train = df_train.pop('target')

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelRF  = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(modelRF, x_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelRF.fit(x_train, y_train)



# sebenarnya tidak dibutuhkan karena jumlah kolomnya sendiri hanya 4
rfe = RFE(estimator=modelRF, n_features_to_select=5)
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
modelRF  = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(modelRF, x_train_rfe, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelRF.fit(x_train_rfe, y_train)



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

y_pred = modelRF.predict(x_train_rfe)

df = pd.DataFrame({
    'y_train': y_train,
    'y_pred': y_pred,
})
print(df.head())
print(df[df['y_train'] != df['y_pred']].head())

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
x_test_rfe = x_test[support]
y_test = df_test.pop('target')
y_pred = modelRF.predict(x_test_rfe)

df = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred,
})
print(df.head())
print(df[df['y_test'] != df['y_pred']])

akurasi = accuracy_score(y_test, y_pred)
print(akurasi)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()