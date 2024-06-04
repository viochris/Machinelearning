import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

diabet = pd.read_csv('ML/youtube/lat7-diabetes/diabetes.csv')
print(diabet)
print(diabet.apply(lambda x: x.dtypes))
print(diabet.apply(lambda x: x.nunique()))
print(diabet.apply(lambda x: x.duplicated().sum()))
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

x = diabet
y = diabet.pop('Outcome')

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42, stratify=y)
print(y_train.value_counts(normalize = True))
print(y_test.value_counts(normalize = True))


k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelLR  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelLR, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))



modelRF  = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(modelRF, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))



modelMPL  = MLPClassifier()
scoring = 'accuracy'
score = cross_val_score(modelMPL, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


modelNB  = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(modelNB, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

modelLR.fit(X_train, y_train)
y_pred_train = modelLR.predict(X_train)

train = pd.DataFrame({
    'y_train': y_train,
    'y_pred_train': y_pred_train
})
print(train)
print(train[train['y_train'] != train['y_pred_train']])

akurasi = accuracy_score(y_train, y_pred_train)
print(akurasi)
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred_train')
plt.ylabel('y_train')
plt.title('confusion matrix')
plt.show()




y_pred_test = modelLR.predict(X_test)

train = pd.DataFrame({
    'y_test': y_test,
    'y_pred_test': y_pred_test
})
print(train)
print(train[train['y_test'] != train['y_pred_test']])

akurasi = accuracy_score(y_test, y_pred_test)
print(akurasi)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred_test')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()