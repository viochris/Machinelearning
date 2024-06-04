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

x = heart
y = heart.pop('target')


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

modelRF.fit(X_train, y_train)
y_pred_train = modelRF.predict(X_train)

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




y_pred_test = modelRF.predict(X_test)

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