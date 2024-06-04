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

df = pd.read_csv('ML/youtube/lat9-cancer/data.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df = df.drop('id', axis = 1)
del df['Unnamed: 32']

for col in df:
    print(col)
    print(df[col].nunique())
    print()

print(df['diagnosis'].unique())
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})


fig, ax = plt.subplots(6, 5, figsize=(20, 20))
# Menggunakan range dari 0 sampai 5 untuk iterasi setiap kolom dalam DataFrame
sns.scatterplot(x=df.index, y=df['diagnosis'], ax=ax[0, 0], color='blue')
sns.scatterplot(x=df.index, y=df['radius_mean'], ax=ax[0, 1], color='green')
sns.scatterplot(x=df.index, y=df['texture_mean'], ax=ax[0, 2], color='orange')
sns.scatterplot(x=df.index, y=df['perimeter_mean'], ax=ax[0, 3], color='red')
sns.scatterplot(x=df.index, y=df['area_mean'], ax=ax[0, 4], color='blue')
sns.scatterplot(x=df.index, y=df['smoothness_mean'], ax=ax[1, 0], color='green')
sns.scatterplot(x=df.index, y=df['compactness_mean'], ax=ax[1, 1], color='orange')
sns.scatterplot(x=df.index, y=df['concavity_mean'], ax=ax[1, 2], color='red')
sns.scatterplot(x=df.index, y=df['concave points_mean'], ax=ax[1, 3], color='blue')
sns.scatterplot(x=df.index, y=df['symmetry_mean'], ax=ax[1, 4], color='green')
sns.scatterplot(x=df.index, y=df['fractal_dimension_mean'], ax=ax[2, 0], color='orange')
sns.scatterplot(x=df.index, y=df['radius_se'], ax=ax[2, 1], color='red')
sns.scatterplot(x=df.index, y=df['texture_se'], ax=ax[2, 2], color='blue')
sns.scatterplot(x=df.index, y=df['perimeter_se'], ax=ax[2, 3], color='green')
sns.scatterplot(x=df.index, y=df['area_se'], ax=ax[2, 4], color='orange')
sns.scatterplot(x=df.index, y=df['smoothness_se'], ax=ax[3, 0], color='red')
sns.scatterplot(x=df.index, y=df['compactness_se'], ax=ax[3, 1], color='blue')
sns.scatterplot(x=df.index, y=df['concavity_se'], ax=ax[3, 2], color='green')
sns.scatterplot(x=df.index, y=df['concave points_se'], ax=ax[3, 3], color='orange')
sns.scatterplot(x=df.index, y=df['symmetry_se'], ax=ax[3, 4], color='red')
sns.scatterplot(x=df.index, y=df['fractal_dimension_se'], ax=ax[4, 0], color='blue')
sns.scatterplot(x=df.index, y=df['radius_worst'], ax=ax[4, 1], color='green')
sns.scatterplot(x=df.index, y=df['texture_worst'], ax=ax[4, 2], color='orange')
sns.scatterplot(x=df.index, y=df['perimeter_worst'], ax=ax[4, 3], color='red')
sns.scatterplot(x=df.index, y=df['area_worst'], ax=ax[4, 4], color='blue')
sns.scatterplot(x=df.index, y=df['smoothness_worst'], ax=ax[5, 0], color='green')
sns.scatterplot(x=df.index, y=df['compactness_worst'], ax=ax[5, 1], color='orange')
sns.scatterplot(x=df.index, y=df['concavity_worst'], ax=ax[5, 2], color='red')
sns.scatterplot(x=df.index, y=df['concave points_worst'], ax=ax[5, 3], color='blue')
sns.scatterplot(x=df.index, y=df['symmetry_worst'], ax=ax[5, 4], color='green')
plt.tight_layout()
plt.show()

def remove_outlier(df, col):
    # q1 = np.percentile(df[col], 25)
    # q3 = np.percentile(df[col], 75)
    # iqr = q3 - q1
    # df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr )]
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr )]
    return df


for col in df:
    df = remove_outlier(df, col)

fig, ax = plt.subplots(6, 5, figsize=(20, 20))
# Menggunakan range dari 0 sampai 5 untuk iterasi setiap kolom dalam DataFrame
sns.scatterplot(x=df.index, y=df['diagnosis'], ax=ax[0, 0], color='blue')
sns.scatterplot(x=df.index, y=df['radius_mean'], ax=ax[0, 1], color='green')
sns.scatterplot(x=df.index, y=df['texture_mean'], ax=ax[0, 2], color='orange')
sns.scatterplot(x=df.index, y=df['perimeter_mean'], ax=ax[0, 3], color='red')
sns.scatterplot(x=df.index, y=df['area_mean'], ax=ax[0, 4], color='blue')
sns.scatterplot(x=df.index, y=df['smoothness_mean'], ax=ax[1, 0], color='green')
sns.scatterplot(x=df.index, y=df['compactness_mean'], ax=ax[1, 1], color='orange')
sns.scatterplot(x=df.index, y=df['concavity_mean'], ax=ax[1, 2], color='red')
sns.scatterplot(x=df.index, y=df['concave points_mean'], ax=ax[1, 3], color='blue')
sns.scatterplot(x=df.index, y=df['symmetry_mean'], ax=ax[1, 4], color='green')
sns.scatterplot(x=df.index, y=df['fractal_dimension_mean'], ax=ax[2, 0], color='orange')
sns.scatterplot(x=df.index, y=df['radius_se'], ax=ax[2, 1], color='red')
sns.scatterplot(x=df.index, y=df['texture_se'], ax=ax[2, 2], color='blue')
sns.scatterplot(x=df.index, y=df['perimeter_se'], ax=ax[2, 3], color='green')
sns.scatterplot(x=df.index, y=df['area_se'], ax=ax[2, 4], color='orange')
sns.scatterplot(x=df.index, y=df['smoothness_se'], ax=ax[3, 0], color='red')
sns.scatterplot(x=df.index, y=df['compactness_se'], ax=ax[3, 1], color='blue')
sns.scatterplot(x=df.index, y=df['concavity_se'], ax=ax[3, 2], color='green')
sns.scatterplot(x=df.index, y=df['concave points_se'], ax=ax[3, 3], color='orange')
sns.scatterplot(x=df.index, y=df['symmetry_se'], ax=ax[3, 4], color='red')
sns.scatterplot(x=df.index, y=df['fractal_dimension_se'], ax=ax[4, 0], color='blue')
sns.scatterplot(x=df.index, y=df['radius_worst'], ax=ax[4, 1], color='green')
sns.scatterplot(x=df.index, y=df['texture_worst'], ax=ax[4, 2], color='orange')
sns.scatterplot(x=df.index, y=df['perimeter_worst'], ax=ax[4, 3], color='red')
sns.scatterplot(x=df.index, y=df['area_worst'], ax=ax[4, 4], color='blue')
sns.scatterplot(x=df.index, y=df['smoothness_worst'], ax=ax[5, 0], color='green')
sns.scatterplot(x=df.index, y=df['compactness_worst'], ax=ax[5, 1], color='orange')
sns.scatterplot(x=df.index, y=df['concavity_worst'], ax=ax[5, 2], color='red')
sns.scatterplot(x=df.index, y=df['concave points_worst'], ax=ax[5, 3], color='blue')
sns.scatterplot(x=df.index, y=df['symmetry_worst'], ax=ax[5, 4], color='green')
plt.tight_layout()
plt.show()


print(df.head())
numvars = df.drop('diagnosis', axis = 1).columns
scaler = MinMaxScaler()
df[numvars] = scaler.fit_transform(df[numvars])
print(df.head())

x = df
y = df.pop('diagnosis')


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