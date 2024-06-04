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

# q1 = df['fixed acidity'].quantile(0.25)
# q3 = df['fixed acidity'].quantile(0.75)
# iqr = q3 - q1
# df = df[(df['fixed acidity'] >= q1 - 1.5*iqr) & (df['fixed acidity'] <= q3 + 1.5*iqr )]


print(df['Species'].unique())
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':2})
print(df.head())

numvars = ['Sepal Length (cm)', 'Sepal Width (cm)',  'Petal Length (cm)',  'Petal Width (cm)']
scaler = MinMaxScaler()
df[numvars] = scaler.fit_transform(df[numvars])

x = df
y = df.pop('Species')

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42, stratify=y)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelLR  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelLR, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelRF  = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(modelRF, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelMPL  = MLPClassifier()
scoring = 'accuracy'
score = cross_val_score(modelMPL, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelNB  = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(modelNB, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

modelNB.fit(X_train, y_train)
y_pred = modelNB.predict(X_test)

df_pred = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred
})
print(df_pred)
df_pred = df_pred[df_pred['y_test'] != df_pred['y_pred']]
print(df_pred)

akurasi = accuracy_score(y_test, y_pred)
print(akurasi)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()