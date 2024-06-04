import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

train = pd.read_csv('ML/youtube/lat1-titanic/train.csv')
test = pd.read_csv('ML/youtube/lat1-titanic/test.csv')

train = train.drop('Cabin', axis = 1)
test = test.drop('Cabin', axis = 1)

print(train[train['Embarked'].isnull()])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
print(train[train.index == 61])

print(test[test['Fare'].isnull()])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
# print(test[test.index == 152])
# print(test.iloc[152])
print(test.loc[152])


train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Sex'] = train['Sex'].map({'male': 0, 'female':1})
test['Sex'] = test['Sex'].map({'male': 0, 'female':1})

train_embarked = pd.get_dummies(train['Embarked'])
test_embarked = pd.get_dummies(test['Embarked'])
print(train_embarked)
print(test_embarked)

train = pd.merge(train.reset_index(), train_embarked.reset_index())
# train = train.drop(['index', 'Embarked'], axis = 1)
del train['index']
del train['Embarked']
test = pd.concat([test, test_embarked], axis = 1)
del test['Embarked']


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                    "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                    "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3}
train['title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['title'] = train['title'].map(title_mapping)
test['title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['title'] = test['title'].map(title_mapping)

train = train.drop(['Name', 'Ticket', 'PassengerId'], axis = 1)
test = test.drop(['Name', 'Ticket', 'PassengerId'], axis = 1)

print('\n\n\n\n')
print(train.head())
print(test.head())
print(train.info())
print(test.info())
print(train.shape, test.shape)
print(train.isnull().sum())
print(test.isnull().sum())

train_data = train.drop('Survived', axis = 1)
target = train['Survived']


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
clf = MLPClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
model = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(model, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
knc = KNeighborsClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
svc = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))


model.fit(train_data, target)
y_pred = model.predict(test)
print(y_pred)


df_test = pd.read_csv('ML/youtube/lat1-titanic/test.csv')
df = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred
})
print(df)