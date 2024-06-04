from doctest import testmod
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


"""
Jadi, secara umum, dict encoding cocok untuk variabel dengan tingkatan 
atau variabel target biner, sedangkan one-hot encoding lebih cocok untuk 
variabel kategori tanpa tingkatan atau variabel dengan lebih dari dua kategori. 
Namun, pilihan encoding tergantung pada konteks spesifik dari dataset dan tujuan analisisnya."""

test = pd.read_csv('Praktek/Youtube/test.csv')
train = pd.read_csv('Praktek/Youtube/train.csv')

print(train.head())
print()
print(test.head())
print()


train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Sex'] = train['Sex'].map({'male':0, 'female':1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})

data_train_embarked = pd.get_dummies(train[['Embarked']])
train = pd.merge(train.reset_index(), data_train_embarked.reset_index())
train = train.drop(['index', 'Embarked'], axis=1)
data_test_embarked = pd.get_dummies(test[['Embarked']])
test = pd.merge(test.reset_index(), data_test_embarked.reset_index())
test = test.drop(['index', 'Embarked'], axis=1)

train['title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                    "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                    "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3}
train['title'] = train['title'].map(title_mapping)
test['title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['title'] = test['title'].map(title_mapping)

train = train.drop(['Name', 'PassengerId', 'Ticket'], axis=1)
test = test.drop(['Name', 'PassengerId', 'Ticket'], axis=1)

print(train.shape, test.shape)
print()
print(train.info())
print()
print(test.info())
print()
print(train.head())
print()
print(test.head())
print()


train_data = train.drop('Survived', axis=1)
target = train['Survived']

# k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
# clf = GaussianNB()
# scoring = 'accuracy'
# score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
# print(score)
# print(round(score.mean(), 2))
# print()


# k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
# clf = DecisionTreeClassifier()
# scoring = 'accuracy'
# score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
# print(score)
# print(round(score.mean(), 2))

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
clf = KNeighborsClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))




model.fit(train_data, target)
prediction = model.predict(test)
print(prediction)


df_test = pd.read_csv('Praktek/Youtube/test.csv')
hasil = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': prediction
})
print(hasil.head())

titanic = pd.read_csv('Praktek/Youtube/test.csv')
titanic['Survived'] = prediction
print(titanic.head())

test['Survived'] = hasil['Survived']
print(test.head())


hasil.to_csv('Praktek/Youtube/submission_more_true4.csv', index=False)
# titanic.to_csv('Praktek/Youtube/titanic_more_true2.csv', index=False)
# test.to_csv('Praktek/Youtube/test_survived_more_true2.csv', index=False)