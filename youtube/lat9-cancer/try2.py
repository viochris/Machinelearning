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


# fig, ax = plt.subplots(6, 5, figsize=(20, 20))
# # Menggunakan range dari 0 sampai 5 untuk iterasi setiap kolom dalam DataFrame
# sns.scatterplot(x=df.index, y=df['diagnosis'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['radius_mean'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=df.index, y=df['texture_mean'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['perimeter_mean'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=df.index, y=df['area_mean'], ax=ax[0, 4], color='blue')
# sns.scatterplot(x=df.index, y=df['smoothness_mean'], ax=ax[1, 0], color='green')
# sns.scatterplot(x=df.index, y=df['compactness_mean'], ax=ax[1, 1], color='orange')
# sns.scatterplot(x=df.index, y=df['concavity_mean'], ax=ax[1, 2], color='red')
# sns.scatterplot(x=df.index, y=df['concave points_mean'], ax=ax[1, 3], color='blue')
# sns.scatterplot(x=df.index, y=df['symmetry_mean'], ax=ax[1, 4], color='green')
# sns.scatterplot(x=df.index, y=df['fractal_dimension_mean'], ax=ax[2, 0], color='orange')
# sns.scatterplot(x=df.index, y=df['radius_se'], ax=ax[2, 1], color='red')
# sns.scatterplot(x=df.index, y=df['texture_se'], ax=ax[2, 2], color='blue')
# sns.scatterplot(x=df.index, y=df['perimeter_se'], ax=ax[2, 3], color='green')
# sns.scatterplot(x=df.index, y=df['area_se'], ax=ax[2, 4], color='orange')
# sns.scatterplot(x=df.index, y=df['smoothness_se'], ax=ax[3, 0], color='red')
# sns.scatterplot(x=df.index, y=df['compactness_se'], ax=ax[3, 1], color='blue')
# sns.scatterplot(x=df.index, y=df['concavity_se'], ax=ax[3, 2], color='green')
# sns.scatterplot(x=df.index, y=df['concave points_se'], ax=ax[3, 3], color='orange')
# sns.scatterplot(x=df.index, y=df['symmetry_se'], ax=ax[3, 4], color='red')
# sns.scatterplot(x=df.index, y=df['fractal_dimension_se'], ax=ax[4, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['radius_worst'], ax=ax[4, 1], color='green')
# sns.scatterplot(x=df.index, y=df['texture_worst'], ax=ax[4, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['perimeter_worst'], ax=ax[4, 3], color='red')
# sns.scatterplot(x=df.index, y=df['area_worst'], ax=ax[4, 4], color='blue')
# sns.scatterplot(x=df.index, y=df['smoothness_worst'], ax=ax[5, 0], color='green')
# sns.scatterplot(x=df.index, y=df['compactness_worst'], ax=ax[5, 1], color='orange')
# sns.scatterplot(x=df.index, y=df['concavity_worst'], ax=ax[5, 2], color='red')
# sns.scatterplot(x=df.index, y=df['concave points_worst'], ax=ax[5, 3], color='blue')
# sns.scatterplot(x=df.index, y=df['symmetry_worst'], ax=ax[5, 4], color='green')
# plt.tight_layout()
# plt.show()

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

# fig, ax = plt.subplots(6, 5, figsize=(20, 20))
# # Menggunakan range dari 0 sampai 5 untuk iterasi setiap kolom dalam DataFrame
# sns.scatterplot(x=df.index, y=df['diagnosis'], ax=ax[0, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['radius_mean'], ax=ax[0, 1], color='green')
# sns.scatterplot(x=df.index, y=df['texture_mean'], ax=ax[0, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['perimeter_mean'], ax=ax[0, 3], color='red')
# sns.scatterplot(x=df.index, y=df['area_mean'], ax=ax[0, 4], color='blue')
# sns.scatterplot(x=df.index, y=df['smoothness_mean'], ax=ax[1, 0], color='green')
# sns.scatterplot(x=df.index, y=df['compactness_mean'], ax=ax[1, 1], color='orange')
# sns.scatterplot(x=df.index, y=df['concavity_mean'], ax=ax[1, 2], color='red')
# sns.scatterplot(x=df.index, y=df['concave points_mean'], ax=ax[1, 3], color='blue')
# sns.scatterplot(x=df.index, y=df['symmetry_mean'], ax=ax[1, 4], color='green')
# sns.scatterplot(x=df.index, y=df['fractal_dimension_mean'], ax=ax[2, 0], color='orange')
# sns.scatterplot(x=df.index, y=df['radius_se'], ax=ax[2, 1], color='red')
# sns.scatterplot(x=df.index, y=df['texture_se'], ax=ax[2, 2], color='blue')
# sns.scatterplot(x=df.index, y=df['perimeter_se'], ax=ax[2, 3], color='green')
# sns.scatterplot(x=df.index, y=df['area_se'], ax=ax[2, 4], color='orange')
# sns.scatterplot(x=df.index, y=df['smoothness_se'], ax=ax[3, 0], color='red')
# sns.scatterplot(x=df.index, y=df['compactness_se'], ax=ax[3, 1], color='blue')
# sns.scatterplot(x=df.index, y=df['concavity_se'], ax=ax[3, 2], color='green')
# sns.scatterplot(x=df.index, y=df['concave points_se'], ax=ax[3, 3], color='orange')
# sns.scatterplot(x=df.index, y=df['symmetry_se'], ax=ax[3, 4], color='red')
# sns.scatterplot(x=df.index, y=df['fractal_dimension_se'], ax=ax[4, 0], color='blue')
# sns.scatterplot(x=df.index, y=df['radius_worst'], ax=ax[4, 1], color='green')
# sns.scatterplot(x=df.index, y=df['texture_worst'], ax=ax[4, 2], color='orange')
# sns.scatterplot(x=df.index, y=df['perimeter_worst'], ax=ax[4, 3], color='red')
# sns.scatterplot(x=df.index, y=df['area_worst'], ax=ax[4, 4], color='blue')
# sns.scatterplot(x=df.index, y=df['smoothness_worst'], ax=ax[5, 0], color='green')
# sns.scatterplot(x=df.index, y=df['compactness_worst'], ax=ax[5, 1], color='orange')
# sns.scatterplot(x=df.index, y=df['concavity_worst'], ax=ax[5, 2], color='red')
# sns.scatterplot(x=df.index, y=df['concave points_worst'], ax=ax[5, 3], color='blue')
# sns.scatterplot(x=df.index, y=df['symmetry_worst'], ax=ax[5, 4], color='green')
# plt.tight_layout()
# plt.show()


print(df.head())
numvars = df.drop('diagnosis', axis = 1).columns
scaler = MinMaxScaler()
df[numvars] = scaler.fit_transform(df[numvars])
print(df.head())

df_train, df_test = train_test_split(df, train_size = 0.7, test_size=0.3, random_state=100)
print(df_train.head())
print(df_test.head())

corr = df_train.corr()
print(corr)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr, annot = True, cmap="YlGnBu")
# plt.show()

x_train = df_train
y_train = df_train.pop('diagnosis')


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
k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
modelLR  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(modelLR, x_train_rfe, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))
modelLR.fit(x_train_rfe, y_train)


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

y_pred = modelLR.predict(x_train_rfe)

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
y_test = df_test.pop('diagnosis')
y_pred = modelLR.predict(x_test_rfe)


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