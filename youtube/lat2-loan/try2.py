import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sympy import false

# print(os.getcwd())
# print(os.listdir())
warnings.filterwarnings('ignore')

loan_data = pd.read_csv('ML/youtube/lat2-loan/lc_2016_2017.csv')
print(loan_data.head())
print(loan_data.shape)
print(loan_data.info())
print(loan_data.isnull().sum())

print(loan_data.loan_status.unique())
loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)']),
    1,0
)
print(loan_data.head())
print(loan_data.good_bad.value_counts())
print(loan_data.good_bad.value_counts(normalize = True))

# null_data = pd.DataFrame(loan_data.isnull().sum() / loan_data.shape[0] *100)
null_data = pd.DataFrame(loan_data.isnull().sum() / len(loan_data) *100)
null_data = null_data[null_data.iloc[:,0] > 50].sort_values(0, ascending=False)
print(null_data)

# loan_data = loan_data.dropna(thresh=len(loan_data) * 0.5, axis = 1)
# loan_data = loan_data.dropna(thresh=loan_data.shape[0] * 0.5, axis = 1)
# loan_data = loan_data.dropna(thresh=len(loan_data) /2, axis = 1)
loan_data = loan_data.dropna(thresh=loan_data.shape[0] /2, axis = 1)

null_data = pd.DataFrame(loan_data.isnull().sum() / loan_data.shape[0] * 100, columns=['nullpct'])
null_data = null_data[null_data['nullpct'] > 50].sort_values('nullpct', ascending=False)
# null_data = null_data[null_data.loc[:, 'nullpct'] > 50].sort_values('nullpct', ascending=False)
print(null_data)

x = loan_data.drop('good_bad', axis = 1)
y = loan_data['good_bad']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)
print(y_train.value_counts(normalize = True))
print(y_test.value_counts(normalize = True))
print(X_train.dtypes)

# for col in X_train.select_dtypes(include = ['object', 'bool']):
#     print(col)
#     print(X_train[col])
#     print()

for col in X_train.select_dtypes(include = ['object', 'bool']).columns:
    print(col)
    print(X_train[col])
    print()
    
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
print(X_train[col_need_to_clean].head())

X_train['term'] = pd.to_numeric(X_train['term'].str.replace(' months', ''))

print(X_train['emp_length'].unique())
X_train['emp_length'] = X_train['emp_length'].str.replace('+ years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', '0')
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')
X_train['emp_length'] = X_train['emp_length'].fillna(0)
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])


print(X_train[col_need_to_clean].head())

penanggalan = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
for tanggal in penanggalan:
    X_train[tanggal] = pd.to_datetime(X_train[tanggal])

print(X_train[penanggalan].head())
print(X_train[col_need_to_clean].head())

print('\n\n\n\n\n\n')
print(X_test[col_need_to_clean].head())

X_test['term'] = pd.to_numeric(X_test['term'].str.replace(' months', ''))

print(X_test['emp_length'].unique())
X_test['emp_length'] = X_test['emp_length'].str.replace('+ years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', '0')
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')
X_test['emp_length'] = X_test['emp_length'].fillna(0)
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])

for tanggal in penanggalan:
    X_test[tanggal] = pd.to_datetime(X_test[tanggal])
    
print(X_train[penanggalan].head())
print(X_train[col_need_to_clean].head())
print(X_test[penanggalan].head())
print(X_test[col_need_to_clean].head())
print(X_train[col_need_to_clean].info())
print(X_test[col_need_to_clean].info())

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(col_need_to_clean)

X_train = X_train[col_need_to_clean]
del X_train['next_pymnt_d']
X_test = X_test[col_need_to_clean]
X_test = X_test.drop('next_pymnt_d', axis = 1)

print(X_train.head())
print(X_test.head())
print(X_train.shape)
print(X_test.shape)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print()

print(date.today().strftime('%Y-%m-%d'))
def date_columns(df, col):
    today = pd.to_datetime(date.today().strftime('%Y-%m-%d'))
    df[col] = pd.to_datetime(df[col], format = '%b-%y')
    df['mth since-' + col] = round(pd.to_numeric((today - df[col]) / np.timedelta64(30, 'D')))
    df.drop(col,axis = 1, inplace = True)
    
date_columns(X_train, 'issue_d') 
date_columns(X_train, 'earliest_cr_line')
date_columns(X_train, 'last_pymnt_d')
date_columns(X_train, 'last_credit_pull_d')

date_columns(X_test, 'issue_d')
date_columns(X_test, 'earliest_cr_line')
date_columns(X_test, 'last_pymnt_d')
date_columns(X_test, 'last_credit_pull_d')

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

print(X_train.head())
print(X_test.head())
print(X_train.shape)
print(X_test.shape)
print(X_train.isnull().sum())
print(X_test.isnull().sum())


k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
model  = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(model, X_train, y_train, cv = k_folds, n_jobs=1, scoring=scoring)
print(score)
print(round(score.mean(), 2))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)



result = pd.DataFrame({
    'y_pred': y_pred,
    'y_test': y_test
})
print(result)

accuracy = accuracy_score(y_test, y_pred)
print('akurasi: {}'.format(accuracy))

cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()

y_pred = model.predict_proba(X_test)
print(y_pred)
# y_pred = model.predict_proba(X_test)[:,0]
# print('Yang Bagus')
# print(y_pred)
# print(y_pred > 0.5)
# print((y_pred > 0.5).astype(int))
# print((y_pred > 0.066).astype(int)) 
# print((y_pred < 0.066).astype(int))
y_pred = model.predict_proba(X_test)[:,1]
print('Yang Jelek')
print(y_pred)
print(y_pred > 0.5)
print((y_pred > 0.5).astype(int))
print((y_pred > 0.066).astype(int))
print((y_pred > 0.22).astype(int))
# print((y_pred < 0.066).astype(int))

plt.hist(y_pred)
plt.show()


print()
print()
fpr,tpr,thresholds = roc_curve(y_test, y_pred)
print(fpr)
print(tpr)
print(thresholds)
print()

# youden j-statistic
j = tpr - fpr
print(j)
ix = np.argmax(j)
print(ix)
best_thresh = thresholds[ix]
print(best_thresh)

y_pred = model.predict_proba(X_test)[:,1]
y_pred = (y_pred > 0.066).astype(int)
print('y_pred')
print(y_pred)
result = pd.DataFrame({
    'y_pred': y_pred,
    'y_test': y_test
})
print(result)
accuracy = accuracy_score(y_test, y_pred)
print('akurasi: {}'.format(accuracy))
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True,  fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('confusion matrix')
plt.show()

print(model.coef_)
print(model.intercept_)

df_coeff = pd.DataFrame(model.coef_)
print(df_coeff)
df_coeff = pd.DataFrame(model.coef_, columns=X_train.columns)
print(df_coeff)
