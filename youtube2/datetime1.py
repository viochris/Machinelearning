import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split
from datetime import date, timedelta

print(os.getcwd())
print(os.listdir())
warnings.filterwarnings('ignore')

loan_data = pd.read_csv('Praktek/YouTube2/lc_2016_2017.csv')
print(loan_data.head())
print(loan_data.shape)
print(loan_data.info())
print(loan_data.isnull().sum())

print(loan_data['loan_status'].unique())
loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)']),
    1,0
)
print(loan_data.head())
print(loan_data['good_bad'].value_counts())
print(loan_data['good_bad'].value_counts(normalize=True))

null_data = pd.DataFrame((loan_data.isnull().sum()/loan_data.shape[0])*100, columns = ['nullpct'])
null_data = null_data[null_data.iloc[:, 0] > 50]
# null_data = null_data[null_data.loc[:, 'nullpct'] > 50]
null_data = null_data.sort_values('nullpct', ascending=False)
print(null_data)

loan_data = loan_data.dropna(thresh=loan_data.shape[0]*0.5, axis=1)
print(loan_data.shape[0]*0.5)

print()
null_data = pd.DataFrame((loan_data.isnull().sum()/loan_data.shape[0])*100)
# null_data = null_data[null_data.iloc[:, 0] > 50]
null_data = null_data[null_data.loc[:, 0] > 50]
null_data = null_data.sort_values(0, ascending=False)
print(null_data)

x = loan_data.drop('good_bad', axis=1)
y = loan_data['good_bad']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


print(X_train.head())
print(X_train.dtypes)
print(X_train.info())

for col in X_train.select_dtypes(include=['object', 'bool']).columns:
    print(col)
    print(X_train[col].unique())
    print()
    
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
print(X_train[col_need_to_clean].head())

X_train['term'] = pd.to_numeric(X_train['term'].str.replace(' months', ''))

print(X_train['emp_length'].unique())
print()
X_train['emp_length'] = X_train['emp_length'].str.replace('+ years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', str(0))
# X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', '0')
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')
X_train['emp_length'] = X_train['emp_length'].fillna(0)
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])

print(X_train['emp_length'].unique())


penanggalan = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
print(X_train[penanggalan].head())
for tanggal in penanggalan:
    X_train[tanggal] = pd.to_datetime(X_train[tanggal])

print(X_train[penanggalan].head())
print(X_train[col_need_to_clean].head())





print()
print()
print(X_test[col_need_to_clean].head())

X_test['term'] = pd.to_numeric(X_test['term'].str.replace(' months', ''))

print(X_test['emp_length'].unique())
print()
X_test['emp_length'] = X_test['emp_length'].str.replace('+ years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', str(0))
# X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', '0')
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')
X_test['emp_length'] = X_test['emp_length'].fillna(0)
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])

print(X_test['emp_length'].unique())
penanggalan = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
print(X_test[penanggalan].head())
print(X_test[penanggalan].dtypes)
# for tanggal in penanggalan:
#     X_test[tanggal] = pd.to_datetime(X_test[tanggal])
X_test['issue_d'] = pd.to_datetime(X_test['issue_d'])
print(X_test[penanggalan].head())
print(X_test[penanggalan].dtypes)