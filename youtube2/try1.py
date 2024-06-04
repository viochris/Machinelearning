import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split

from sqlalchemy import column

warnings.filterwarnings('ignore')

# print(os.getcwd())
# print()
# print(os.listdir())


loan_data = pd.read_csv('Praktek/YouTube2/lc_2016_2017.csv')
print(loan_data.head())
print(loan_data.shape)
print(loan_data.shape[0])
print(loan_data.shape[1])
print(loan_data.info())
print(loan_data.isnull().sum())

print(loan_data['loan_status'].unique())
loan_data['Good Bad'] = np.where(
    loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)']),
    1,0
)
# loan_data['Good Bad2'] = np.select(
#     [loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)'])],
#     [1],
#     default=[0]
# )
print(loan_data.head())
print(loan_data['Good Bad'].value_counts())
# print(loan_data['Good Bad2'].value_counts())
print(loan_data['Good Bad'].value_counts(normalize=True))
# print(loan_data['Good Bad2'].value_counts(normalize=True))

# missing_values = pd.DataFrame((loan_data.isnull().sum() / len(loan_data))*100)
# missing_values = pd.DataFrame((loan_data.isnull().sum() / loan_data.shape[0])*100)
# missing_values = pd.DataFrame((loan_data.isnull().mean()])*100)
# print(missing_values)
# missing_values = missing_values[missing_values[0] > 50]
# missing_values = missing_values.sort_values(0, ascending=False)
# print(missing_values)

# missing_values = pd.DataFrame((loan_data.isnull().sum() / len(loan_data))*100, columns = ['nullpct'])
missing_values = pd.DataFrame((loan_data.isnull().sum() / loan_data.shape[0])*100, columns=['nullpct'])
# missing_values = pd.DataFrame((loan_data.isnull().mean())*100, columns=['nullpct'])
print(missing_values)
missing_values = missing_values[missing_values['nullpct'] > 50]
missing_values = missing_values.sort_values('nullpct', ascending=False)
print(missing_values)

# Drop feature tersebut
loan_data = loan_data.dropna(thresh = loan_data.shape[0]*0.5, axis=1)
print(loan_data.shape[0]*0.5)


# loan_data = loan_data.drop(['member_id', 'desc', 'dti_joint', 'annual_inc_joint', 'verification_status_joint', 'mths_since_last_record', 'mths_since_last_major_derog'], axis=1)
# loan_data = loan_data.drop('member_id', axis=1)

# Pengecheckan ulang apakah feature tersebut berhasil di drop
missing_values = pd.DataFrame((loan_data.isnull().sum() / loan_data.shape[0])*100, columns=['nullpct'])
missing_values = missing_values[missing_values['nullpct'] > 50]
missing_values = missing_values.sort_values('nullpct', ascending=False)
print(missing_values)
print(loan_data)
print(loan_data.head())



X = loan_data.drop('Good Bad', axis=1)
y = loan_data['Good Bad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify=y, random_state=42)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))