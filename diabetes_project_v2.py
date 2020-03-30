#!/usr/bin/python
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

def prepare_data():
    if os.path.exists("df_train.csv") \
        and os.path.exists("df_valid.csv") \
        and os.path.exists("df_test.csv") \
        and os.path.exists("df_train_all.csv") \
        and os.path.exists("cols.txt"):

        df_train = pd.read_csv("df_train.csv")
        df_train_all = pd.read_csv("df_train_all.csv")
        df_valid = pd.read_csv("df_valid.csv")
        df_test = pd.read_csv("df_test.csv")
        with open("cols.txt") as f:
            col2use = eval(f.read())

        return df_train, df_train_all, df_valid, df_test, col2use

    df = pd.read_csv('diabetic_data.csv')
    print('Number of samples:',len(df))
    df.info()
    df.head()
    df.groupby('readmitted').size()
    df.groupby('discharge_disposition_id').size()
    df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]
    df['OUTPUT_LABEL'] = (df.readmitted == '<30').astype('int')

    def calc_prevalence(y_actual):
            return (sum(y_actual)/len(y_actual))

    print('Prevalence:%.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))
    print('Number of columns:',len(df.columns))

    df[list(df.columns)[:10]].head()
    df[list(df.columns)[10:20]].head()
    df[list(df.columns)[20:30]].head()
    df[list(df.columns)[30:40]].head()
    df[list(df.columns)[40:]].head()
    for c in list(df.columns):
            n = df[c].unique()
            if len(n)<30:
                    print(c)
                    print(n)
            else:
                    print(c + ': ' +str(len(n)) + ' unique values')
    df = df.replace('?',np.nan)

    cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient','number_diagnoses']

    df[cols_num].isnull().sum()
    cols_cat = ['race', 'gender',
         'max_glu_serum', 'A1Cresult',
         'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
         'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
         'tolbutamide',
         'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
         'troglitazone',
         'tolazamide', 'insulin',
         'glyburide-metformin', 'glipizide-metformin',
         'glimepiride-pioglitazone', 'metformin-rosiglitazone',
         'metformin-pioglitazone', 'change', 'diabetesMed','payer_code']

    df[cols_cat].isnull().sum()
    df['race'] = df['race'].fillna('UNK')
    df['payer_code'] = df['payer_code'].fillna('UNK')
    df['medical_specialty'] = df['medical_specialty'].fillna('UNK')
    print('Number medical specialty:', df.medical_specialty.nunique())
    df.groupby('medical_specialty').size().sort_values(ascending = False)
    top_10 = ['UNK','InternalMedicine','Emergency/Trauma',
        'Family/GeneralPractice', 'Cardiology','Surgery-General' ,
        'Nephrology','Orthopedics',
        'Orthopedics-Reconstructive','Radiologist']
    df['med_spec'] = df['medical_specialty'].copy()
    df.loc[~df.med_spec.isin(top_10),'med_spec'] = 'Other'
    df.groupby('med_spec').size()
    cols_cat_num = ['admission_type_id',
        'discharge_disposition_id', 'admission_source_id']
    df[cols_cat_num] = df[cols_cat_num].astype('str')
    df_cat = pd.get_dummies(
        df[cols_cat + cols_cat_num + ['med_spec']],drop_first = True)
    df_cat.head()
    df = pd.concat([df,df_cat], axis = 1)
    cols_all_cat = list(df_cat.columns)
    df[['age', 'weight']].head()
    df.groupby('age').size()
    age_id = {'[0-10)':0,
            '[10-20)':10,
            '[20-30)':20,
            '[30-40)':30,
            '[40-50)':40,
            '[50-60)':50,
            '[60-70)':60,
            '[70-80)':70,
            '[80-90)':80,
            '[90-100)':90}
    df['age_group'] = df.age.replace(age_id)
    df.weight.notnull().sum()
    df['has_weight'] = df.weight.notnull().astype('int')
    cols_extra = ['age_group','has_weight']
    print('Total number of features:',
        len(cols_num + cols_all_cat + cols_extra))
    print('Numerical Features:',len(cols_num))
    print('Categorical Features:',len(cols_all_cat))
    print('Extra features:',len(cols_extra))

    df[cols_num + cols_all_cat + cols_extra].isnull().sum().sort_values(
        ascending = False).head(10)
    col2use = cols_num + cols_all_cat + cols_extra
    df_data = df[col2use + ['OUTPUT_LABEL']]
    df_data = df_data.sample(n = len(df_data), random_state=42)
    df_data = df_data.reset_index(drop = True)

    df_valid_test=df_data.sample(frac=0.30, random_state=42)
    print('Split size: %.3f'%(len(df_valid_test)/len(df_data)))
    df_test = df_valid_test.sample(frac = 0.5, random_state=42)
    df_valid = df_valid_test.drop(df_test.index)
    df_train_all=df_data.drop(df_valid_test.index)

    print('Test prevalence(n = %d):%.3f'%(
        len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))
    print('Valid prevalence(n = %d):%.3f'%(
        len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
    print('Train all prevalence(n = %d):%.3f'%(
        len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))
    print('all samples (n = %d)'%len(df_data))
    assert len(df_data) == (
        len(df_test)+len(df_valid)+len(df_train_all)),'math didnt work'

    rows_pos = df_train_all.OUTPUT_LABEL == 1
    df_train_pos = df_train_all.loc[rows_pos]
    df_train_neg = df_train_all.loc[~rows_pos]
    df_train = pd.concat([df_train_pos, df_train_neg.sample(
        n = len(df_train_pos), random_state = 42)],axis = 0)
    df_train = df_train.sample(
        n = len(df_train), random_state = 42).reset_index(drop = True)

    print('Train balanced prevalence(n = %d):%.3f'%(len(df_train),
        calc_prevalence(df_train.OUTPUT_LABEL.values)))

    df_train_all.to_csv('df_train_all.csv',index=False)
    df_train.to_csv('df_train.csv',index=False)
    df_valid.to_csv('df_valid.csv',index=False)
    df_test.to_csv('df_test.csv',index=False)

    with open("cols.txt", 'wb') as f:
        f.write(str(col2use))

    return df_train, df_train_all, df_valid, df_test, col2use

def train():
    num_classes = 2

    df_train, df_train_all, df_valid, df_test, col2use = prepare_data()

    X_train = df_train_all[col2use].values
    X_valid = df_valid[col2use].values
    X_test = df_test[col2use].values

    y_train = df_train_all['OUTPUT_LABEL'].values
    y_valid = df_valid['OUTPUT_LABEL'].values
    y_test = df_test['OUTPUT_LABEL'].values

    print('Training All shapes:', X_train.shape, y_train.shape)
    print('Validation shapes:', X_valid.shape, y_valid.shape)
    print('Testing shapes:', X_test.shape, y_test.shape)

    scaler  = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # Neural network
    model = Sequential()
    model.add(Dense(64, input_dim=143, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    # import pickle
    # scalerfile = 'scaler.sav'
    # pickle.dump(scaler, open(scalerfile, 'wb'))

if __name__ == "__main__":
    train()
