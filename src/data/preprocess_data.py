import sys
from pathlib import Path

import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

def preprocess_whole_data():
    Path(const.DATA_FOLDER + const.PREPROCESSED_CATEGORY_FOLDER).mkdir(exist_ok=True)
    data = pd.concat(map(pd.read_csv, [const.UNPROCESSED_PARTIAL_CSV_PATH.format(1),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(2),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(3),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(4)]), ignore_index=True)

    data[const.TIME] = pd.to_datetime(data['Stime'], unit='s')
    data[const.TIME] = data[const.TIME].dt.floor('Min')
    
    januar_time = list(filter(lambda x: x.month == 1, data[const.TIME]))
    februar_time = list(filter(lambda x: x.month == 2, data[const.TIME]))
    februar_time = [time.replace(month=1, day=23) + datetime.timedelta(minutes=38) for time in februar_time]
    data[const.TIME] = januar_time + februar_time

    data['tc_flw_http_mthd'].fillna(value=data.tc_flw_http_mthd.mean(), inplace=True)
    data.rename(columns={'tc_flw_http_mthd': 'ct_flw_http_mthd'}, inplace=True)

    data['is_ftp_login'].fillna(value=data.is_ftp_login.mean(), inplace=True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])

    data['attack_cat'].fillna('Normal', inplace=True)
    data['attack_cat'].replace('Backdoors', 'Backdoor', inplace=True)
    data['attack_cat'] = data['attack_cat'].str.strip()

    data.drop(columns=const.USELESS_FEATURES_FOR_PARTIAL_CSVS, inplace=True)
    data.rename(columns=lambda x: x.lower(), inplace=True)
    data.to_csv(const.WHOLE_DATASET_PATH, index=False)


def preprocess_train_test_data():
    Path(const.PREPROCESSED_CAT_PATH).mkdir(parents=True, exist_ok=True)

    for dataset_type in ['train', 'test']:
        if dataset_type == 'train':
            data = pd.read_csv(const.UNPROCESSED_TRAINING_SET_PATH)
            PATH = const.CAT_TRAIN_DATASET_PATH
        elif dataset_type == 'test':
            data = pd.read_csv(const.UNPROCESSED_TESTING_SET_PATH)
            PATH = const.CAT_TEST_DATASET_PATH

        data["attack_cat"].fillna('Normal', inplace=True)
        data["attack_cat"].replace('Backdoors', 'Backdoor', inplace=True)
        data['attack_cat'] = data['attack_cat'].str.strip()

        data = data[~data.attack_cat.isin(const.TO_DELETE)]
        data.drop(columns=const.USELESS_FEATURES_FOR_CLASSIFICATION, inplace=True)
        data.rename(columns=lambda x: x.lower(), inplace=True) 
        data.to_csv(PATH, index=False)


def format_data(df, is_cat_multiclass, is_model_reccurent=False):
    label_encoder = LabelEncoder()
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    standard_scaler = StandardScaler()

    if const.TIME in df:  # whole_dataset.csv obsahuje aj cas pre potreby anomaly dat, tu ho netreba
        df = df.drop(const.TIME, axis=1)
    if 'service' in df:
        df = df.drop('service', axis=1)  # dataset whole_dataset.csv obsahuje 'service' na vytvaranie casovych radov
    if is_cat_multiclass:
        df = df.drop('label', axis=1)
    else:
        df = df.drop('attack_cat', axis=1)

    x = df.iloc[:, :-1]
    x = minmax_scaler.fit_transform(x)
    x = standard_scaler.fit_transform(x)

    y = label_encoder.fit_transform(df.iloc[:, -1])

    if is_model_reccurent:  # Reshape -> [samples, time steps, features]
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

    return x, y


def clamp_and_reduce_values(df):
    df_numeric = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(exclude=[np.number])

    for feature in df_numeric.columns:  # Clamp extreme Values
        if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
            df_numeric[feature] = np.where(df_numeric[feature] < df_numeric[feature].quantile(0.95), df_numeric[feature], df_numeric[feature].quantile(0.95))

    for feature in df_cat.columns:  # Reduce the labels in categorical features
        if df_cat[feature].nunique() > 6:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')


def get_classes():
    return {0: 'Analysis',
            1: 'Backdoor',
            2: 'DoS',
            3: 'Exploits',
            4: 'Fuzzers',
            5: 'Generic',
            6: 'Normal',
            7: 'Reconnaissance',
            8: 'Shellcode',
            9: 'Worms'
            }


def get_filtered_classes():
    return {0: 'DoS',
            1: 'Exploits',
            2: 'Fuzzers',
            3: 'Generic',
            4: 'Normal',
            5: 'Reconnaissance'
            }
