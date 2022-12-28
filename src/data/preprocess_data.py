import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const


def preprocess_whole_data():
    # process whole dataset
    Path(const.DATA_FOLDER + const.PREPROCESSED_CATEGORY_FOLDER).mkdir(exist_ok=True)
    data = pd.concat(map(pd.read_csv, [const.UNPROCESSED_PARTIAL_CSV.format(1), 
                                       const.UNPROCESSED_PARTIAL_CSV.format(2), 
                                       const.UNPROCESSED_PARTIAL_CSV.format(3), 
                                       const.UNPROCESSED_PARTIAL_CSV.format(4)]), ignore_index=True)

    data[const.TIME] = pd.to_datetime(data['Stime'], unit='s')
    data['tc_flw_http_mthd'].fillna(value = data.tc_flw_http_mthd.mean(), inplace = True)
    data.rename(columns = {'tc_flw_http_mthd':'ct_flw_http_mthd'}, inplace=True)

    data['is_ftp_login'].fillna(value = data.is_ftp_login.mean(), inplace = True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])

    data['attack_cat'].fillna('Normal', inplace=True)
    data["attack_cat"].replace('Backdoors', 'Backdoor', inplace=True)
    data['attack_cat'] = data['attack_cat'].str.strip()

    data.drop(columns=const.USELESS_FEATURES_FOR_PARTIAL_CSVS, inplace=True)
    data.to_csv(const.WHOLE_DATASET, index=False)
    

def preprocess_cat_data(dataset_type):
    # preprocess train or test data
    Path(const.PREPROCESSED_CAT_FOLDER).mkdir(parents=True, exist_ok=True)

    if dataset_type == 'train':
        data = pd.read_csv(const.UNPROCESSED_TRAINING_SET)
        PATH = const.CAT_TRAIN_DATASET
    elif dataset_type == 'test':
        data = pd.read_csv(const.UNPROCESSED_TESTING_SET)
        PATH = const.CAT_TEST_DATASET
    
    data["attack_cat"].fillna('Normal', inplace=True)
    data["attack_cat"].replace('Backdoors','Backdoor', inplace=True)
    data['attack_cat'] = data['attack_cat'].str.strip()

    data.drop(columns=const.USELESS_FEATURES_FOR_CATEGORIZE, inplace=True)
    data.to_csv(PATH, index=False)
    

# def clamp_numeric_data(df, cols):
#     for feature in cols:
#         max = df[feature].max()
#         median = df[feature].median()
#         quantile = df[feature].quantile(0.95)
#         if max > 10 and max > median * 10:
#             df[feature] = np.where(df[feature] < quantile, df[feature], quantile)

            
# def log_numeric_data(df, cols):
#     for feature in cols:
#         if df[feature].nunique() > 50:
#             if df[feature].min() == 0:
#                 df[feature] = np.log(df[feature] + 1)
#             else:
#                 df[feature] = np.log(df[feature])


def format_data(df):
    label_encoder = LabelEncoder()
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    standard_scaler = StandardScaler()

    if const.TIME in df:
        df = df.drop(const.TIME, axis=1)
    if 'service' in df:
        df = df.drop('service', axis=1)
    if 'Label' in df:
        df = df.drop('Label', axis=1)

    x = df.iloc[:,:-1]
    y = label_encoder.fit_transform(df.iloc[:,-1])

    x = minmax_scaler.fit_transform(x)
    x = standard_scaler.fit_transform(x)

    return x, y


def get_classes():
    return { 0: 'Analysis',
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
