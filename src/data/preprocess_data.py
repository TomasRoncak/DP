import pandas as pd
import numpy as np
import sys

from os import path, makedirs
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

def preprocess_data():
    # process whole dataset
    if not path.exists(const.PREPROCESSED_DATASET_PATH):   
        makedirs(const.PREPROCESSED_DATASET_PATH)
    
    data = pd.concat(map(pd.read_csv, [const.RAW_DATASET_PATH + 'UNSW-NB15_1.csv', 
                                       const.RAW_DATASET_PATH + 'UNSW-NB15_2.csv', 
                                       const.RAW_DATASET_PATH + 'UNSW-NB15_3.csv', 
                                       const.RAW_DATASET_PATH + 'UNSW-NB15_4.csv']), ignore_index=True)

    data['time'] = pd.to_datetime(data['Stime'], unit='s')
    data['tc_flw_http_mthd'].fillna(value = data.tc_flw_http_mthd.mean(), inplace = True)
    data.rename(columns = {'tc_flw_http_mthd':'ct_flw_http_mthd'}, inplace=True)

    data['is_ftp_login'].fillna(value = data.is_ftp_login.mean(), inplace = True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])

    data["attack_cat"].replace('Backdoors','Backdoor', inplace=True)
    data['attack_cat'].fillna('Normal', inplace=True)
    data['attack_cat'] = data['attack_cat'].str.strip()

    data.drop(columns=const.USELESS_FEATURES, inplace=True)
    data.to_csv(const.WHOLE_DATASET, index=False)
    

def preprocess_cat_data(dataset_type):
    # preprocess train or test data
    if dataset_type == 'train':
        data = pd.read_csv(const.RAW_DATASET_PATH + 'UNSW_NB15_training-set.csv')
        PATH = const.WHOLE_CAT_TRAIN_DATASET
    elif dataset_type == 'test':
        data = pd.read_csv(const.RAW_DATASET_PATH + 'UNSW_NB15_testing-set.csv')
        PATH = const.WHOLE_CAT_TEST_DATASET
    
    data.drop(columns=['id', 'label', 'ct_ftp_cmd', 'rate'], axis=1, inplace=True)
    data["attack_cat"].fillna('Normal', inplace=True)
    data["attack_cat"].replace('Backdoors','Backdoor', inplace=True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    cat_cols = data.select_dtypes(exclude=[np.number]).columns
    
    clamp_numeric_data(data, numeric_cols)
    log_numeric_data(data, numeric_cols)
    reduce_cat_labels(data, cat_cols)

    data.to_csv(PATH, index=False)
    

def clamp_numeric_data(df, cols):
    for feature in cols:
        max = df[feature].max()
        median = df[feature].median()
        quantile = df[feature].quantile(0.95)
        if max > 10 and max > median * 10:
            df[feature] = np.where(df[feature] < quantile, df[feature], quantile)

            
def log_numeric_data(df, cols):
    for feature in cols:
        if df[feature].nunique() > 50:
            if df[feature].min() == 0:
                df[feature] = np.log(df[feature] + 1)
            else:
                df[feature] = np.log(df[feature])

                
def reduce_cat_labels(df, cols):
    for feature in cols:  # proto and service reduce to 10 labels
        if feature in ['time', 'attack_cat']:
            continue
        if df[feature].nunique() > 10:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')


def format_data(df):
    # one hot encoding categorical data, scaling and normalizing numerical data
    label_encoder = LabelEncoder()
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    standard_scaler = StandardScaler()

    if 'time' in df:
        df = df.drop('time', axis=1)

    x = df.iloc[:,:-1]
    y = label_encoder.fit_transform(df.iloc[:,-1])

    x['service'] = label_encoder.fit_transform(x['service'])
    x['proto'] = label_encoder.fit_transform(x['proto'])
    x['state'] = label_encoder.fit_transform(x['state'])

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    x[numeric_cols] = minmax_scaler.fit_transform(x[numeric_cols])
    x[numeric_cols] = standard_scaler.fit_transform(x[numeric_cols])

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
            8: 'Sheelcode',
            9: 'Worms'  
            }
