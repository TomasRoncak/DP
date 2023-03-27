import sys
from pathlib import Path

import numpy as np
import datetime
import pandas as pd

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
    data = data.loc[data['attack_cat'].isin(const.ATTACK_CATEGORIES)]  # Filter out not wanted attack categories

    data.drop(columns=const.USELESS_FEATURES_FOR_PARTIAL_CSVS, inplace=True)
    data.rename(columns=lambda x: x.lower(), inplace=True) 
    data.to_csv(const.WHOLE_DATASET_PATH, index=False)


def split_whole_dataset():
    whole_df = pd.read_csv(const.WHOLE_DATASET_PATH)
    whole_df = whole_df.drop_duplicates()
    whole_df.drop('service', axis=1, inplace=True)
    time_column = whole_df.pop(const.TIME)   # Move column to first place in df
    whole_df.insert(0, const.TIME, time_column)

    normal_data = whole_df[whole_df['attack_cat'] == 'Normal'][::10]    # Reduce normal traffic by 10
    whole_df = whole_df[whole_df.attack_cat != 'Normal']
    whole_df = pd.concat([whole_df, normal_data])

    whole_train_val_data = pd.DataFrame()
    whole_test_data = pd.DataFrame()

    for cat in const.ATTACK_CATEGORIES:
        if cat == 'All':
            continue
        cat_data = whole_df[whole_df['attack_cat'] == cat]
        train_val_data = cat_data[::2]
        test_data = cat_data[1::2]

        whole_train_val_data = pd.concat([whole_train_val_data, train_val_data])
        whole_test_data = pd.concat([whole_test_data, test_data])

    whole_train_val_data.drop(const.TIME, axis=1, inplace=True)
    whole_train_val_data.to_csv(const.CAT_TRAIN_VAL_DATASET, index=False)
    whole_test_data.to_csv(const.CAT_TEST_DATASET, index=False)


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