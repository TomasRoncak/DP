import json
import sys

from pathlib import Path
import random
import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import config as conf
import constants as const

"""
creates dataset suitable for training according to extracted features (on data without attacks)

:param window_size: number specifying which dataset to use according to window size
"""
def merge_features_to_dataset(window_size, with_attacks):
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))

    DATASET_PATH = const.EXTRACTED_ATTACK_DATASET_PATH if with_attacks else const.EXTRACTED_BENIGN_DATASET_PATH
    TIME_PATH = const.TS_BENIGN_DATASET_PATH.format(window_size, list(protocol_features.keys())[0])

    time = pd.read_csv(TIME_PATH, usecols = [const.TIME], squeeze=True).apply(lambda x: x[:-2] + '00')  # delete seconds from time
    data = pd.DataFrame(time, columns=[const.TIME])

    for protocol in protocol_features:   # loop through protocols and their set of features
        PROTOCOL_DATASET_PATH = const.TS_ATTACK_DATASET_PATH if with_attacks else const.TS_BENIGN_DATASET_PATH
        protocol_data = pd.read_csv(PROTOCOL_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
        protocol_data.columns = protocol_data.columns.str.replace('_sum', '_{0}'.format(protocol))
        data = pd.concat([data, protocol_data], axis=1)
    
    data.replace(0, np.NaN, inplace=True)
    data = remove_outliers(data, upper=(not with_attacks))  # if data contains attacks dont cut upper outliers

    if with_attacks:
        if conf.remove_benign_outlier:      # removal of benign outliers from attack dataset
            data.drop(range(98,103), inplace=True)
        if conf.remove_first_attacks:
            data = data.iloc[40:]

    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    data.to_csv(DATASET_PATH.format(window_size), index=False)


def merge_features_to_attack_cat_dataset(window_size):
    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    TIME_PATH = const.TS_BENIGN_DATASET_PATH.format(window_size, list(protocol_features.keys())[0])
    time = pd.read_csv(TIME_PATH, usecols = [const.TIME], squeeze=True).apply(lambda x: x[:-2] + '00')  # delete seconds from time

    for attack_type in const.ATTACK_CATEGORIES:
        data = pd.DataFrame(time, columns=[const.TIME])
        for protocol in protocol_features: 
            benign_protocol_data = pd.read_csv(const.TS_BENIGN_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
            attack_protocol_data = pd.read_csv(const.TS_ATTACK_CATEGORY_DATASET_PATH.format(window_size, attack_type, protocol), usecols = protocol_features[protocol])
            combined_data = remove_outliers(benign_protocol_data, upper=True) + attack_protocol_data

            combined_data.columns = combined_data.columns.str.replace('_sum', '_{0}'.format(protocol))
            data = pd.concat([data, combined_data], axis=1)

        data.to_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_type), index=False)


def remove_outliers(data, upper):
    for column in data.columns:
        if column == const.TIME:
            continue
        median = data[column].median()
        if upper:
            upper_outliers = data[column] > median * 1.5
            data[column][upper_outliers] = np.nan
        lower_outliers = data[column] * 1.5 < median
        data[column][lower_outliers] = np.nan
    return interpolate_data(data)


def interpolate_data(data):
    # fill zeroes with interpolate values + rand value to not be a straight line
    for index, row in data.iterrows():
        for col in data.columns:
            if row[col] != row[col]:    # if is NaN
                mean = data[col].mean()
                data.loc[index, col] = mean + random.uniform(-mean/8, mean/8)
    return data