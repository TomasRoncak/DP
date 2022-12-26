import json
import sys
import os

from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import config as conf
import constants as const

"""
creates dataset suitable for training according to extracted features (on data without attacks)

:param window_size: number specifying which dataset to use according to window size
"""
def merge_features_to_dataset(window_size, with_attacks):
    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    if with_attacks:
        protocol_features['all'] = ['Label_sum']

    DATASET_PATH = const.EXTRACTED_ATTACK_DATASET_PATH if with_attacks else const.EXTRACTED_BENIGN_DATASET_PATH
    TIME_PATH = const.TS_BENIGN_DATASET_PATH.format(window_size, list(protocol_features.keys())[0])

    time = pd.read_csv(TIME_PATH, usecols = [const.TIME]).squeeze().apply(lambda x: x[5:16])  # slice year and seconds from time
    data = pd.DataFrame(time, columns=[const.TIME])

    for protocol in protocol_features:  # loop through protocols and their set of features
        PROTOCOL_DATASET_PATH = const.TS_ATTACK_DATASET_PATH if with_attacks else const.TS_BENIGN_DATASET_PATH
        protocol_data = pd.read_csv(PROTOCOL_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
        protocol_data.columns = protocol_data.columns.str.replace('_sum', '_{0}'.format(protocol))
        data = pd.concat([data, protocol_data], axis=1)
    
    data = handle_outliers(data, with_attacks, remove_from_dataset=True)
    data.to_csv(DATASET_PATH.format(window_size), index=False)


def merge_features_to_dataset_by_attacks(window_size):
    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    protocol_features['all'] = ['Label_sum']

    TIME_PATH = const.TS_BENIGN_DATASET_PATH.format(window_size, list(protocol_features.keys())[0])
    time = pd.read_csv(TIME_PATH, usecols = [const.TIME]).squeeze().apply(lambda x: x[5:16])  # slice year and seconds from time

    for attack_type in const.ATTACK_CATEGORIES:
        data = pd.DataFrame(time, columns=[const.TIME])
        for protocol in protocol_features: 
            benign_protocol_data = pd.read_csv(const.TS_BENIGN_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
            attack_protocol_data = pd.read_csv(const.TS_ATTACK_CATEGORY_DATASET_PATH.format(window_size, attack_type, protocol), usecols = protocol_features[protocol])
            if protocol_features[protocol] == ['Label_sum']:
                combined_data = attack_protocol_data
            else:
                combined_data = handle_outliers(benign_protocol_data, with_attacks=True) + attack_protocol_data
            combined_data.columns = combined_data.columns.str.replace('_sum', '_{0}'.format(protocol))

            data = pd.concat([data, combined_data], axis=1)

        data.to_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_type), index=False)


def handle_outliers(data, with_attacks, remove_from_dataset=False):
    data.replace(0, np.NaN, inplace=True)
    data = remove_outliers(data, upper=(not with_attacks))  # if data contains attacks dont cut upper outliers
    data = interpolate_data(data)

    if remove_from_dataset:
        if conf.remove_benign_outlier:  # removal of benign outliers from attack dataset
            data.drop(range(98, 103), inplace=True)
        if conf.remove_first_attacks:
            data = data.iloc[40:]
    return data


def remove_outliers(data, upper):
    for column in data.columns:
        if column in(const.TIME, 'Label_all'):
            continue
        median = data[column].median()
        if upper:
            upper_outliers = data[column] > median * 1.5
            data.loc[upper_outliers, [column]] = np.nan
        lower_outliers = data[column] * 1.5 < median
        data.loc[lower_outliers, [column]] = np.nan
    return data


def interpolate_data(data):
    # fill zeroes(nan) with interpolate values + rand value to not be a straight line
    for index, row in data.iterrows():
        for column in data.columns:
            if column in(const.TIME, 'Label_all'):
                continue
            if row[column] != row[column]:  # if is NaN
                mean = data[column].mean()
                data.loc[index, column] = mean + random.uniform(-mean/2, mean/4)
    return data


def plot_merged_dataset(window_size):
    def plot(data, feature, path):
        if feature == const.TIME:
            return
        pd.DataFrame(data, columns=[const.TIME, feature]).plot(x=const.TIME, y=feature, rot=90, figsize=(15, 5))
        plt.legend('', frameon=False)   # hide legend
        plt.tight_layout()
        plt.xlabel('Čas', fontsize=15)
        plt.ylabel('Počet', fontsize=15)
        plt.title(feature, fontsize=15)
        plt.savefig(path + feature, bbox_inches='tight')
        plt.close()

    CSV_PATH = const.EXTRACTED_ATTACK_CAT_DATASET_PATH
    if os.path.exists(CSV_PATH.format(window_size, const.ATTACK_CATEGORIES[0])):
        for attack in const.ATTACK_CATEGORIES:
            path = const.EXTRACTED_DATASETS_PLOTS_FOLDER.format(window_size, attack)
            Path(path).mkdir(parents=True, exist_ok=True)
            data = pd.read_csv(CSV_PATH.format(window_size, attack))
            for feature in data.columns:
                plot(data, feature, path)

    CSV_PATH = const.EXTRACTED_ATTACK_DATASET_PATH.format(window_size, 'All_attacks')
    if os.path.exists(CSV_PATH):
        path = const.EXTRACTED_DATASETS_PLOTS_FOLDER.format(window_size, 'All_attacks')
        Path(path).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(CSV_PATH)
        for feature in data.columns:
            plot(data, feature, path)

    CSV_PATH = const.EXTRACTED_BENIGN_DATASET_PATH.format(window_size, 'Benign')
    if os.path.exists(CSV_PATH):
        path = const.EXTRACTED_DATASETS_PLOTS_FOLDER.format(window_size, 'Benign')
        Path(path).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(CSV_PATH)
        for feature in data.columns:
            plot(data, feature, path)