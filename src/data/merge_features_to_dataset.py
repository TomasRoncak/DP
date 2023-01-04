import json
import sys

from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import config as conf
import constants as const

def merge_features_to_dataset(window_size):
    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    protocol_features['all'] = ['Label_sum', const.TIME]

    for attack_type in const.ATTACK_CATEGORIES:
        data = pd.DataFrame()
        for protocol in protocol_features:
            benign_protocol_data = pd.read_csv(const.TS_ATTACK_CATEGORY_DATASET_PATH.format(window_size, 'Normal', protocol), usecols = protocol_features[protocol])
            benign_protocol_data = handle_outliers(benign_protocol_data, with_attacks=False)

            if attack_type != 'Normal':
                protocol_data = pd.read_csv(const.TS_ATTACK_CATEGORY_DATASET_PATH.format(window_size, attack_type, protocol), usecols = protocol_features[protocol])
                if any(x in protocol_features[protocol] for x in ['Label_sum', const.TIME]):
                    combined_data = protocol_data   # Labels and time don't have benign data to append
                else:
                    combined_data = benign_protocol_data + protocol_data
            else:
                combined_data = benign_protocol_data
            combined_data.columns = combined_data.columns.str.replace('_sum', '_{0}'.format(protocol))
            data = pd.concat([data, combined_data], axis=1)

        if conf.remove_benign_outlier:  # Removal of benign outliers from attack dataset
            data.drop(range(98, 103), inplace=True)
        if conf.remove_first_attacks:
            data = data.iloc[40:]
        
        data[const.TIME] = data[const.TIME].apply(lambda x: x[5:16])
        data.to_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_type), index=False)
    
    plot_merged_dataset(window_size)


def handle_outliers(data, with_attacks=True):
    data.replace(0, np.NaN, inplace=True)
    data = remove_outliers(data, upper=(not with_attacks))  # If data contains attacks dont cut upper outliers
    data = interpolate_data(data)
    return data


def remove_outliers(data, upper):
    for column in data.columns:
        if column in (const.TIME, 'Label_sum'):
            continue
        median = data[column].median()
        if upper:
            upper_outliers = data[column] > median * 1.5
            data.loc[upper_outliers, [column]] = np.nan
        lower_outliers = data[column] * 1.5 < median
        data.loc[lower_outliers, [column]] = np.nan
    return data


def interpolate_data(data):
    # Fill zeroes(nan) with interpolate values + rand value to not be a straight line
    for index, row in data.iterrows():
        for column in data.columns:
            if column in (const.TIME, 'Label_sum'):
                continue
            if row[column] != row[column]:  # If is NaN
                mean = data[column].mean()
                data.loc[index, column] = mean + random.uniform(-mean/6, mean/6)
    return data


def plot_merged_dataset(window_size):
    for attack in const.ATTACK_CATEGORIES:
        path = const.EXTRACTED_DATASETS_PLOTS_FOLDER.format(window_size, attack)
        Path(path).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack))
        
        for feature in data.columns:
            if feature == const.TIME:
                continue
            pd.DataFrame(data, columns=[const.TIME, feature]).plot(x=const.TIME, y=feature, rot=90, figsize=(15, 5))
            plt.legend('', frameon=False)   # Hide legend
            plt.tight_layout()
            plt.xlabel('Čas', fontsize=15)
            plt.ylabel('Počet', fontsize=15)
            plt.title(feature, fontsize=15)
            plt.savefig(path + feature, bbox_inches='tight')
            plt.close()