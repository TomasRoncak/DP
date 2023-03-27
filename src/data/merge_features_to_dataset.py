import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import config as conf
import constants as const


def merge_features_to_dataset(window_size):
    print('Vytváram finálny dataset podľa vybraných atribútov.')
    try:
        protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
        Path(const.EXTRACTED_DATASETS_PATH.format(window_size)).mkdir(parents=True, exist_ok=True)
    except:
        print('Súbor (.json) s vybranými atribútmi nebol nájdený !')
        return
    
    time = pd.read_csv(const.TS_DATASET_BY_CATEGORY_PATH.format(window_size, 'Normal', 'all'), usecols = [const.TIME])
    for attack_type in const.ATTACK_CATEGORIES:
        data = pd.DataFrame(time)
        for protocol in protocol_features:
            path = const.TS_DATASET_BY_CATEGORY_PATH.format(window_size, attack_type, protocol)
            benign_protocol_data = pd.read_csv(
                                        const.TS_DATASET_BY_CATEGORY_PATH.format(window_size, 'Normal', protocol), 
                                        usecols = protocol_features[protocol]
                                   )
            combined_data = handle_outliers(benign_protocol_data)   # At this point it contains only benign traffic

            if os.path.exists(path) and attack_type != 'Normal':  
                malign_protocol_data = pd.read_csv(path, usecols = protocol_features[protocol])
                combined_data += malign_protocol_data   # Here is the data combined with malign traffic
            
            combined_data.columns = combined_data.columns.str.replace('_sum', '_{0}'.format(protocol))
            data = pd.concat([data, combined_data], axis=1)

        if conf.remove_first_attacks:
            data = data.iloc[40:]
        
        data.to_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_type), index=False)
    
    print('Vytvorenie dokončené !')
    plot_merged_dataset(window_size)


def handle_outliers(data):
    data.replace(0, np.NaN, inplace=True)
    data = remove_outliers(data, upper=True)  # If data contains attacks dont cut upper outliers
    data = interpolate_data(data)
    return data


def remove_outliers(data, upper):
    for column in data.columns:
        if column == const.TIME:
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
            if column == const.TIME:
                continue
            if row[column] != row[column]:  # If is NaN
                mean = data[column].mean()
                data.loc[index, column] = mean + random.uniform(-mean/6, mean/6)
    return data


def plot_merged_dataset(window_size):
    print('Ukladám grafy ...')
    for attack in const.ATTACK_CATEGORIES:
        path = const.EXTRACTED_DATASETS_PLOTS_PATH.format(window_size, attack)
        Path(path).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack))
        
        for feature in data.columns:
            if feature == const.TIME:
                continue
            pd.DataFrame(data, columns=[const.TIME, feature]) \
                .plot(x=const.TIME, y=feature, rot=90, figsize=(15, 5))
            plt.legend('', frameon=False)   # Hide legend
            plt.tight_layout()
            plt.xlabel('Čas', fontsize=15)
            plt.ylabel('Počet', fontsize=15)
            plt.title(feature, fontsize=15)
            plt.savefig(path + feature, bbox_inches='tight')
            plt.close()
    print('Ukladanie hotové !')