import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config as conf
import constants as const


class TimeseriesHandler:
    def __init__(self, use_real_data, window_size, data_split, attack_cat):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stand_scaler = StandardScaler()

        self.attack_minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.attack_stand_scaler = StandardScaler()

        self.data_split = data_split

        if use_real_data:
            self.df = pd.read_csv(const.REAL_DATASET, sep='\t', usecols=const.REAL_DATASET_FEATURES)
            self.df.dropna(inplace=True)
        else:
            self.df = pd.read_csv(const.EXTRACTED_BENIGN_DATASET_PATH.format(window_size))
            self.attack_df = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_cat))

        self.features = self.df.columns.tolist()
        self.features.remove(const.TIME)    # time is not considered a feature

        self.n_features = len(self.features)    
        self.numeric_cols = self.df.columns[self.df.dtypes.apply(lambda c: np.issubdtype(c, np.number))].to_list()


    def normalize_data(self):
        self.df[self.numeric_cols] = self.minmax_scaler.fit_transform(self.df[self.numeric_cols])
        self.attack_df[self.numeric_cols] = self.attack_minmax_scaler.fit_transform(self.attack_df[self.numeric_cols])


    def scale_data(self):
        self.df[self.numeric_cols] = self.stand_scaler.fit_transform(self.df[self.numeric_cols])
        self.attack_df[self.numeric_cols] = self.attack_stand_scaler.fit_transform(self.attack_df[self.numeric_cols])


    def split_dataset(self):
        train_size = int(len(self.df) * self.data_split)
        train, test = self.df[:train_size], self.df[train_size:]
        return train, test


    def generate_time_series(self, n_input):
        self.normalize_data()
        self.scale_data()
        
        self.df = self.df.to_numpy()
        self.attack_df = self.attack_df.to_numpy()

        train, test = self.split_dataset()

        self.benign_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=1)
        self.benign_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, length=n_input, batch_size=1)
        self.attack_data_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(self.attack_df, self.attack_df, length=n_input, batch_size=1)


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
    
    if not with_attacks:
        remove_outliers(data) # removal of benign outliers from benign dataset

    if with_attacks:
        if conf.remove_benign_outlier:      # removal of benign outliers from attack dataset
            data.drop(range(98,103), inplace=True)
        if conf.remove_first_attacks:
            data = data.iloc[40:]

    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)

    data.replace(0, np.NaN, inplace=True)
    data.interpolate()  # replace zeroes with interpolate number

    data.dropna().to_csv(DATASET_PATH.format(window_size), index=False)


def merge_features_to_attack_cat_dataset(window_size):
    Path(const.EXTRACTED_DATASETS_FOLDER.format(window_size)).mkdir(parents=True, exist_ok=True)
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    TIME_PATH = const.TS_BENIGN_DATASET_PATH.format(window_size, list(protocol_features.keys())[0])
    time = pd.read_csv(TIME_PATH, usecols = [const.TIME], squeeze=True).apply(lambda x: x[:-2] + '00')  # delete seconds from time

    for attack_type in const.ATTACK_CATEGORIES:
        data = pd.DataFrame(time, columns=[const.TIME])
        for protocol in protocol_features: 
            if protocol == 'all':
                continue
            benign_protocol_data = pd.read_csv(const.TS_BENIGN_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
            attack_protocol_data = pd.read_csv(const.TS_ATTACK_CATEGORY_DATASET_PATH.format(window_size, attack_type, protocol), usecols = protocol_features[protocol])
            combined_data = benign_protocol_data + attack_protocol_data
            combined_data.columns = combined_data.columns.str.replace('_sum', '_{0}'.format(protocol))
            data = pd.concat([data, combined_data], axis=1)

        data.to_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_type), index=False)


def remove_outliers(data):
    for column in data.columns:
        if column == const.TIME:
            continue
        median = data[column].median()
        std = data[column].std()
        outliers = (data[column] - median).abs() > std
        data[column][outliers] = np.nan
        data[column].fillna(median, inplace=True)