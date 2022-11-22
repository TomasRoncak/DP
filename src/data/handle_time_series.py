import json
import pandas as pd
import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
import constants as const

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import STL

class TimeseriesHandler:
    def __init__(self, use_real_data, window_size, data_split):
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
            self.attack_df = pd.read_csv(const.EXTRACTED_ATTACK_DATASET_PATH.format(window_size))
            self.attack_df = self.attack_df.to_numpy()

        self.features = self.df.columns.tolist()
        self.df = self.df.to_numpy()
        self.n_features = len(self.features)


    def normalize_data(self):
        self.df = self.minmax_scaler.fit_transform(self.df)
        self.attack_df = self.attack_minmax_scaler.fit_transform(self.attack_df)


    def scale_data(self):
        self.df = self.stand_scaler.fit_transform(self.df)
        self.attack_df = self.attack_stand_scaler.fit_transform(self.attack_df)


    def split_dataset(self):
        train_size = int(len(self.df) * self.data_split)
        train, test = self.df[:train_size], self.df[train_size:]
        return train, test


    def generate_time_series(self, n_input, stl_decompose):
        if stl_decompose:
            df_trend = pd.DataFrame()
            for feature in self.df.columns:
                result = STL(self.df[feature], period=6, robust = True).fit()
                df_trend[feature] = result.trend.values.tolist()
            self.df = df_trend.to_numpy()
        
        self.normalize_data()
        self.scale_data()

        train, test = self.split_dataset()

        self.benign_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=1)
        self.benign_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, length=n_input, batch_size=1)
        self.attack_data_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(self.attack_df, self.attack_df, length=n_input, batch_size=1)


def create_extracted_dataset(window_size, with_attacks):
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    data = pd.DataFrame()

    for protocol in protocol_features:   # loop through protocols and their set of features
        PROTOCOL_DATASET_PATH = const.TS_ATTACK_DATASET if with_attacks else const.TS_BENIGN_DATASET
        protocol_data = pd.read_csv(PROTOCOL_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
        protocol_data.columns = protocol_data.columns.str.replace('_sum', '_{0}'.format(protocol))
        data = pd.concat([data, protocol_data], axis=1)
    
    DATASET_PATH = const.EXTRACTED_ATTACK_DATASET_PATH if with_attacks else const.EXTRACTED_BENIGN_DATASET_PATH
    if not with_attacks:
        for column in data.columns:
            median = data[column].median()
            std = data[column].std()
            outliers = (data[column] - median).abs() > std
            data[column][outliers] = np.nan
            data[column].fillna(median, inplace=True)

    data.dropna().to_csv(DATASET_PATH.format(window_size), index=False)