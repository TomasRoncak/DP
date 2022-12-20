import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        
        self.df['time'] = self.df['time'].str[5:16]     # slice year and seconds from time
        self.time = self.df['time']
        self.df.drop('time', axis=1, inplace=True)
        self.attack_df.drop('time', axis=1, inplace=True)

        self.df = self.df.to_numpy()
        self.attack_df = self.attack_df.to_numpy()

        train, test = self.split_dataset()

        self.benign_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=1)
        self.benign_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, length=n_input, batch_size=1)
        self.attack_data_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(self.attack_df, self.attack_df, length=n_input, batch_size=1)