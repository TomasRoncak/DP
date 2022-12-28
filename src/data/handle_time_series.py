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
            self.df = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, 'Normal'))
            self.attack_df = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_cat))

            self.attack_labels = self.attack_df['Label_all']
            self.attack_df.drop('Label_all', axis=1, inplace=True)
            self.df.drop('Label_all', axis=1, inplace=True)

        self.features = self.df.columns.tolist()
        self.features.remove(const.TIME)    # time is not considered a feature

        self.n_features = len(self.features)    
        self.numeric_cols = self.df.columns[self.df.dtypes.apply(lambda c: np.issubdtype(c, np.number))].to_list()


    def normalize_benign_data(self, df):
        df.loc[:, self.numeric_cols] = self.minmax_scaler.fit_transform(df.loc[:, self.numeric_cols])
        return df


    def normalize_attack_data(self, attack_df):
        attack_df.loc[:, self.numeric_cols] = self.attack_minmax_scaler.fit_transform(attack_df.loc[:, self.numeric_cols])
        return attack_df


    def scale_benign_data(self, df):
        df.loc[:, self.numeric_cols] = self.stand_scaler.fit_transform(df.loc[:, self.numeric_cols])
        return df


    def scale_attack_data(self, attack_df):
        attack_df.loc[:, self.numeric_cols] = self.attack_stand_scaler.fit_transform(attack_df.loc[:, self.numeric_cols])
        return attack_df


    def split_dataset(self, df):
        train_size = int(len(df) * self.data_split)
        return df[:train_size], df[train_size:]


    def generate_time_series(self, n_input):
        self.time = self.df[const.TIME][n_input:].to_numpy()
        self.attack_time = self.attack_df[const.TIME][n_input:].to_numpy()

        self.df.drop(const.TIME, axis=1, inplace=True)
        self.attack_df.drop(const.TIME, axis=1, inplace=True)

        train, test = self.split_dataset(self.df)

        train_norm = self.normalize_benign_data(train)
        train_scaled = self.scale_benign_data(train_norm)

        test_norm = self.normalize_benign_data(test)
        test_scaled = self.scale_benign_data(test_norm)

        attack_norm = self.normalize_attack_data(self.attack_df)
        attack_scaled = self.scale_attack_data(attack_norm)

        train_scaled = train_scaled.to_numpy()
        test_scaled = test_scaled.to_numpy()
        attack_scaled = attack_scaled.to_numpy()

        self.benign_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train_scaled, train_scaled, length=n_input, batch_size=1)
        self.benign_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test_scaled, test_scaled, length=n_input, batch_size=1)
        self.attack_data_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(attack_scaled, attack_scaled, length=n_input, batch_size=1)