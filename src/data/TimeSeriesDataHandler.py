import os 
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Hide info messages in terminal

import pandas as pd
import tensorflow as tf

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import constants as const


class TimeSeriesDataHandler:
    def __init__(self, window_size, n_steps, attack_cat):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stand_scaler = StandardScaler()

        self.benign_df = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, 'Normal'), parse_dates=[const.TIME])
        self.attack_df = pd.read_csv(const.EXTRACTED_ATTACK_CAT_DATASET_PATH.format(window_size, attack_cat))

        self.time = list(self.benign_df[const.TIME])[n_steps:]

        self.benign_df.drop(const.TIME, axis=1, inplace=True)
        self.attack_df.drop(const.TIME, axis=1, inplace=True)

        self.features = self.benign_df.columns.tolist()

        self.generate_time_series(n_steps)

    def normalize_train_data(self, df):
        return self.minmax_scaler.fit_transform(df)
    
    def scale_train_data(self, df):
        return self.stand_scaler.fit_transform(df)
    
    def normalize_data(self, df):
        return self.minmax_scaler.transform(df)
    
    def scale_data(self, df):
        return self.stand_scaler.transform(df)

    def inverse_transform(self, predict):
        tmp = self.stand_scaler.inverse_transform(predict)
        return self.minmax_scaler.inverse_transform(tmp)

    def split_dataset(self, df):
        train_size = int(len(df) * 0.8)
        return df[:train_size], df[train_size:]

    def generate_time_series(self, n_input):
        train, test = self.split_dataset(self.benign_df)

        train_norm = self.normalize_train_data(train)
        train_scaled = self.scale_train_data(train_norm)

        test_norm = self.normalize_data(test)
        test_scaled = self.scale_data(test_norm)

        attack_norm = self.normalize_data(self.attack_df)
        attack_scaled = self.scale_data(attack_norm)

        self.benign_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            train_scaled, train_scaled, length=n_input, batch_size=1)
        self.benign_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            test_scaled, test_scaled, length=n_input, batch_size=1)
        self.attack_data_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            attack_scaled, attack_scaled, length=n_input, batch_size=1)