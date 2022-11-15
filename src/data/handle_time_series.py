import json
import pandas as pd
import tensorflow as tf
import sys

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
import constants as const

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import STL

class TimeseriesHandler:
    def __init__(self, use_real_data, window_size):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stand_scaler = StandardScaler()

        if use_real_data:
            self.df = pd.read_csv(const.REAL_DATASET, sep='\t', usecols=const.REAL_DATASET_FEATURES)
            self.df.dropna(inplace=True)
        else:
            self.df = pd.read_csv(const.EXTRACTED_BENIGN_DATASET_PATH.format(window_size))
        self.features = self.df.columns.tolist()
        self.n_features = len(self.features)
        self.df = self.df.to_numpy()


    def normalize_data(self):
        self.df = self.minmax_scaler.fit_transform(self.df)


    def scale_data(self):
        self.df = self.stand_scaler.fit_transform(self.df)


    def split_dataset(self, percent):
        train_size = int(len(self.df) * percent)
        train, test = self.df[0:train_size], self.df[train_size:len(self.df)]
        return train, test


    def generate_time_series(self, n_input, stl_decompose, percent_split=0.8):
        if stl_decompose:
            df_trend = pd.DataFrame()
            for feature in self.df.columns:
                result = STL(self.df[feature], period=6, robust = True).fit()
                df_trend[feature] = result.trend.values.tolist()
            self.df = df_trend.to_numpy()
        
        self.normalize_data()
        self.scale_data()
        train, test = self.split_dataset(percent_split)

        self.train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=1)
        self.test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, length=n_input, batch_size=1)


def create_extracted_dataset(window_size, with_attacks):
    protocol_features = json.load(open(const.SELECTED_FEATURES_JSON.format(window_size)))
    data = pd.DataFrame()

    for protocol in protocol_features:   # loop through protocols and their set of features
        PROTOCOL_DATASET_PATH = const.TS_ATTACK_DATASET if with_attacks else const.TS_BENIGN_DATASET
        protocol_data = pd.read_csv(PROTOCOL_DATASET_PATH.format(window_size, protocol), usecols = protocol_features[protocol])
        protocol_data.columns = protocol_data.columns.str.replace('_sum', '_{0}'.format(protocol))
        data = pd.concat([data, protocol_data], axis=1)
    
    DATASET_PATH = const.EXTRACTED_ATTACK_DATASET_PATH if with_attacks else const.EXTRACTED_BENIGN_DATASET_PATH
    data.dropna().to_csv(DATASET_PATH.format(window_size), index=False)