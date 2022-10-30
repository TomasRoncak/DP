import json
import pandas as pd
import tensorflow as tf
import sys
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
import constants as const

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(df)


def scale_data(df):
    scaler = StandardScaler()
    scaler.fit_transform(df)


def split_dataset(df, percent):
    train_size = int(len(df) * percent)
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test


def create_time_series_data_generator(train, test, n_input):
    train_data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, 
                                                                        train,
                                                                        length=n_input, 
                                                                        batch_size=1
                                                                        )

    test_data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, 
                                                                        test,
                                                                        length=n_input, 
                                                                        sampling_rate=1,
                                                                        stride=1,
                                                                        batch_size=1
                                                                        )

    return train_data_gen, test_data_gen, train.shape[1]


def generate_time_series(window_size, n_input):
    df = pd.read_csv(const.EXTRACTED_DATASET_PATH.format(window_size)).to_numpy()

    normalize_data(df)
    scale_data(df)
    
    train, test = split_dataset(df, 0.8)
    return create_time_series_data_generator(train, test, n_input)


def create_extracted_dataset(window_size):
    features = json.load(open(const.FULL_SELECTED_FEATURES_FILE.format(window_size)))
    data = pd.DataFrame()

    for feature in features:
        df = pd.read_csv(const.FULL_WINDOWED_DATA_FILE.format(window_size, feature), usecols = features[feature])
        df.columns = df.columns.str.replace('_sum', '_{0}'.format(feature))
        data = pd.concat([data, df], axis=1)
        data = data.dropna()

    data.to_csv(const.FULL_EXTRACTED_DATA_FILE.format(window_size), index=False)