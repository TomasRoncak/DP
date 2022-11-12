import json
import pandas as pd
import tensorflow as tf
import sys

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
import constants as const

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import STL

minmax_scaler = MinMaxScaler(feature_range=(0, 1))
stand_scaler = StandardScaler()

def normalize_data(df):
    return minmax_scaler.fit_transform(df)


def scale_data(df):
    return stand_scaler.fit_transform(df)


def split_dataset(df, percent):
    train_size = int(len(df) * percent)
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test


def ts_data_generator(data, n_input):
    return tf.keras.preprocessing.sequence.TimeseriesGenerator(data, 
                                                               data,
                                                               length=n_input, 
                                                               batch_size=1
                                                               )


def generate_time_series(window_size, n_input, get_train=True, stl_decompose=False, use_real_data=False):
    if use_real_data:
        df = pd.read_csv(const.REAL_DATASET, sep='\t')
        df = df[['conn_count_uid_in', 'conn_count_uid_out', 'dns_count_uid_out', 'http_count_uid_in', 'ssl_count_uid_in']]
        df.dropna(inplace=True)
    else:
        df = pd.read_csv(const.EXTRACTED_DATASET_PATH.format(window_size))

    features = df.columns

    if stl_decompose:
        df_trend = pd.DataFrame()
        for feature in features:
            result = STL(df[feature], period=6, robust = True).fit()
            df_trend[feature] = result.trend.values.tolist()
        df = df_trend
        
    df = df.to_numpy()
    df = normalize_data(df)
    df = scale_data(df)
    
    train, test = split_dataset(df, 0.8)
    data = train if get_train else test
    return ts_data_generator(data, n_input), data.shape[1], list(features)


def create_extracted_dataset(window_size):
    features = json.load(open(const.FULL_SELECTED_FEATURES_FILE.format(window_size)))
    data = pd.DataFrame()

    for feature in features:
        df = pd.read_csv(const.FULL_WINDOWED_DATA_FILE.format(window_size, feature), usecols = features[feature])
        df.columns = df.columns.str.replace('_sum', '_{0}'.format(feature))
        data = pd.concat([data, df], axis=1)
        data = data.dropna()

    data.to_csv(const.FULL_EXTRACTED_DATA_FILE.format(window_size), index=False)