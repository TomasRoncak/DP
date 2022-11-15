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


def generate_time_series(
    window_size, 
    n_input, 
    get_train=True, 
    stl_decompose=False, 
    use_real_data=False
):

    if use_real_data:
        df = pd.read_csv(const.REAL_DATASET, sep='\t', usecols=const.REAL_DATASET_FEATURES)
        df.dropna(inplace=True)
    else:
        df = pd.read_csv(const.EXTRACTED_BENIGN_DATASET_PATH.format(window_size))

    if stl_decompose:
        df_trend = pd.DataFrame()
        for feature in df.columns:
            result = STL(df[feature], period=6, robust = True).fit()
            df_trend[feature] = result.trend.values.tolist()
        df = df_trend
        
    df = df.to_numpy()
    df = normalize_data(df)
    df = scale_data(df)
    
    train, test = split_dataset(df, 0.8)
    data = train if get_train else test
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(data, data, length=n_input, batch_size=1)

    return generator, data.shape[1], list(df.columns)


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