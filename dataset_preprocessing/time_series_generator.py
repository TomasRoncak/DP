from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler


def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(df)


def split_dataset(df, percent):
    train_size = int(len(df) * percent)
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test


def create_time_series_data_generator(train, test):
    n_input = 3
    train_data_gen = TimeseriesGenerator(train, 
                                        train,
                                        length=n_input, 
                                        sampling_rate=1,
                                        stride=1,
                                        batch_size=1
                                        )

    test_data_gen = TimeseriesGenerator(test, 
                                        test,
                                        length=n_input, 
                                        sampling_rate=1,
                                        stride=1,
                                        batch_size=1
                                        )

    return train_data_gen, test_data_gen


def generate_time_series(df):
    normalize_data(df)
    train, test = split_dataset(df, 0.8)
    return create_time_series_data_generator(train, test)