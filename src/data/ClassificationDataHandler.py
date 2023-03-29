import sys
from pathlib import Path
import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')

import constants as const

from models.functions import WARNING_TEXT_RED

class ClassificationDataHandler:
    def __init__(self, model_name, is_multiclass, on_test_set, attack_categories):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.is_multiclass = is_multiclass
        self.is_model_reccurent = model_name in ['lstm', 'gru']
        self.on_test_set = on_test_set
        self.attack_categories = attack_categories
        self.train_data = pd.read_csv(const.CAT_TRAIN_VAL_DATASET)
        self.test_data = pd.read_csv(const.CAT_TEST_DATASET, parse_dates=[const.TIME], date_parser=parse_date_as_timestamp)

    def handle_train_val_data(self):
        self.train_data.drop('label' if self.is_multiclass else 'attack_cat', axis=1, inplace=True)

        trainX, trainY, self.testDf = self.train_data.iloc[:, :-1], self.train_data.iloc[:, -1], self.test_data
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, train_size=0.8, shuffle=True)
        self.trainX, self.trainY = self.scale_data(trainX, trainY, isTrain=True)
        self.valX, self.valY = self.scale_data(valX, valY)
        
        if self.is_model_reccurent:  # Reshape -> [samples, time steps, features]
           self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
           self.valX = np.reshape(self.valX, (self.valX.shape[0], 1, self.valX.shape[1]))

    def get_test_data_from_window(self, data, anomaly_detection_time):
        return data[(data[const.TIME] >= anomaly_detection_time[0]) & (data[const.TIME] <= anomaly_detection_time[1])]

    def handle_test_data(self, anomaly_detection_time=None, anomaly_count=None):
        if len(self.attack_categories) > 1 and anomaly_count is not None and self.is_multiclass:
            data = pd.read_csv(
                    const.CAT_ATTACK_CATEGORY_DATASET.format(self.attack_categories[anomaly_count-1]), 
                    parse_dates=[const.TIME], date_parser=parse_date_as_timestamp
            )
            data = self.get_test_data_from_window(data, anomaly_detection_time)
        else:
            data = self.test_data
        if not self.on_test_set:
            data = self.get_test_data_from_window(data, anomaly_detection_time)

        data = data.drop(const.TIME, axis=1)
        data.drop('label' if self.is_multiclass else 'attack_cat', axis=1, inplace=True)

        if self.is_multiclass:
            self.testX, self.testY = self.scale_data(data.iloc[:, :-1], data.iloc[:, -1])
        else:
            self.testX, _ = self.scale_data(data.iloc[:, :-1], None)
            self.testY = data.iloc[:, -1]
      
        if self.testX.size == 0:
            print(WARNING_TEXT_RED + ': Klasifikačné dáta časového okna neboli nájdené !')
            return
        if self.is_model_reccurent:  # Reshape -> [samples, time steps, features]
            self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

    def scale_data(self, dataX, dataY, isTrain=False):
        if isTrain:
            dataX = self.minmax_scaler.fit_transform(dataX)
            dataX = self.standard_scaler.fit_transform(dataX)
            dataY = self.label_encoder.fit_transform(dataY)
        else:
            dataX = self.minmax_scaler.transform(dataX)
            dataX = self.standard_scaler.transform(dataX)
            if dataY is not None:
                dataY = self.label_encoder.transform(dataY)
        return dataX, dataY
    

def preprocess_whole_data():
    Path(const.DATA_FOLDER + const.PREPROCESSED_CATEGORY_FOLDER).mkdir(exist_ok=True)
    data = pd.concat(map(pd.read_csv, [const.UNPROCESSED_PARTIAL_CSV_PATH.format(1),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(2),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(3),
                                       const.UNPROCESSED_PARTIAL_CSV_PATH.format(4)]), ignore_index=True)

    data[const.TIME] = pd.to_datetime(data['Stime'], unit='s')
    data[const.TIME] = data[const.TIME].dt.floor('Min')
    
    januar_time = list(filter(lambda x: x.month == 1, data[const.TIME]))
    februar_time = list(filter(lambda x: x.month == 2, data[const.TIME]))
    februar_time = [time.replace(month=1, day=23) + datetime.timedelta(minutes=38) for time in februar_time]
    data[const.TIME] = januar_time + februar_time

    data['tc_flw_http_mthd'].fillna(value=data.tc_flw_http_mthd.mean(), inplace=True)
    data.rename(columns={'tc_flw_http_mthd': 'ct_flw_http_mthd'}, inplace=True)

    data['is_ftp_login'].fillna(value=data.is_ftp_login.mean(), inplace=True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])

    data['attack_cat'].fillna('Normal', inplace=True)
    data['attack_cat'].replace('Backdoors', 'Backdoor', inplace=True)
    data['attack_cat'] = data['attack_cat'].str.strip()
    data = data.loc[data['attack_cat'].isin(const.ATTACK_CATEGORIES)]  # Filter out not wanted attack categories

    data.drop(columns=const.USELESS_FEATURES_FOR_PARTIAL_CSVS, inplace=True)
    data.rename(columns=lambda x: x.lower(), inplace=True) 
    data.to_csv(const.WHOLE_DATASET_PATH, index=False)


def split_whole_dataset():
    whole_df = pd.read_csv(const.WHOLE_DATASET_PATH)
    whole_df = whole_df.drop_duplicates()
    whole_df.drop('service', axis=1, inplace=True)
    time_column = whole_df.pop(const.TIME)   # Move column to first place in df
    whole_df.insert(0, const.TIME, time_column)

    normal_data = whole_df[whole_df['attack_cat'] == 'Normal'][::10]    # Reduce normal traffic by 10
    whole_df = whole_df[whole_df.attack_cat != 'Normal']
    whole_df = pd.concat([whole_df, normal_data])

    whole_train_val_data = pd.DataFrame()
    whole_test_data = pd.DataFrame()

    for cat in const.ATTACK_CATEGORIES:
        if cat == 'All':
            continue
        cat_data = whole_df[whole_df['attack_cat'] == cat]
        train_val_data = cat_data[::2]
        test_data = cat_data[1::2]

        whole_train_val_data = pd.concat([whole_train_val_data, train_val_data])
        whole_test_data = pd.concat([whole_test_data, test_data])
        if cat != 'Normal':
            pd.concat([test_data, normal_data]).to_csv(const.CAT_ATTACK_CATEGORY_DATASET.format(cat.lower()), index=False)

    whole_train_val_data.drop(const.TIME, axis=1, inplace=True)
    whole_train_val_data.to_csv(const.CAT_TRAIN_VAL_DATASET, index=False)
    whole_test_data.to_csv(const.CAT_TEST_DATASET, index=False)


def parse_date_as_timestamp(date):
    return [pd.Timestamp(one_date) for one_date in date]
