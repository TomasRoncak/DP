import csv
import datetime
from os import path, stat
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import constants as const

class TimeSeriesDataCreator:
    def __init__(self, window_length):
        self.dataset = pd.read_csv(const.WHOLE_DATASET, parse_dates=[const.TIME])
        self.relevant_protocols = self.get_relevant_protocols(self.dataset)
        self.window_length = window_length
        self.include_attacks = None
        self.current_protocol = None
        self.data_type = None

    def get_relevant_protocols(self, dataset):
        relevant_protocols = ['all']
        services = dataset.service.unique()

        for protocol in services:
            if dataset['service'][dataset['service'] == protocol].count() > 1000:
                relevant_protocols.append(protocol)
        return relevant_protocols


    def time_from_string(self, time_str):
        return datetime.datetime.strptime(time_str, const.TIME_FORMAT)


    def create_csv(self, data, columns):
        if self.data_type == 'by_attacks':
            Path(const.PROCESSED_ATTACK_FOLDER.format(self.window_length, self.current_protocol)).mkdir(parents=True, exist_ok=True)
            PATH = const.TS_ATTACK_CATEGORY_DATASET_PATH.format(self.window_length, self.current_protocol)
        elif self.data_type == 'by_protocols':
            Path(const.PROCESSED_PROTOCOL_FOLDER.format(self.window_length, self.current_protocol)).mkdir(parents=True, exist_ok=True)
            PATH = const.TS_ATTACK_DATASET_PATH if self.include_attacks else const.TS_BENIGN_DATASET_PATH
        PROCESSED_DATA_PATH = PATH.format(self.window_length, self.current_protocol)

        with open(PROCESSED_DATA_PATH, 'a') as csv_file:                                                  
            writer = csv.writer(csv_file)
            if stat(PROCESSED_DATA_PATH).st_size == 0:      # if folder is empty, insert column names first
                writer.writerow([x + '_sum' if x != const.TIME else x for x in columns])
            writer.writerow(data)


    def compute_window_statistics(self, data):    
        # data is content of one sliding window (x rows)
        # calculate statistics across columns or if time, take first time (one row)
        data_row_stats = [data[column].sum() if column != const.TIME else data[column].iloc[0] for column in data]  
        data_row_stats.append(len(data))  # number of connections

        column_names = data.columns.values.tolist()
        column_names.append('connections')

        self.create_csv(data_row_stats, column_names)


    def perform_sliding_window(self, data):  
        window_length = datetime.timedelta(seconds=self.window_length)
        start_time, end_time = data[const.TIME].agg(['min', 'max'])[['min', 'max']]
        
        tmp_1 = self.time_from_string('2015-01-23  01:00:00')
        tmp_2 = self.time_from_string('2015-02-18  00:00:00')

        while start_time < end_time:
            if start_time > tmp_1 and start_time < tmp_2:
                start_time = self.time_from_string('2015-02-18  00:25:00')
            sliding_window = data[(data[const.TIME] >= start_time) & (data[const.TIME] <= start_time + window_length)]
            if not sliding_window.empty:
                self.compute_window_statistics(sliding_window) 
            start_time += window_length


    """
    clean and create time series dataset out of flow-based network capture dataset

    :param window_size: integer specifying the length of sliding window to be used
    :param include_attacks: boolean specifying if dataset should contain network attacks
    :param save_plots: boolean specifying if protocol feature plots should saved
    """
    def create_ts_dataset_by_protocols(self, include_attacks):
        self.include_attacks = include_attacks
        self.data_type = 'by_protocols'
        for protocol in self.relevant_protocols:
            self.current_protocol = protocol
            data = self.dataset.copy()
            if protocol != 'all':
                data = data.loc[data['service'] == protocol]        # get raw data by protocol (http, ...)
            if not self.include_attacks:
                data = data.loc[data['attack_cat'] == 'Normal']     # get benign data from data filtered by protocol

            data.drop(columns=const.UNUSED_FEATURES_FOR_ANOMALY, inplace=True)
            self.perform_sliding_window(data)
        
        self.save_ts_plots()

    
    def create_ts_dataset_by_attacks(self):
        self.data_type = 'by_attacks'
        for attack in const.ATTACK_CATEGORIES:
            self.current_protocol = attack
            data = self.dataset.copy()
            data = data.loc[data['attack_cat'] == attack]
            data.drop(columns=const.UNUSED_FEATURES_FOR_ANOMALY, inplace=True)
            self.perform_sliding_window(data)

        self.save_ts_plots()


    def save_ts_plots(self): 
        for protocol in const.PROTOCOLS:
            if self.include_attacks:
                DATASET_FILE_NAME = const.TS_ATTACK_DATASET_PATH.format(self.window_length, protocol) 
                PLOTS_PATH = const.ATTACKS_PLOTS_FOLDER.format(self.window_length, protocol)
            else:
                DATASET_FILE_NAME = const.TS_BENIGN_DATASET_PATH.format(self.window_length, protocol)
                PLOTS_PATH = const.BENIGN_PLOTS_FOLDER.format(self.window_length, protocol)

            Path(PLOTS_PATH).mkdir(parents=True, exist_ok=True)

            data = pd.read_csv(DATASET_FILE_NAME)

            for feature in data.columns:
                if feature == const.TIME or (data[feature] == 0).all():   # if column contains only zeroes
                    continue
                if not path.exists(PLOTS_PATH + feature):
                    pd.DataFrame(data, columns=[const.TIME, feature]).plot(x=const.TIME, y=feature, rot=90, figsize=(15, 5))
                    plt.tight_layout()
                    plt.savefig(PLOTS_PATH + feature)
                    plt.close()