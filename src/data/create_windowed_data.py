import csv
import datetime
from os import path, stat
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants as const

class TimeSeriesDataCreator:
    def __init__(self, window_length):
        self.dataset = pd.read_csv(const.WHOLE_DATASET, parse_dates=[const.TIME])
        self.relevant_protocols = self.get_relevant_protocols(self.dataset)
        self.window_length = window_length
        self.include_attacks = None
        self.current_protocol = None
        self.attack_cat = None
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


    def create_csv(self, data):
        if self.data_type == 'by_attacks':
            if self.current_protocol == '-':
                self.current_protocol = 'all'
            Path(const.PROCESSED_ATT_PROTOCOL_FOLDER.format(self.window_length, self.attack_cat, self.current_protocol)).mkdir(parents=True, exist_ok=True)
            PATH = const.TS_ATTACK_CATEGORY_DATASET_PATH.format(self.window_length, self.attack_cat, self.current_protocol)
        elif self.data_type == 'by_protocols':
            Path(const.PROCESSED_PROTOCOL_FOLDER.format(self.window_length, self.current_protocol)).mkdir(parents=True, exist_ok=True)
            PATH = const.TS_ATTACK_DATASET_PATH if self.include_attacks else const.TS_BENIGN_DATASET_PATH
        PROCESSED_DATA_PATH = PATH.format(self.window_length, self.current_protocol)

        with open(PROCESSED_DATA_PATH, 'a') as csv_file:                                                  
            writer = csv.writer(csv_file)
            if stat(PROCESSED_DATA_PATH).st_size == 0:      # if folder is empty, insert column names first
                writer.writerow([x + '_sum' if x != const.TIME else x for x in self.column_names])
            writer.writerow(data)


    def compute_window_statistics(self, data, curr_time):    
        # data is content of one sliding window (x rows)
        # calculate statistics across columns or if time, take first time (one row)
        size = data.columns.size + 1 # 1 = connections
        if not data.empty:
            data_row_stats = [data[column].sum() if column != const.TIME else curr_time for column in data]  
            data_row_stats.append(len(data))  # number of connections
        else:
            data_row_stats = np.zeros(size, dtype=object)    # if no data in window found, fill with empty array
            data_row_stats[len(data_row_stats)-2] = str(curr_time)       # time is neccesary
        self.create_csv(data_row_stats)


    def perform_sliding_window(self, data):  
        window_length = datetime.timedelta(seconds=self.window_length)
        current_time, end_time = data[const.TIME].agg(['min', 'max'])[['min', 'max']]
        
        tmp_1 = self.time_from_string('2015-01-23  01:00:00')
        tmp_2 = self.time_from_string('2015-02-18  00:00:00')

        while current_time < end_time:
            if current_time > tmp_1 and current_time < tmp_2:   # skip days where wasn't any traffic
                current_time = self.time_from_string('2015-02-18  00:25:00')
            sliding_window = data[(data[const.TIME] >= current_time) & (data[const.TIME] <= current_time + window_length)]
            self.compute_window_statistics(sliding_window, current_time)
            current_time += window_length


    """
    clean and create time series dataset out of flow-based network capture dataset

    :param window_size: integer specifying the length of sliding window to be used
    :param include_attacks: boolean specifying if dataset should contain network attacks
    :param save_plots: boolean specifying if protocol feature plots should saved
    """
    def create_ts_dataset_by_protocols(self, include_attacks):
        self.include_attacks = include_attacks
        self.data_type = 'by_protocols'
        self.column_names = self.dataset.drop(columns=['service', 'attack_cat']).columns.values.tolist()
        self.column_names.append('connections')

        for protocol in self.relevant_protocols:
            self.current_protocol = protocol
            data = self.dataset.copy()
            if protocol != 'all':
                data = data.loc[data['service'] == protocol]        # get raw data by protocol (http, ...)
            if not self.include_attacks:
                data = data.loc[data['attack_cat'] == 'Normal']     # get benign data from data filtered by protocol

            data.drop(columns=['service', 'attack_cat'], inplace=True)
            self.perform_sliding_window(data)
        
        self.save_ts_plots(self.relevant_protocols)

    
    def create_ts_dataset_by_attacks(self):
        self.data_type = 'by_attacks'
        self.column_names = self.dataset.drop(columns=['service', 'attack_cat']).columns.values.tolist()
        self.column_names.append('connections')

        for attack in const.ATTACK_CATEGORIES:
            for protocol in self.relevant_protocols:
                self.current_protocol = protocol
                self.attack_cat = attack
                data = self.dataset.copy()
                data = data.loc[(data['attack_cat'] == attack) & (data['service'] == protocol)]
                data.drop(columns=['service', 'attack_cat'], inplace=True)
                self.perform_sliding_window(data)

        self.save_ts_plots(const.ATTACK_CATEGORIES)


    def save_ts_plots(self, columns): 
        if self.data_type == 'by_protocols':
            for column in columns:
                if self.include_attacks:
                    DATASET_FILE_NAME = const.TS_ATTACK_DATASET_PATH.format(self.window_length, column) 
                    PLOTS_PATH = const.ATTACKS_PLOTS_BY_PROT_FOLDER.format(self.window_length, column)
                else:
                    DATASET_FILE_NAME = const.TS_BENIGN_DATASET_PATH.format(self.window_length, column)
                    PLOTS_PATH = const.BENIGN_PLOTS__BY_PROT_FOLDER.format(self.window_length, column)
                self.plot(PLOTS_PATH, DATASET_FILE_NAME)
        elif self.data_type == 'by_attacks':
            for attack in columns:
                for protocol in self.relevant_protocols:
                    if protocol == 'all':   #TODO
                        continue
                    if os.path.exists(const.PROCESSED_ATT_PROTOCOL_FOLDER.format(self.window_length, attack, protocol)):
                        DATASET_FILE_NAME = const.TS_ATTACK_CATEGORY_DATASET_PATH.format(self.window_length, attack, protocol) 
                        PLOTS_PATH = const.ATTACKS_PLOTS_BY_ATT_FOLDER.format(self.window_length, attack, protocol)
                        self.plot(PLOTS_PATH, DATASET_FILE_NAME)


    def plot(self, path, dataset_file_name):
        Path(path).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(dataset_file_name)

        for feature in data.columns:
            if feature == const.TIME or (data[feature] == 0).all():   # if column contains only zeroes
                continue
            if not os.path.exists(path + feature):
                pd.DataFrame(data, columns=[const.TIME, feature]).plot(x=const.TIME, y=feature, rot=90, figsize=(15, 5))
                plt.tight_layout()
                plt.savefig(path + feature)
                plt.close()