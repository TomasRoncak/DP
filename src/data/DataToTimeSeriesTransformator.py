import copy
import csv
import datetime
from os import stat
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constants as const


class DataToTimeSeriesTransformator:
    def __init__(self, window_length):
        self.dataset = pd.read_csv(const.WHOLE_DATASET_PATH, parse_dates=[const.TIME])
        self.start_time, self.end_time = self.dataset[const.TIME].agg(['min', 'max'])[['min', 'max']]
        self.relevant_protocols = self.get_relevant_protocols(self.dataset)
        self.column_names = self.dataset.drop(columns=['service', 'attack_cat']).columns.values.tolist()
        self.column_names.append('connections')
        self.window_length = window_length
        self.include_attacks = None
        self.current_protocol = None
        self.attack_cat = None

        self.create_time_series_dataset()

    def get_relevant_protocols(self, dataset):
        relevant_protocols = ['all']
        services = dataset.service.unique()

        for protocol in services:
            if dataset['service'][dataset['service'] == protocol].count() > 1000:
                relevant_protocols.append(protocol)
        return relevant_protocols

    def create_csv(self, data):
        Path(const.PROCESSED_ANOMALY_PROTOCOL_PATH \
            .format(self.window_length, self.attack_cat, self.current_protocol)) \
            .mkdir(parents=True, exist_ok=True)
        PATH = const.TS_DATASET_BY_CATEGORY_PATH \
            .format(self.window_length, self.attack_cat, self.current_protocol)
        PROCESSED_DATA_PATH = PATH.format(self.window_length, self.current_protocol)

        with open(PROCESSED_DATA_PATH, 'a') as csv_file:                                                  
            writer = csv.writer(csv_file)
            if stat(PROCESSED_DATA_PATH).st_size == 0:  # If folder is empty, insert column names first
                writer.writerow([x + '_sum' if x != const.TIME else x for x in self.column_names])
            writer.writerow(data)

    def compute_window_statistics(self, data, curr_time):    
        # Data is content of one sliding window (x rows)
        # Calculate statistics across columns or if time, take first time (one row)
        size = data.columns.size + 1  # 1 -> connections column
        if not data.empty:
            data_row_stats = [data[column].sum() if column != const.TIME else curr_time for column in data]  
            data_row_stats.append(len(data))  # Number of connections
        else:
            data_row_stats = np.zeros(size, dtype=object)  # If no data in window found, fill with empty array
            data_row_stats[len(data_row_stats)-2] = str(curr_time)  # Time
        self.create_csv(data_row_stats)

    def perform_sliding_window(self, data):  
        window_length = datetime.timedelta(seconds=self.window_length)
        curr_time = copy.deepcopy(self.start_time)

        while curr_time < self.end_time:
            sliding_window = data[(data[const.TIME] >= curr_time) & (data[const.TIME] <= curr_time + window_length)]
            self.compute_window_statistics(sliding_window, curr_time)
            curr_time += window_length
    
    def create_time_series_dataset(self):
        print('Vytváram dáta časových radov.')
        for attack in const.ATTACK_CATEGORIES:
            self.attack_cat = attack
            data = self.dataset.copy()

            if attack == 'All':
                attack_data = data.loc[data['attack_cat'] != 'Normal']  # Get all kinds of attacks except normal
            else:
                attack_data = data.loc[data['attack_cat'] == attack]    # Get specific kind of attack

            for protocol in self.relevant_protocols:
                self.current_protocol = protocol
                data = attack_data.copy()
                if protocol != 'all':
                    data = data.loc[data['service'] == protocol]
                data.drop(columns=['service', 'attack_cat'], inplace=True)
                self.perform_sliding_window(data)
        print('Vytvorenie dokončené !')

        self.save_ts_plots()

    def save_ts_plots(self):
        print('Ukladám grafy ...')
        for attack in const.ATTACK_CATEGORIES:
            for protocol in self.relevant_protocols:
                DATASET_FILE_NAME = const.TS_DATASET_BY_CATEGORY_PATH.format(self.window_length, attack, protocol) 
                PLOTS_PATH = const.PROTOCOL_PLOTS_PATH.format(self.window_length, attack, protocol)
                self.plot(PLOTS_PATH, DATASET_FILE_NAME)
        print('Ukladanie hotové !')

    def plot(self, path, dataset_file_name):
        try:
            data = pd.read_csv(dataset_file_name, parse_dates=[const.TIME])
        except:
            return

        sns.set_style('darkgrid')
        Path(path).mkdir(parents=True, exist_ok=True)
        for feature in data.columns:
            if feature == const.TIME or (data[feature] == 0).all():   # If column contains only zeroes
                continue
            plt.rcParams['figure.figsize'] = (15, 5)
            plt.plot(data[const.TIME], data[feature])
            plt.xticks(rotation='vertical', fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel('Čas', fontsize=17, labelpad=10)
            plt.ylabel('Počet', fontsize=17, labelpad=10)
            plt.legend('', frameon=False)   # Hide legend
            plt.tight_layout()
            plt.title(feature, fontsize=20)

            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=70))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            plt.savefig(path + feature, bbox_inches='tight')
            plt.close()