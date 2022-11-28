import matplotlib.pyplot as plt 
import constants as const
import pandas as pd
import datetime
import csv

from os import path, makedirs, stat

def get_relevant_protocols(dataset):
    relevant_protocols = ['all']
    services = dataset.service.unique()

    for protocol in services:
        if dataset['service'][dataset['service'] == protocol].count() > 1000:
            relevant_protocols.append(protocol)
    return relevant_protocols


def time_from_string(time_str):
    return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def create_csv(data, columns, window_length, include_attacks, protocol):
    WINDOW_FOLDER = const.PROCESSED_DATASET_PATH + '{0}/'.format(window_length)
    PROCESSED_DATA_PATH = const.TS_ATTACK_DATASET.format(window_length, protocol) if include_attacks \
        else const.TS_BENIGN_DATASET.format(window_length, protocol)

    if not path.exists(WINDOW_FOLDER):   
        makedirs(WINDOW_FOLDER)                         # create folder <window_size>
    if not path.exists(WINDOW_FOLDER + protocol):    
        makedirs(WINDOW_FOLDER + protocol)              # create folder <window_size>/<protocol>
    
    with open(PROCESSED_DATA_PATH, 'a') as csv_file:                                                  
        writer = csv.writer(csv_file)
        if stat(PROCESSED_DATA_PATH).st_size == 0:      # if folder is empty, insert column names first
            writer.writerow([x + '_sum' if x != 'time' else x for x in columns])
        writer.writerow(data)


def compute_window_statistics(data, window_length, include_attacks, protocol):    
    # data is content of one sliding window (x rows)
    # calculate statistics across columns or if time, take first time (one row)
    data_row_stats = [data[column].sum() if column != 'time' else data[column].iloc[0] for column in data]  
    data_row_stats.append(len(data))  # number of connections

    column_names = data.columns.values.tolist()
    column_names.append('connections')

    create_csv(data_row_stats, column_names, window_length.total_seconds(), include_attacks, protocol)


def perform_sliding_window(data, window_length, include_attacks, protocol):  
    window_length = datetime.timedelta(seconds=window_length)
    start_time, end_time = data['time'].agg(['min', 'max'])[['min', 'max']]
    
    tmp_1 = time_from_string('2015-01-23  01:00:00')
    tmp_2 = time_from_string('2015-02-18  00:00:00')

    while start_time < end_time:
        if start_time > tmp_1 and start_time < tmp_2:
            start_time = time_from_string('2015-02-18  00:25:00')
        sliding_window = data[(data['time'] >= start_time) & (data['time'] <= start_time + window_length)]
        if not sliding_window.empty:
            compute_window_statistics(sliding_window, window_length, include_attacks, protocol) 
        start_time += window_length


"""
clean and create time series dataset out of flow-based network capture dataset

:param window_size: integer specifying the length of sliding window to be used
:param include_attacks: boolean specifying if dataset should contain network attacks
:param save_plots: boolean specifying if protocol feature plots should saved
"""
def create_windowed_dataset(window_size, include_attacks):
    dataset = pd.read_csv(const.WHOLE_DATASET, parse_dates=['time'])
    relevant_protocols = get_relevant_protocols(dataset)

    for protocol in relevant_protocols:
        data = dataset.copy()
        if protocol != 'all':
            data = data.loc[data['service'] == protocol]        # get raw data by protocol (http, ...)
        if not include_attacks:
            data = data.loc[data['attack_cat'] == 'Normal']     # get benign data from data filtered by protocol

        data.drop(columns=const.UNUSED_FEATURES_FOR_ANOMALY, inplace=True)
        perform_sliding_window(data, window_size, include_attacks, protocol)
    
    save_ts_plots(window_size, include_attacks)


def save_ts_plots(window_size, include_attacks): 
    for protocol in const.PROTOCOLS:
        if include_attacks:
            DATASET_FILE_NAME = const.TS_ATTACK_DATASET.format(window_size, protocol) 
            PLOTS_PATH = const.ATTACKS_PLOTS_PATH.format(window_size, protocol)
        else:
            DATASET_FILE_NAME = const.TS_BENIGN_DATASET.format(window_size, protocol)
            PLOTS_PATH = const.BENIGN_PLOTS_PATH.format(window_size, protocol)

        if not path.exists(PLOTS_PATH):
            makedirs(PLOTS_PATH)

        data = pd.read_csv(DATASET_FILE_NAME)

        for feature in data.columns:
            if feature == 'time' or (data[feature] == 0).all():   # if column contains only zeroes
                continue
            if not path.exists(PLOTS_PATH + feature):
                pd.DataFrame(data, columns=['time', feature]).plot(x='time', y=feature, rot=90, figsize=(15, 5))
                plt.tight_layout()
                plt.savefig(PLOTS_PATH + feature)
                plt.close()