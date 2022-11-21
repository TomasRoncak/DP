import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import datetime
import csv
import constants as const

from os import path, makedirs, stat

def get_relevant_protocols(dataset):
    relevant_protocols = ['all']
    services = dataset.service.unique()

    for protocol in services:
        if dataset['service'][dataset['service'] == protocol].count() > 1000:
            relevant_protocols.append(protocol)
    return relevant_protocols


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

    while start_time < end_time:
        sliding_window = data[(data['time'] >= start_time) & (data['time'] <= start_time + window_length)]
        if not sliding_window.empty:
            compute_window_statistics(sliding_window, window_length, include_attacks, protocol) 
        start_time += window_length


def preprocess_dataset(window_size, include_attacks, save_plots):
    dataset = pd.concat(map(pd.read_csv, [const.RAW_DATASET_PATH + 'UNSW-NB15_1.csv', 
                                          const.RAW_DATASET_PATH + 'UNSW-NB15_2.csv', 
                                          const.RAW_DATASET_PATH + 'UNSW-NB15_3.csv', 
                                          const.RAW_DATASET_PATH + 'UNSW-NB15_4.csv']), ignore_index=True)
    relevant_protocols = get_relevant_protocols(dataset)
    dataset = clean_data(dataset)

    for protocol in relevant_protocols:
        data = dataset.copy()
        if protocol != 'all':
            data = data.loc[data['service'] == protocol]
        if not include_attacks:
            data = data.loc[data['Label'] == 0]

        data.drop(columns=const.UNUSED_FEATURES, inplace=True)
        perform_sliding_window(data, window_size, include_attacks, protocol)
    
    if save_plots:
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


def clean_data(data):
    data['time'] = pd.to_datetime(data['Stime'], unit='s')
    data['tc_flw_http_mthd'].fillna(value = data.tc_flw_http_mthd.mean(), inplace = True)
    data['is_ftp_login'].fillna(value = data.is_ftp_login.mean(), inplace = True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])
    data = data.fillna(0)

    return data