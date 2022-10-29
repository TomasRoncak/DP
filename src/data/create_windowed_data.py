import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import datetime
import csv
import constants as const

from os import path, makedirs, stat

def get_relevant_protocols(dataset):
    relevant_protocols = []
    services = dataset.service.unique()
    #protocols = dataset.proto.unique()

    for protocol in services:
        if dataset['service'][dataset['service'] == protocol].count() > 1000:
            relevant_protocols.append(protocol)
    return relevant_protocols


def create_csv(data, columns, window_length, include_attacks, protocol):
    CSV_PATH = const.FULL_WINDOWED_ATTACK_DATA_FILE.format(window_length, protocol) if include_attacks else const.FULL_WINDOWED_DATA_FILE.format(window_length, protocol)

    if not path.exists(const.BASE_PATH + '{0}/'.format(window_length)):   
        makedirs(const.BASE_PATH + '{0}/'.format(window_length))                          # create file <window_size>
    if not path.exists(const.BASE_PATH + '{0}/{1}'.format(window_length, protocol)):    
        makedirs(const.BASE_PATH + '{0}/{1}'.format(window_length, protocol))             # create file <protocol>
    
    with open(CSV_PATH, 'a') as csv_file:                                                 # create or append to file windowed_dataset.csv
        writer = csv.writer(csv_file)
        if stat(CSV_PATH).st_size == 0:
            writer.writerow([x + '_sum' if x != 'time' else x for x in columns])
        writer.writerow(data)


def calculate_statistics(window, window_length, include_attacks, protocol): 
    window = window.fillna(0)

    data = [] 
    for column in window:
        if column != 'time':
            data.append(window[column].astype(int).sum())
    data.append(window[column].iloc[0])
    data.append(len(window))  # number of connections

    columns = window.columns.values.tolist()
    columns.append('connections')

    create_csv(data, columns, window_length.total_seconds(), include_attacks, protocol)


def moving_window(data, window_length, include_attacks, protocol):   
    window_length = datetime.timedelta(seconds=window_length)
    
    agg = data['time'].agg(['min', 'max'])
    start_time = agg['min']
    end_time = agg['max']

    while (start_time < end_time):
        windowed_time = start_time + window_length
        window = data[(data['time'] >= start_time) & (data['time'] <= windowed_time)]

        if not window.empty:
            calculate_statistics(window, window_length, include_attacks, protocol) 
        start_time = windowed_time


def clean_data(data):
    data['time'] = pd.to_datetime(data['Stime'], unit='s')
    data['tc_flw_http_mthd'].fillna(value = data.tc_flw_http_mthd.mean(), inplace = True)
    data['is_ftp_login'].fillna(value = data.is_ftp_login.mean(), inplace = True)
    data['is_ftp_login'] = np.where(data['is_ftp_login'] > 1, 1, data['is_ftp_login'])

    data.drop(columns=const.UNUSED_FEATURES, inplace=True)


def plot_time_series(data, window_size, protocol, include_attacks): 
    PLOT_PATH = const.PLOTS_PATH.format(window_size, protocol, 'plots/') if include_attacks else const.PLOTS_PATH.format(window_size, protocol, 'plots_no_attacks/')

    if not path.exists(PLOT_PATH):
        makedirs(PLOT_PATH)

    for column in data.columns:
        if column == 'time':
            continue
        pd.DataFrame(data, columns=['time', column]).plot(x='time', y=column, rot=90, figsize=(15, 5))
        if not path.exists(PLOT_PATH + column):
            plt.tight_layout()
            plt.savefig(PLOT_PATH + column)


def save_time_series_plots(window_size, include_attacks):
    for protocol in const.PROTOCOLS:
        FILE_NAME = const.FULL_WINDOWED_ATTACK_DATA_FILE.format(window_size, protocol) if include_attacks \
               else const.FULL_WINDOWED_DATA_FILE.format(window_size, protocol)
        data = pd.read_csv(FILE_NAME)
        plot_time_series(data, window_size, protocol, include_attacks)


def preprocess_dataset(window_size, include_attacks, save_plots):
    dataset = pd.concat(map(pd.read_csv, [const.DATASET_PATH + 'UNSW-NB15_1.csv', const.DATASET_PATH + 'UNSW-NB15_2.csv', 
                                          const.DATASET_PATH + 'UNSW-NB15_3.csv', const.DATASET_PATH + 'UNSW-NB15_4.csv']), ignore_index=True)
    relevant_protocols = get_relevant_protocols(dataset)

    for protocol in relevant_protocols:
        matching_data = dataset.loc[dataset['service'] == protocol]
        if not include_attacks:
            matching_data = matching_data.loc[matching_data['Label'] == 0]
        clean_data(matching_data)
        moving_window(matching_data, window_size, include_attacks, protocol)
    
    dataset = dataset if include_attacks else dataset.loc[dataset['Label'] == 0]
    clean_data(dataset)
    moving_window(dataset, window_size, include_attacks, 'all')

    if save_plots:
        save_time_series_plots(window_size, include_attacks)