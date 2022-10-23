import matplotlib.pyplot as plt 
import pandas as pd
import datetime
import csv

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
    PATH = 'dataset_preprocessing/processed_dataset'
    FILE_NAME = 'windowed_dataset.csv' if include_attacks else 'windowed_dataset_no_attacks.csv'
    CSV_PATH = PATH + '/{0}/{1}/'.format(window_length, protocol) + FILE_NAME

    if not path.exists(PATH + '/{0}/'.format(window_length)):   
        makedirs(PATH + '/{0}/'.format(window_length))                          # create file <window_size>
    elif not path.exists(PATH + '/{0}/{1}'.format(window_length, protocol)):    
        makedirs(PATH + '/{0}/{1}'.format(window_length, protocol))             # create file <protocol>
    
    with open(CSV_PATH, 'a') as csv_file:                                       # create or append to file windowed_dataset.csv
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
    data.append(len(window))    # number of connections

    columns = window.columns.values.tolist()
    columns.append('connections')

    create_csv(data, columns, window_length.total_seconds(), include_attacks, protocol)


def moving_window(data, window_length, include_attacks, protocol):   
    VARIABLES_NOT_TO_CALCULATE_ON = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', \
                                     'service', 'Stime', 'Ltime', 'attack_cat', 'dur', 'ct_ftp_cmd']
    data['time'] = pd.to_datetime(data['Stime'], unit='s')
    data.drop(columns=VARIABLES_NOT_TO_CALCULATE_ON, inplace=True)
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


def plot_time_series(data, window_size, protocol):    
    PATH = 'dataset_preprocessing/processed_dataset/{0}/{1}/graphs/'.format(window_size, protocol)

    if not path.exists(PATH):
        makedirs(PATH)

    for i in range(0, len(data.columns), 2):
        if data.columns[i+1] == 'time_sum':
            break
        fig, axes = plt.subplots(nrows=1, ncols=2)
        pd.DataFrame(data, columns=['time_sum', data.columns[i]]).plot(x='time_sum', y=data.columns[i], ax=axes[0], rot=90, figsize=(15, 5))
        if i + 1 <= len(data.columns):
            pd.DataFrame(data, columns=['time_sum', data.columns[i+1]]).plot(x='time_sum', y=data.columns[i+1], ax=axes[1], rot=90, figsize=(15, 5))
        if not path.exists(PATH + data.columns[i] + " & " + data.columns[i + 1]):
            plt.tight_layout()
            plt.savefig(PATH + data.columns[i] + " & " + data.columns[i + 1])
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(data, columns=['time_sum', 'connections_sum']).plot(x='time_sum', y='connections_sum', ax=axes[0], rot=90, figsize=(15, 5))
    pd.DataFrame(data, columns=['time_sum', 'Label_sum']).plot(x='time_sum', y='Label_sum', ax=axes[1], rot=90, figsize=(15, 5))
    plt.tight_layout()
    if not path.exists(PATH + 'connections_sum & Label_sum'):
        plt.savefig(PATH + 'connections_sum & Label_sum')


def save_time_series_plots(window_size):
    protocols = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'pop3', 'smtp', 'ssh']
    for protocol in protocols:
        data = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/{1}_time_series.csv'.format(window_size, protocol))
        plot_time_series(data, window_size, protocol)


def process_dataset(window_size, include_attacks):
    dataset = pd.concat(map(pd.read_csv, ['dataset_preprocessing/original_dataset/UNSW-NB15_1.csv', 'dataset_preprocessing/original_dataset/UNSW-NB15_2.csv', 
                                          'dataset_preprocessing/original_dataset/UNSW-NB15_3.csv', 'dataset_preprocessing/original_dataset/UNSW-NB15_4.csv']), ignore_index=True)
    relevant_protocols = get_relevant_protocols(dataset)

    for protocol in relevant_protocols:
        matching_data = dataset.loc[dataset['service'] == protocol]
        if not include_attacks:
            matching_data = matching_data.loc[matching_data['Label'] == 0]
        moving_window(matching_data, window_size, include_attacks, protocol)
    moving_window(dataset, window_size, include_attacks, 'all')