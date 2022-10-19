import matplotlib.pyplot as plt 
import pandas as pd
import datetime
import csv
import os


def get_relevant_protocols(dataset):
    relevant_protocols = []
    services = dataset.service.unique()
    #protocols = dataset.proto.unique()

    for protocol in services:
        if dataset['service'][dataset['service'] == protocol].count() > 1000:
            relevant_protocols.append(protocol)
    return relevant_protocols


def create_csv(data, columns, window_length, protocol=None):
    PATH = 'dataset_preprocessing/processed_dataset'
    if protocol is not None and not os.path.exists(PATH + '/{0}/{1}'.format(window_length, protocol)):
        os.makedirs(PATH + '/{0}/{1}'.format(window_length, protocol))
    elif protocol is None and not os.path.exists(PATH + '/{0}/'.format(window_length)):
        os.makedirs(PATH + '/{0}/'.format(window_length))

    if protocol is None:
        if not os.path.exists(PATH + '/{0}/all'.format(window_length)):
            os.makedirs(PATH + '/{0}/all'.format(window_length))
        path = PATH + '/{0}/all/statistics_wsize_{0}.csv'.format(window_length)
    else:
        path = PATH + '/{1}/{0}/statistics_wsize_{1}.csv'.format(protocol, window_length)

    exists = os.path.exists(path)
    with open(path, 'a') as statistics_csv_file:
        writer = csv.writer(statistics_csv_file)
        if not exists:
            writer.writerow([x + '_sum' for x in columns])
        writer.writerow(data)


#scitanie vsetkych podstatnych vlastnosti do jednej hodnoty pre dany window a ulozit ich ako riadok s casom do .csv
def calculate_statistics(window, window_length, protocol=None): 
    window = window.fillna(0)

    data = [] 
    for column in window:
        if column != 'time':
            data.append(window[column].astype(int).sum())
    data.append(window[column].iloc[0])
    data.append(len(window))  #num of connections

    columns = window.columns.values.tolist()
    columns.append('connections')

    create_csv(data, columns, window_length.total_seconds(), protocol)


def moving_window(data, window_length, protocol=None):   
    VARIABLES_NOT_TO_CALCULATE_ON = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', \
                                     'service', 'Stime', 'Ltime', 'attack_cat', 'dur', 'ct_ftp_cmd']
    
    data['time'] = pd.to_datetime(data['Stime'], unit='s')

    #data['time'].agg(['min', 'max'])
    start_time = data['time'].min()
    end_time = data['time'].max()
    window_length = datetime.timedelta(seconds=window_length)

    data.drop(columns=VARIABLES_NOT_TO_CALCULATE_ON, inplace=True)

    while (start_time < end_time):
        windowed_time = start_time + window_length
        window = data[(data['time'] >= start_time) & (data['time'] <= windowed_time)]

        if not window.empty:
            calculate_statistics(window, window_length, protocol) 
        start_time = windowed_time


def plot_time_series(data, window_size, protocol):    
    PATH = 'dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol)

    for i in range(0, len(data.columns), 2):
        if data.columns[i+1] == 'time_sum':
            break
        fig, axes = plt.subplots(nrows=1, ncols=2)
        pd.DataFrame(data, columns=['time_sum', data.columns[i]]).plot(x='time_sum', y=data.columns[i], ax=axes[0], rot=90, figsize=(15, 5))
        if i + 1 <= len(data.columns):
            pd.DataFrame(data, columns=['time_sum', data.columns[i+1]]).plot(x='time_sum', y=data.columns[i+1], ax=axes[1], rot=90, figsize=(15, 5))
        if not os.path.exists(PATH + data.columns[i] + " & " + data.columns[i + 1]):
            plt.tight_layout()
            plt.savefig(PATH + data.columns[i] + " & " + data.columns[i + 1])
    
    pd.DataFrame(data, columns=['time_sum', 'connections_sum']).plot(x='time_sum', y='connections_sum', rot=90, figsize=(15, 5))
    plt.tight_layout()
    plt.savefig(PATH + 'connections_sum')


def save_time_series_plots(window_size):
    protocols = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'pop3', 'smtp', 'ssh']
    for protocol in protocols:
        data = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/statistics_wsize_{0}.csv'.format(window_size, protocol))
        plot_time_series(data, window_size, protocol)


def process_dataset(window_size):
    dataset = pd.concat(map(pd.read_csv, ['dataset_preprocessing/original_dataset/UNSW-NB15_1.csv', 'dataset_preprocessing/original_dataset/UNSW-NB15_2.csv', 
                                          'dataset_preprocessing/original_dataset/UNSW-NB15_3.csv', 'dataset_preprocessing/original_dataset/UNSW-NB15_4.csv']), ignore_index=True)
    relevant_protocols = get_relevant_protocols(dataset)

    for protocol in relevant_protocols:
        matching_rows = dataset.loc[dataset['service'] == protocol]
        moving_window(matching_rows, window_size, protocol)
    moving_window(dataset, window_size)