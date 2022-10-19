import matplotlib.pyplot as plt 
import pandas as pd
import datetime
import csv
import os

from dateutil.relativedelta import relativedelta


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
            writer.writerow([x + '_sum' if x not in ['Stime', 'Ltime', 'num_connections', 'time'] else x for x in columns])
        writer.writerow(data)


#scitanie vsetkych podstatnych vlastnosti do jednej hodnoty pre dany window a ulozit ich ako riadok s casom do .csv
def calculate_statistics(window, start_time, end_time, window_length, protocol=None): 
    columns = window.columns.values.tolist()
    columns.extend(['time', 'num_connections'])
    data = [start_time, end_time] 
    
    for column in window:
        if column not in ['Ltime', 'Stime']:
            data.append(window[column].astype(int).sum())

    date = datetime.datetime.now() - datetime.timedelta(seconds=int(window['Stime'].mean()))
    new_date = date + relativedelta(years=38, weeks=1) - relativedelta(months=7)

    data.append(new_date)
    data.append(len(window))

    create_csv(data, columns, window_length, protocol)


def moving_window(data, window_length, protocol=None):   
    VARIABLES_TO_CALCULATE_ON = ['Stime', 'Ltime','sbytes', 'dbytes', 'sloss', 'dloss', 'Sjit', \
                                 'Djit', 'tcprtt', 'Spkts', 'Dpkts', 'Sload', 'Dload', 'Label']
    dataset_end_time = data['Ltime'].max()  #toky trvajuce dlhsie ako window_length sa neuvazuju
    start_time = data['Stime'].min()  

    while (start_time < dataset_end_time):
        end_time = start_time + window_length
        window = data[(data['Stime'] >= start_time) & (data['Ltime'] <= end_time)]
        
        if not window[VARIABLES_TO_CALCULATE_ON].empty:
            calculate_statistics(window[VARIABLES_TO_CALCULATE_ON], start_time, end_time, window_length, protocol)
        start_time = end_time


def plot_time_series(data, window_size, protocol):    
    PATH = 'dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol)
    plot_list = [['sbytes_sum', 'dbytes_sum'], ['Spkts_sum', 'Dpkts_sum'], \
                 ['Sload_sum', 'Dload_sum'], ['Sjit_sum', 'Djit_sum'], \
                 ['sloss_sum', 'dloss_sum']]

    for x in plot_list:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        pd.DataFrame(data, columns=['time', x[0]]).plot(x ='time', y=x[0], ax=axes[0], rot=90, figsize=(15, 5))
        pd.DataFrame(data, columns=['time', x[1]]).plot(x ='time', y=x[1], ax=axes[1], rot=90, figsize=(15, 5))
        if not os.path.exists(PATH + x[0] + " & " + x[1]):
            plt.tight_layout()
            plt.savefig(PATH + x[0] + " & " + x[1])

    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(data, columns=['time','num_connections']).plot(x ='time', y='num_connections', rot=90, figsize=(15, 5))
    if not os.path.exists(PATH + 'num_connections'):
        plt.tight_layout()
        plt.savefig(PATH + 'num_connections')

    pd.DataFrame(data, columns=['time','Label_sum']).plot(x ='time', y='Label_sum', rot=90, figsize=(15, 5))
    if not os.path.exists(PATH + 'Label_sum'):
        plt.tight_layout()
        plt.savefig(PATH + 'Label_sum')

        pd.DataFrame(data, columns=['time','tcprtt_sum']).plot(x ='time', y='tcprtt_sum', rot=90, figsize=(15, 5))
    if not os.path.exists(PATH + 'tcprtt_sum'):
        plt.tight_layout()
        plt.savefig(PATH + 'tcprtt_sum')

def save_time_series_plots(window_size):
    protocols = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'pop3', 'smtp', 'ssh']
    for protocol in protocols:
        data = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/statistics_wsize_{0}.csv'.format(window_size, protocol))
        plot_time_series(data, window_size, protocol)

def process_dataset(window_size):
    dataset = pd.concat(map(pd.read_csv, ['dataset_preprocessing/dataset/UNSW-NB15_1.csv', 'dataset_preprocessing/dataset/UNSW-NB15_2.csv', 
                                          'dataset_preprocessing/dataset/UNSW-NB15_3.csv', 'dataset_preprocessing/dataset/UNSW-NB15_4.csv']), ignore_index=True)
    relevant_protocols = get_relevant_protocols(dataset)

    for protocol in relevant_protocols:
        matching_rows = dataset.loc[dataset['service'] == protocol]
        moving_window(matching_rows, window_size, protocol)
    moving_window(dataset, window_size)