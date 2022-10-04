import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
import glob
import datetime
import matplotlib.dates as md
from dateutil.relativedelta import relativedelta

def moving_window(data, window_length, protocol=None):   
    variables_to_calculate_on = ['Stime', 'Ltime','sbytes', 'dbytes', 'Spkts', 'Dpkts', 'Sload', 'Dload', 'Label']
    dataset_end_time = data['Ltime'].max()           #toky ktore maju trvanie vacsie ako window_length sa nepouzivaju kvoli podmienke
    start_time = data['Stime'].min()  

    while (start_time < dataset_end_time):
        end_time = start_time + window_length
        window = data[(data['Stime'] >= start_time) & (data['Ltime'] <= end_time)]
        
        if not window[variables_to_calculate_on].empty:
            calculate_statistics(window[variables_to_calculate_on], start_time, end_time, window_length, protocol)
        start_time = end_time

def calculate_statistics(window, start_time, end_time, window_length, protocol=None): #scitat vsetky podstatne vlastnosti do jednej hodnoty pre dany window a ulozit ako riadok s casom do csv
    columns = window.columns.values.tolist()
    columns.append('time')
    columns.append('num_connections')
    data = [start_time, end_time] 
    
    for column in window:
        if column not in ['Ltime', 'Stime']:
            data.append(window[column].astype(int).sum())

    date = datetime.datetime.now() - datetime.timedelta(seconds=int(window['Stime'].mean()))
    new_date = date + relativedelta(years=38, weeks=1) - relativedelta(months=7)
    data.append(new_date)
    data.append(len(window))

    create_csv(data, columns, window_length, protocol)

def get_relevant_protocols(dataset):
    relevant_protocols = []

    services = dataset.service.unique()
    protocols = dataset.proto.unique()

    for protocol in services:
        if dataset['service'][dataset['service'] == protocol].count() > 1000:
            relevant_protocols.append(protocol)
    return relevant_protocols

def create_csv(data, columns, window_length, protocol=None):
    if protocol is not None and not os.path.exists('file/{0}/{1}'.format(window_length, protocol)):
        os.makedirs('file/{0}/{1}'.format(window_length, protocol))
    elif protocol is None and not os.path.exists('file/{0}/'.format(window_length)):
        os.makedirs('file/{0}/'.format(window_length))

    if protocol is None:
        path = 'file/{0}/general_statistics_wsize_{0}.csv'.format(window_length)
    else:
        path = 'file/{1}/{0}/statistics_wsize_{1}.csv'.format(protocol, window_length)

    exists = os.path.exists(path)
    with open(path, 'a') as statistics_csv_file:
        writer = csv.writer(statistics_csv_file)
        if not exists:
            writer.writerow([x + '_sum' if x not in ['Stime', 'Ltime', 'num_connections', 'time'] else x for x in columns])
        writer.writerow(data)

def plot_time_series(data):    
    pd.DataFrame(data, columns=['time','num_connections']).plot(x ='time', y='num_connections', rot=90, figsize=(15, 5))
    pd.DataFrame(data, columns=['time','Label_sum']).plot(x ='time', y='Label_sum', rot=90, figsize=(15, 5))
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(data, columns=['time','sbytes_sum']).plot(x ='time', y='sbytes_sum', ax=axes[0], rot=90, figsize=(15, 5))
    pd.DataFrame(data, columns=['time','dbytes_sum']).plot(x ='time', y='dbytes_sum', ax=axes[1], rot=90, figsize=(15, 5))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(data, columns=['time','Spkts_sum']).plot(x ='time', y='Spkts_sum', ax=axes[0], rot=90, figsize=(15, 5))
    pd.DataFrame(data, columns=['time','Dpkts_sum']).plot(x ='time', y='Dpkts_sum', ax=axes[1], rot=90, figsize=(15, 5))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(data, columns=['time','Sload_sum']).plot(x ='time', y='Sload_sum', ax=axes[0], rot=90, figsize=(15, 5))
    pd.DataFrame(data, columns=['time','Dload_sum']).plot(x ='time', y='Dload_sum', ax=axes[1], rot=90, figsize=(15, 5))
    

dataset = pd.concat(map(pd.read_csv, ['unsw-nb15/UNSW-NB15_1.csv', 'unsw-nb15/UNSW-NB15_2.csv', 
                                     'unsw-nb15/UNSW-NB15_3.csv', 'unsw-nb15/UNSW-NB15_4.csv']), ignore_index=True)
window_size = 500
relevant_protocols = get_relevant_protocols(dataset)
    
for protocol in relevant_protocols:
    matching_rows = dataset.loc[dataset['service'] == protocol]
    moving_window(matching_rows, window_size, protocol)

moving_window(dataset, window_size)

general_data = pd.read_csv('file/{0}/general_statistics_wsize_{0}.csv'.format(window_size))
#general_data = pd.read_csv('file/400/another.csv')

#arp_data = pd.read_csv('file/arp/statistics.csv')
#ospf_data = pd.read_csv('file/ospf/statistics.csv')
#tcp_data = pd.read_csv('file/tcp/statistics.csv')
#udp_data = pd.read_csv('file/udp/statistics.csv')

#dns_data = pd.read_csv('file/{0}/dns/statistics_wsize_{0}.csv'.format(window_size))
#ssh_data = pd.read_csv('file/{0}/ssh/statistics_wsize_{0}.csv'.format(window_size))
#ftp_data = pd.read_csv('file/{0}/ftp/statistics_wsize_{0}.csv'.format(window_size))
#ftp2_data = pd.read_csv('file/{0}/ftp-data/statistics_wsize_{0}.csv'.format(window_size))
#pop3_data = pd.read_csv('file/{0}/pop3/statistics_wsize_{0}.csv'.format(window_size))
#smtp_data = pd.read_csv('file/{0}/smtp/statistics_wsize_{0}.csv'.format(window_size))

plot_time_series(general_data)