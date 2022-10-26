import csv
import json
import pandas as pd
import seaborn as sns

from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.diagnostic import acorr_ljungbox


def remove_nonunique_columns(df, print_steps):
    percent = 2
    to_remove = []
    for i, v in enumerate(df.nunique()):
        p = float(v) / df.shape[0] * 100
        if p < percent:
            to_remove.append(df.columns[i])
    
    if print_steps:
        print('removed {0} features by nonuniquenes:'.format(len(to_remove)), ', '.join(to_remove), end='\n\n')
    df.drop(columns=to_remove, inplace=True)


def remove_unaffected_columns(df, df_no_attacks, print_steps):
    max_similarity = 0.98

    sim_cols = set()
    for j in range(df.shape[1]):
        column = df.iloc[:, j]
        parallel_column = df_no_attacks[column._name]
        cs_att = cosine_similarity(column.values.reshape(1, -1), parallel_column.values.reshape(1, -1))
        if cs_att > max_similarity or parallel_column.values.size == len(parallel_column.values[parallel_column.values == 0]): #column contains only zeroes:
            sim_cols.add(df.columns.values[j])
    
    if print_steps:    
        print('removed {0} features by cosine similarity (unaffected columns):'\
            .format(len(sim_cols)), ', '.join(sim_cols), end='\n\n')
    df.drop(columns=sim_cols, inplace=True)


def adfueller_test(df, print_steps):
    failed_columns = []
    for col in df.columns:
        if len(df[col]) > 1:
            dftest = adfuller(df[col], autolag='AIC')
            if (dftest[1] > 0.05):
                failed_columns.append(col)
                #print(col, dftest[1])
            #if (dftest[1] < 0.05):
            #    print(col)
    if print_steps:
        print("removed {0} features by adfueller test:".format(len(failed_columns)), ', '.join(failed_columns), end='\n\n')
    df.drop(columns=failed_columns, inplace=True)


def randomness_test(df, print_steps):
    to_remove = []
    for col in df.columns:
        if len(df[col]) > 1:
            res = acorr_ljungbox(df[col], lags=[3, 6, 24], return_df=True)
            if all(p > 0.05 for p in res.lb_pvalue): 
                to_remove.append(col)
    if print_steps:
        print("removed {0} features by randomness test:".format(len(to_remove)), ' '.join(to_remove), end='\n\n')
    df.drop(columns=to_remove, inplace=True)


def remove_colinearity(df, protocol, labels, print_steps):
    if df.empty:
        return

    corr_with_label = df.corrwith(labels).sort_values().to_dict()
    corr_matrix = df.corr()
    res = corr_matrix.unstack().drop_duplicates()
    res = res[(res>0.95) & (res<1)].to_dict()

    to_delete = set()
    correlating_cols = list(res.keys())
    for columns in correlating_cols:
        if corr_with_label[columns[0]] in to_delete or corr_with_label[columns[1]] in to_delete:
            continue
        if corr_with_label[columns[0]] > corr_with_label[columns[1]]:
            to_delete.add(columns[1])
        else:
            to_delete.add(columns[0])

    if print_steps:
        print("removed {0} features by colinearity test:".format(len(to_delete)), ' '.join(to_delete), end='\n\n')
    df.drop(columns=to_delete, inplace=True)

    fig, ax = pyplot.subplots(figsize=(8, 6))
    svm = sns.heatmap(df.corr(), ax=ax, annot=True, fmt=".2f", cmap="YlGnBu")
    figure = svm.get_figure()
    figure.savefig('dataset_preprocessing/processed_dataset/correlations/correlations_{0}.png'.format(protocol), dpi=400)


def peak_value_cutoff(df):
    percent = 0.05
    df.clip(lower=df.quantile(0), upper=df.quantile(1-percent), axis=1, inplace=True)


def select_features(protocol, window_size, print_steps):
    df = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol) + 'windowed_dataset.csv' )
    df_no_attacks = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol) + 'windowed_dataset_no_attacks.csv')

    labels = df_no_attacks['Label_sum'].copy()

    df.drop(columns=['time', 'Label_sum'], inplace=True)
    df_no_attacks.drop(columns=['time', 'Label_sum'], inplace=True)

    remove_nonunique_columns(df, print_steps)
    remove_unaffected_columns(df, df_no_attacks, print_steps)
    peak_value_cutoff(df)
    adfueller_test(df, print_steps)
    randomness_test(df, print_steps)
    remove_colinearity(df, protocol, labels, print_steps)

    return list(df.columns)


def perform_feature_selection(window_size, print_steps):
    protocols = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh']  # pop3 was removed(no attacks)
    chosen_cols = {}

    for protocol in protocols:
        if print_steps:
            print("---------------------------------------------------------")
            print("Protocol:", protocol, "\n")
        chosen_cols[protocol] = select_features(protocol, window_size, print_steps)
        if print_steps:
            print(protocol, ', '.join(chosen_cols[protocol]))

    delete = [key for key in chosen_cols if chosen_cols[key] == []]
    for key in delete:
        del chosen_cols[key]

    with open('dataset_preprocessing/selected_features.json', 'w') as f:
        f.write(json.dumps(chosen_cols))
            