import pandas as pd
import numpy as np

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


def remove_similar_columns(df, df_att, print_steps):
    max_similarity = 0.98

    sim_cols = set()
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        for j in range(i + 1, df.shape[1]):
            otherCol = df.iloc[:, j]
            other_col_att = df_att.iloc[:, j]
            cs = cosine_similarity(col.values.reshape(1, -1), otherCol.values.reshape(1, -1))
            cs_att = cosine_similarity(otherCol.values.reshape(1, -1), other_col_att.values.reshape(1, -1))
            if cs > max_similarity and cs_att > max_similarity:
                #print('\t', cs, df.columns[i], '\t', df.columns[j])
                sim_cols.add(df.columns.values[j])

    if print_steps:    
        print('removed {0} features by cosine similarity:'.format(len(sim_cols)), ', '.join(sim_cols), end='\n\n')
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


def remove_colinearity(df, print_steps):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_remove = [column for column in upper.columns if any(upper[column] > 0.95)]
    if print_steps:
        print("removed {0} features by colinearity:".format(len(to_remove)), ', '.join(to_remove), end='\n\n')
    df.drop(to_remove, axis=1, inplace=True)


def peak_value_cutoff(df):
    percent = 0.05
    df.clip(lower=df.quantile(0), upper=df.quantile(1-percent), axis=1, inplace=True)


def select_features(protocol, window_size, print_steps, include_attacks):
    FILE_NAME = 'windowed_dataset.csv' if include_attacks else 'windowed_dataset_no_attacks.csv'
    OTHER_FILE_NAME = 'windowed_dataset.csv' if not include_attacks else 'windowed_dataset_no_attacks.csv'
    
    df = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol) + FILE_NAME)
    df_other = pd.read_csv('dataset_preprocessing/processed_dataset/{0}/{1}/'.format(window_size, protocol) + OTHER_FILE_NAME)
    
    df.drop(columns=['time', 'Label_sum'], inplace=True)
    df_other.drop(columns=['time', 'Label_sum'], inplace=True)

    remove_similar_columns(df, df_other, print_steps)
    peak_value_cutoff(df)
    remove_nonunique_columns(df, print_steps)
    adfueller_test(df, print_steps)
    randomness_test(df, print_steps)
    #remove_colinearity(df, print_steps)

    return df.columns


def perform_feature_selection(window_size, print_steps, include_attacks):
    protocols = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh'] #pop3 was removed(no attacks)
    chosen_cols = {}

    for protocol in protocols:
        if print_steps:
            print("---------------------------------------------------------")
            print("Protocol:", protocol, "\n")
        chosen_cols[protocol] = select_features(protocol, window_size, print_steps, include_attacks)
        if print_steps:
            print(protocol, ', '.join(chosen_cols[protocol].values))
    return chosen_cols