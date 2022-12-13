import json

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import constants as const


def remove_nonunique_columns(df, print_steps):
    percent = 2
    to_remove = []
    for i, v in enumerate(df.nunique()):
        p = float(v) / df.shape[0] * 100
        if p < percent:
            to_remove.append(df.columns[i])
    
    if print_steps:
        print('removed {0} features by Nonuniqueness test:'.format(len(to_remove)), ', '.join(to_remove), end='\n\n')
    df.drop(columns=to_remove, inplace=True)


def remove_unaffected_columns(df_attacks, df_benign, print_steps):
    max_similarity = 0.98

    sim_cols = set()
    for j in range(df_attacks.shape[1]):
        column = df_attacks.iloc[:, j]
        parallel_column = df_benign[column._name]
        cs_att = cosine_similarity(column.values.reshape(1, -1), parallel_column.values.reshape(1, -1))
        if cs_att > max_similarity or parallel_column.values.size == len(parallel_column.values[parallel_column.values == 0]): #column contains only zeroes:
            sim_cols.add(df_attacks.columns.values[j])
    
    if print_steps:    
        print('removed {0} features by Cosine similarity test:'\
            .format(len(sim_cols)), ', '.join(sim_cols), end='\n\n')
    df_attacks.drop(columns=sim_cols, inplace=True)


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
        print('removed {0} features by Augmented Dickey Fuller test:'.format(len(failed_columns)), ', '.join(failed_columns), end='\n\n')
    df.drop(columns=failed_columns, inplace=True)


def randomness_test(df, print_steps):
    to_remove = []
    for col in df.columns:
        if len(df[col]) > 1:
            res = acorr_ljungbox(df[col], lags=[3, 6, 24], return_df=True)
            if all(p > 0.05 for p in res.lb_pvalue): 
                to_remove.append(col)
    if print_steps:
        print('removed {0} features by Ljung-Box test:'.format(len(to_remove)), ', '.join(to_remove), end='\n\n')
    df.drop(columns=to_remove, inplace=True)


def remove_colinearity(df, protocol, labels, window_size, print_steps):
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
        print('removed {0} features by Collinearity test:'.format(len(to_delete)), ', '.join(to_delete), end='\n\n')
    df.drop(columns=to_delete, inplace=True)

    #if not path.exists(const.CORELLATIONS_FILE_PATH.format(window_size)):
    #    makedirs(const.CORELLATIONS_FILE_PATH.format(window_size))

    #fig, ax = plt.subplots(figsize=(8, 6))
    #svm = sns.heatmap(df.corr(), ax=ax, annot=True, fmt='.2f', cmap='YlGnBu')
    #figure = svm.get_figure()
    #figure.savefig(const.CORELLATIONS_PNG_FILE.format(window_size, protocol), dpi=400)


def peak_value_cutoff(df):
    percent = 0.05
    df.clip(lower=df.quantile(0), upper=df.quantile(1-percent), axis=1, inplace=True)


def select_features(protocol, window_size, print_steps):
    df_attack = pd.read_csv(const.TS_ATTACK_DATASET_PATH.format(window_size, protocol))
    df_benign = pd.read_csv(const.TS_BENIGN_DATASET_PATH.format(window_size, protocol))

    labels = df_benign['Label_sum'].copy()

    df_attack.drop(columns=[const.TIME, 'Label_sum'], inplace=True)
    df_benign.drop(columns=[const.TIME, 'Label_sum'], inplace=True)

    remove_nonunique_columns(df_attack, print_steps)
    remove_unaffected_columns(df_attack, df_benign, print_steps)
    peak_value_cutoff(df_attack)
    adfueller_test(df_attack, print_steps)
    randomness_test(df_attack, print_steps)
    remove_colinearity(df_attack, protocol, labels, window_size, print_steps)

    return list(df_attack.columns)

"""
perform feature selection on created time series dataset (nonunique, randomness, colinearity, adfueller, peak cutoff)

:param window_size: integer specifying the length of sliding window to be used
:param print_steps: boolean specifying if selection steps should be described in more detail
"""
def select_features_for_an(window_size, print_steps):
    chosen_cols = {}

    for protocol in const.PROTOCOLS:
        if print_steps:
            print('---------------------------------------------------------')
            print('Protocol:', protocol, '\n')
        chosen_cols[protocol] = select_features(protocol, window_size, print_steps)
        if print_steps:
            print('Remaining features:', ', '.join(chosen_cols[protocol]))

    delete = [key for key in chosen_cols if chosen_cols[key] == []]
    for key in delete:
        del chosen_cols[key]

    with open(const.SELECTED_FEATURES_JSON.format(window_size), 'w') as f:
        f.write(json.dumps(chosen_cols))
            