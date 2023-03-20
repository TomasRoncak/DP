import json

import IPython
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import constants as const
from models.functions import bcolors


def print_test_results(to_remove, test_name):
    to_remove_len = len(to_remove)
    
    if not to_remove_len or to_remove_len > 4:
        text = 'Bolo odstránených {0} atribútov testom {1}:'
    elif to_remove_len == 1:
        text = 'Bol odstránený {0} atribút testom {1}:'
    else:
        text = 'Boli odstránené {0} atribúty testom {1}:'
    
    print(text.format(to_remove_len, test_name))
    if to_remove_len:
        print(IPython.utils.text.columnize(list(to_remove)))
    else:
        print(' '.join(to_remove))


def remove_nonunique_features(df, print_steps):
    percent = 2
    to_remove = []
    for i, v in enumerate(df.nunique()):
        p = float(v) / df.shape[0] * 100
        if p < percent:
            to_remove.append(df.columns[i])
    
    if print_steps:
        print_test_results(to_remove, test_name='unikátnosti')
    df.drop(columns=to_remove, inplace=True)


def remove_similar_features(df_attacks, df_benign, print_steps):
    max_similarity = 0.98
    to_remove = set()

    for j in range(df_attacks.shape[1]):
        column = df_attacks.iloc[:, j]
        parallel_column = df_benign[column._name]
        par_col_values = parallel_column.values
        cs_att = cosine_similarity(column.values.reshape(1, -1), par_col_values.reshape(1, -1))
        if cs_att > max_similarity or \
                par_col_values.size == len(par_col_values[par_col_values == 0]): # Column contains only zeroes:
                    to_remove.add(df_attacks.columns.values[j])
    
    if print_steps:  
        print_test_results(to_remove, test_name='kosínovej podobnosti')  
    df_attacks.drop(columns=to_remove, inplace=True)


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
        print_test_results(failed_columns, test_name='Augmented Dickey Fuller')  
    df.drop(columns=failed_columns, inplace=True)


def ljung_box_randomness_test(df, print_steps):
    to_remove = []
    for col in df.columns:
        if len(df[col]) > 1:
            res = acorr_ljungbox(df[col], lags=[3, 6, 24], return_df=True)
            if all(p > 0.05 for p in res.lb_pvalue): 
                to_remove.append(col)
    if print_steps:
        print_test_results(to_remove, test_name='Ljung-Box')  
    df.drop(columns=to_remove, inplace=True)


def remove_colinearity(df, labels, print_steps):
    if df.empty:
        return

    to_remove = set()
    feature_and_label_corr = df.corrwith(labels).sort_values().to_dict()    # Correlations between features and labels

    features_corr = df.corr().unstack().drop_duplicates()   # Correlations between features
    features_corr = features_corr[(features_corr>0.95) & (features_corr<1)].to_dict()   
    correlating_cols = list(features_corr.keys())

    for columns in correlating_cols:
        # Choose feature(column) from the two comparing columns, which has greater correlation with labels
        if any(col in to_remove for col in columns):
            continue
        if feature_and_label_corr[columns[0]] > feature_and_label_corr[columns[1]]:
            to_remove.add(columns[1])
        else:
            to_remove.add(columns[0])

    if print_steps:
        print_test_results(to_remove, test_name='kolinearity')  
    df.drop(columns=to_remove, inplace=True)


def peak_value_cutoff(df):
    percent = 0.05
    df.clip(lower=df.quantile(0), upper=df.quantile(1-percent), axis=1, inplace=True)


def select_features(protocol, window_size, print_steps):
    df_attack = pd.read_csv(const.TS_DATASET_BY_CATEGORY_PATH.format(window_size, 'All', protocol))  # Only attacks traffic
    df_benign = pd.read_csv(const.TS_DATASET_BY_CATEGORY_PATH.format(window_size, 'Normal', protocol))  # Only benign traffic
    labels = df_attack[const.LABEL_SUM].copy()

    df_benign.drop(columns=[const.TIME, const.LABEL_SUM], inplace=True)
    df_attack.drop(columns=[const.TIME, const.LABEL_SUM], inplace=True)
    df_attack = df_attack.add(df_benign, fill_value=0)  # Combine benign and attacks traffic

    remove_nonunique_features(df_attack, print_steps)
    remove_similar_features(df_attack, df_benign, print_steps)
    peak_value_cutoff(df_attack)
    adfueller_test(df_attack, print_steps)
    ljung_box_randomness_test(df_attack, print_steps)
    remove_colinearity(df_attack, labels, print_steps)

    return list(df_attack.columns)

'''
perform feature selection on created time series dataset (nonunique, colinearity, adfueller, ljung_box, peak cutoff)

:param window_size: integer specifying the length of sliding window to be used
:param print_steps: boolean specifying if selection steps should be described in more detail
'''
def select_features_for_an(window_size, print_steps):
    chosen_cols = {}

    for protocol in const.PROTOCOLS:
        if print_steps:
            print('---------------------------------------------------------')
            print(bcolors.BOLD + 'Protokol: {0} \n'.format(protocol) + bcolors.ENDC)
        
        chosen_cols[protocol] = select_features(protocol, window_size, print_steps)
        
        if print_steps:
            print(bcolors.BOLD + 'Zostávajúce atribúty:', ', '.join(chosen_cols[protocol]) + bcolors.ENDC)

    delete = [key for key in chosen_cols if chosen_cols[key] == []]
    for key in delete:
        del chosen_cols[key]

    with open(const.SELECTED_FEATURES_JSON.format(window_size), 'w') as f:
        f.write(json.dumps(chosen_cols))
            