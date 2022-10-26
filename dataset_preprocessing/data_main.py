from window_data_creation import preprocess_dataset
from feature_selection import perform_feature_selection
from time_series_generator import generate_time_series

window_size = 180.0

"""
clean and create time series dataset out of flow-based network capture dataset

:param include_attacks: boolean telling if dataset should contain network attacks
:param save_plots: boolean telling if plots of protocol features should be created and saved
"""

#preprocess_dataset(window_size, include_attacks=True, save_plots=True)

"""
perform feature selection on created time series dataset (nonunique, randomness, colinearity, adfueller, peak cutoff)

:param print_steps : boolean telling if steps should be described in more detail
"""

#perform_feature_selection(window_size, print_steps=True)

import json
import csv
import pandas as pd

features = json.load(open('dataset_preprocessing/selected_features.json'))
data = pd.DataFrame()

for feature in features:
    FILE_PATH = 'dataset_preprocessing/processed_dataset/{0}/{1}/windowed_dataset_no_attacks.csv'.format(window_size, feature)
    df = pd.read_csv(FILE_PATH, usecols = features[feature])
    df.columns = df.columns.str.replace('_sum', '_{0}'.format(feature))
    data = pd.concat([data, df], axis=1)
    data = data.dropna()

data.to_csv('dataset_preprocessing/processed_dataset/{0}/extracted_dataset.csv'.format(window_size), index=False)