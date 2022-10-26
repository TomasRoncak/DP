from window_data_creation import preprocess_dataset
from feature_selection import perform_feature_selection
from time_series_generator import generate_time_series, create_extracted_dataset

window_size = 180.0

"""
clean and create time series dataset out of flow-based network capture dataset

:param include_attacks: boolean telling if dataset should contain network attacks
:param save_plots: boolean telling if plots of protocol features should be created and saved
"""

#preprocess_dataset(window_size, include_attacks=True, save_plots=True)
#preprocess_dataset(window_size, include_attacks=False, save_plots=True)

"""
perform feature selection on created time series dataset (nonunique, randomness, colinearity, adfueller, peak cutoff)

:param print_steps: boolean telling if steps should be described in more detail
"""

perform_feature_selection(window_size, print_steps=True)

"""
creates dataset suitable for training according to extracted features (on data without attacks)

:param window_size: number telling which dataset to use according to window size
"""
create_extracted_dataset(window_size)