from window_data_creation import preprocess_dataset
from feature_selection import perform_feature_selection
from time_series_generator import generate_time_series

window_size = 180.0

#preprocess_dataset(window_size, include_attacks=True, save_plots=True)
perform_feature_selection(window_size, print_steps=False, include_attacks=False)