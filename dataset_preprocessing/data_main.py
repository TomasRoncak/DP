from window_data_creation import process_dataset, save_time_series_plots
from feature_selection import perform_feature_selection
from time_series_generator import generate_time_series

window_size = 350.0

process_dataset(window_size, include_attacks=False)
#save_time_series_plots(window_size)
#chosen_cols = perform_feature_selection(window_size, False)

#print("------------------------------------------------------------------------------------------------------------------")
#for key in chosen_cols:
#    print(key, ' -> ', ', '.join(chosen_cols[key].values))
