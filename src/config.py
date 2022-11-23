## DATA CONFIG ##
window_size = 180.0
dataset_split = 0.8
n_steps = 5
stl_decomposition = False
use_real_data = False
remove_first_attacks = True
remove_benign_outlier = True

## MODEL CONFIG ##
model_name = 'CNN'
epochs = 300
learning_rate = 0.001
patience = 10
patience_anomaly_limit = 10
dropout_rate = 0.3
blocks = 12
optimizer = 'adam'
