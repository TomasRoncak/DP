## DATA CONFIG ##
window_size = 180.0
dataset_split = 0.8
n_steps = 5
use_real_data = False
remove_first_attacks = True
remove_benign_outlier = True

## AN MODEL CONFIG ##
an_model_name = 'CNN'
an_epochs = 300
an_activation = 'relu'

## CAT MODEL CONFIG ##
cat_model_name = 'MLP'
cat_epochs = 50
cat_batch_size = 2000
cat_activation = 'relu'

learning_rate = 0.001
momentum = 0
patience = 10
patience_anomaly_limit = 5
dropout_rate = 0.01
blocks = 12
optimizer = 'adam'
