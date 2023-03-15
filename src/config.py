## DATA CONFIG ##
window_size = 180
dataset_split = 0.8
n_steps = 5
use_real_data = False
remove_first_attacks = True
remove_benign_outlier = True
attack_category = 'All'

models_number = 4

## ANOMALY MODEL CONFIG ##
an_model_name = 'cnn'
an_epochs = 75
an_activation = 'relu'
radar_plot_format = 'svg'

## CATEGORY MODEL CONFIG ##
cat_model_name = 'cnn'
cat_epochs = 40
cat_batch_size = 2000
cat_activation = 'relu'
is_cat_multiclass = False

## OTHERS ##
learning_rate = 0.001
momentum = 0
early_stop_patience = 10
patience_anomaly_limit = 2
dropout_rate = 0.01
blocks = 16
optimizer = 'adam'

run_wandb_sweep = False
sweep_config_random = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [32, 64, 128]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.3
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.3
        },
        'epochs': {
            'values': [100]
        },
        'optimizer': {
            'values': ['sgd', 'sgd-momentum', 'rms-prop', 'adam', 'adagrad']
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.99
        },
        'activation': {
            'values': ['relu', 'tanh', 'selu']
        }
    }
}