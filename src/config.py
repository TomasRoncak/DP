## DATA CONFIG ##
window_size = 180
n_steps = 5  # Number of previous steps used to predict future value (if changed, time series data must be regenerated)
use_real_data = False
remove_first_attacks = True
remove_benign_outlier = True
attack_category = 'All'

models_number = 1

## ANOMALY MODEL CONFIG ##
an_model_name = 'cnn'
an_epochs = 75
an_activation = 'relu'
radar_plot_format = 'svg'

## CATEGORY MODEL CONFIG ##
cat_model_name = 'cnn'
cat_epochs = 40
cat_batch_size = 5000
cat_activation = 'relu'
is_cat_multiclass = False

## OTHERS ##
learning_rate = 0.001
momentum = 0
early_stop_patience = 4  # Aplikuje sa na epochy nie batche
patience_anomaly_limit = 2  # Limit pre pocet detegovanych anomalii v urcitom rozsahu tokov
dropout_rate = 0.01
blocks = 16
optimizer = 'adam'

run_wandb_sweep = True
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
            'values': [40]
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
        },
        'blocks': {
            'values': [4, 8, 12, 16, 20]
        }
    }
}