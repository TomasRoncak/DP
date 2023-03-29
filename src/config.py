## DATA CONFIG ##
attack_category = 'Combined'
attack_categories = ['Exploits', 'Fuzzers', 'DoS', 'Generic']  # Fill out with attakc data if dataset contains only certain types of attacks
create_final_ts_dataset = False
create_time_series_data = False
n_steps = 5  # Number of previous steps used to predict future value (if changed, time series data must be regenerated)
preprocess_data = False
remove_first_attacks = True
select_features = False
window_size = 180


## ANOMALY MODEL CONFIG ##
an_activation = 'relu'
an_epochs = 75
an_model_name = 'cnn'
create_radar_chart = False
predict_anomaly = False
predict_anomaly_on_test = False
radar_plot_format = 'svg'
train_anomaly = False


## CATEGORY MODEL CONFIG ##
cat_activation = 'relu'
cat_batch_size = 3000
cat_epochs = 50
cat_model_name = 'mlp'
is_cat_multiclass = False
predict_category_on_test = False
train_category = False


## OTHERS ##
blocks = 16
dropout_rate = 0.01
early_stop_patience = 4  # Aplikuje sa na epochy nie batche
learning_rate = 0.001
models_number = 1
momentum = 0
optimizer = 'adam'
patience_anomaly_limit = 2  # Limit pre pocet detegovanych anomalii v urcitom rozsahu tokov
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
            'values': [50]
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