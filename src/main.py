from data.create_windowed_data import create_windowed_dataset
from data.handle_time_series import TimeseriesHandler, merge_features_to_dataset
from features.feature_selection_an import select_features_for_an
from models.train_model import train_anomaly
from models.predict_model import Prediction

import config as conf

process_an_data = False
process_cat_data = False

train_an = False
train_cat = False

predict_an = True
predict_cat = False

if process_an_data:
    create_windowed_dataset(conf.window_size, include_attacks=True)
    create_windowed_dataset(conf.window_size, include_attacks=False)

    select_features_for_an(conf.window_size, print_steps=False)

    merge_features_to_dataset(conf.window_size, with_attacks=True)
    merge_features_to_dataset(conf.window_size, with_attacks=False)

if process_cat_data:
    pass

ts_handler = TimeseriesHandler(conf.use_real_data, conf.window_size, conf.dataset_split)
ts_handler.generate_time_series(conf.n_steps)

if train_an:
    train_anomaly(
        ts_handler,
        conf.model_name,
        conf.n_steps, 
        conf.learning_rate, 
        conf.optimizer, 
        conf.patience, 
        conf.epochs,
        conf.dropout_rate,
        conf.blocks
    )

if train_cat:
    pass

if predict_an:
    predict = Prediction(
        ts_handler,
        conf.model_name,
        conf.patience_anomaly_limit,
        model_number=2
    )

    predict.predict_benign()
    predict.predict_attack()

if predict_cat:
    pass