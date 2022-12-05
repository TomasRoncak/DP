import config as conf
from data.create_windowed_data import TimeSeriesDataCreator
from data.handle_time_series import (TimeseriesHandler,
                                     merge_features_to_dataset, 
                                     merge_features_to_attack_cat_dataset)
from data.preprocess_data import preprocess_cat_data, preprocess_whole_data
from features.feature_selection_an import select_features_for_an
from models.predict_model import Prediction
from models.train_model import train_anomaly, train_categorical

models_number = 4
attack_category = 'All_attacks'

process_an_data = False
train_an = False
predict_an = False 

process_cat_data = False
train_cat = False
predict_cat = False


if process_an_data:
    preprocess_whole_data()

    ts_data_get = TimeSeriesDataCreator(window_length=conf.window_size)
    ts_data_get.create_ts_dataset_by_protocols(include_attacks=True)
    ts_data_get.create_ts_dataset_by_protocols(include_attacks=False)
    ts_data_get.create_ts_dataset_by_attacks()

    select_features_for_an(conf.window_size, print_steps=False)

    merge_features_to_dataset(conf.window_size, with_attacks=False)
    merge_features_to_dataset(conf.window_size, with_attacks=True)
    merge_features_to_attack_cat_dataset(conf.window_size)

if process_cat_data:
    preprocess_cat_data('train')
    preprocess_cat_data('test')

if train_an or predict_an:
    ts_handler = TimeseriesHandler(conf.use_real_data, conf.window_size, conf.dataset_split, attack_cat=attack_category)
    ts_handler.generate_time_series(conf.n_steps)

if train_an:
    train_anomaly(
        ts_handler,
        conf.an_model_name,
        conf.n_steps, 
        conf.learning_rate, 
        conf.optimizer, 
        conf.patience, 
        conf.an_epochs,
        conf.dropout_rate,
        conf.blocks,
        models_number
    )

if train_cat:
    train_categorical(
        conf.cat_model_name,
        conf.learning_rate,
        conf.optimizer,
        conf.patience,
        conf.cat_epochs,
        conf.cat_batch_size,
        conf.dropout_rate,
        models_number
    )

if predict_an or predict_cat:
    predict = Prediction(
        ts_handler,
        conf.an_model_name,
        conf.cat_model_name,
        conf.patience_anomaly_limit,
        model_number=models_number
    )

    if predict_an:
        predict.predict_benign()
        predict.predict_attack()

    if predict_cat:
        predict.categorize_attack()