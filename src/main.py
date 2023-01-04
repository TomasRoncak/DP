from data.TimeSeriesDataCreator import TimeSeriesDataCreator
from data.TimeSeriesDataHandler import TimeSeriesDataHandler

import config as conf
from data.merge_features_to_dataset import merge_features_to_dataset
from data.preprocess_data import preprocess_data_flows, preprocess_whole_data
from features.feature_selection_an import select_features_for_an
from models.AnomalyModel import AnomalyModel
from models.ClassificationModel import ClassificationModel

## Anomaly ##
anomaly_model = None
process_an_data = False
train_an = False
predict_an = True

## Category ##
category_model = None
process_cat_data = False
train_cat = False
predict_cat = True

if process_an_data:
    preprocess_whole_data()

    TimeSeriesDataCreator(window_length=conf.window_size)

    select_features_for_an(conf.window_size, print_steps=False)
    merge_features_to_dataset(conf.window_size)

if process_cat_data:
    preprocess_data_flows()

if train_an or predict_an:
    ts_handler = TimeSeriesDataHandler(
        conf.use_real_data, conf.window_size, conf.dataset_split, conf.n_steps, conf.attack_category)
    anomaly_model = AnomalyModel(ts_handler, conf.models_number,
                                 conf.an_model_name, conf.window_size, conf.patience_anomaly_limit)

    if train_an:
        if not conf.run_wandb_sweep:
            anomaly_model.train_anomaly_model(
                conf.learning_rate,
                conf.optimizer,
                conf.early_stop_patience,
                conf.an_epochs,
                conf.dropout_rate,
                conf.blocks,
                conf.an_activation,
                conf.momentum
            )
        else:
            anomaly_model.run_sweep(
                conf.an_model_name,
                conf.n_steps,
                conf.early_stop_patience,
                conf.blocks,
                conf.sweep_config_random
            )
    if predict_an:
        anomaly_model.predict_on_benign_ts()
        anomaly_model.predict_on_attack_ts()


if train_cat or predict_cat:
    category_model = ClassificationModel(
        conf.models_number, conf.cat_model_name)
    if train_cat:
        if not conf.run_wandb_sweep:
            category_model.train_categorical_model(
                conf.learning_rate,
                conf.optimizer,
                conf.early_stop_patience,
                conf.cat_epochs,
                conf.cat_batch_size,
                conf.dropout_rate,
                conf.cat_activation,
                conf.momentum
            )
        else:
            category_model.run_sweep(
                conf.cat_model_name,
                conf.early_stop_patience,
                conf.sweep_config_random
            )
    if predict_cat:
        category_model.categorize_attacks(
            on_test_set=True,
            anomaly_detection_time=None
        )
        if anomaly_model is not None:
            category_model.categorize_attacks(
                on_test_set=False,
                anomaly_detection_time=anomaly_model.anomaly_detection_time
            )
