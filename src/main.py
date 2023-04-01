import config as conf
from data.TimeSeriesDatasetCreator import TimeSeriesDatasetCreator
from data.DataToTimeSeriesTransformator import DataToTimeSeriesTransformator
from data.TimeSeriesDataFormatter import TimeSeriesDataFormatter
from data.ClassificationDataHandler import ClassificationDataHandler, preprocess_whole_data, split_whole_dataset
from features.feature_selection_an import select_features_for_an
from models.AnomalyModel import AnomalyModel
from models.ClassificationModel import ClassificationModel
from models.functions import create_radar_plot

anomaly_model = None
category_model = None

if conf.preprocess_data:
    preprocess_whole_data()
    split_whole_dataset()


if conf.create_time_series_data:
    DataToTimeSeriesTransformator(window_length=conf.window_size)   # Creates time series datasets


if conf.select_features:
    select_features_for_an(conf.window_size, print_steps=False)


if conf.create_final_ts_dataset:
    ts_creator = TimeSeriesDatasetCreator(conf.window_size)
    ts_creator.merge_features_to_dataset(conf.window_size)

## MODELS OBJECTS INSTANTIATION ##
if conf.train_anomaly or conf.predict_anomaly or conf.predict_anomaly_on_test or conf.create_radar_chart:
    ts_handler = TimeSeriesDataFormatter(
        conf.window_size, 
        conf.n_steps,
        conf.attack_category
    )
    anomaly_model = AnomalyModel(
        ts_handler, 
        conf.models_number,
        conf.an_model_name, 
        conf.window_size, 
        conf.patience_anomaly_limit
    )

if conf.train_category or conf.predict_category_on_test or conf.predict_anomaly:
    classification_handler = ClassificationDataHandler(
                                conf.cat_model_name, 
                                conf.is_cat_multiclass, 
                                conf.predict_category_on_test, 
                                conf.attack_categories
                            )
    classification_handler.handle_train_val_data()
    if conf.predict_category_on_test:
        classification_handler.handle_test_data()

    category_model = ClassificationModel(
        conf.models_number, 
        conf.cat_model_name,
        classification_handler,
        conf.is_cat_multiclass
    )

## ANOMALY MODEL ##
if conf.train_anomaly:
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
            conf.early_stop_patience,
            conf.sweep_config_random
        )

if conf.predict_anomaly_on_test:
    anomaly_model.predict_ts(on_test_set=True)

if conf.predict_anomaly:
    anomaly_model.predict_ts(on_test_set=False, categorize_attacks_func=category_model.categorize_attacks)
    category_model.calculate_metrics_across_windows()

if conf.create_radar_chart:
    create_radar_plot(ts_handler.features, conf.models_number, on_test_set=True, pic_format=conf.radar_plot_format)
    create_radar_plot(ts_handler.features, conf.models_number, on_test_set=False, pic_format=conf.radar_plot_format)

## CATEGORY MODEL ##
if conf.train_category:
    if not conf.run_wandb_sweep:
        category_model.train_categorical_model(
            conf.learning_rate,
            conf.optimizer,
            conf.early_stop_patience,
            conf.cat_epochs,
            conf.cat_batch_size,
            conf.dropout_rate,
            conf.cat_activation,
            conf.momentum,
            conf.blocks
        )
    else:
        category_model.run_sweep(
            conf.early_stop_patience,
            conf.sweep_config_random
        )

if conf.predict_category_on_test:
    category_model.categorize_attacks(on_test_set=conf.predict_category_on_test)