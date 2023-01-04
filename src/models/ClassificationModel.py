import sys

import numpy as np
import pandas as pd

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from pathlib import Path

from keras.layers import Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from preprocess_data import format_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import constants as const
from models.functions import (get_callbacks, get_filtered_classes,
                              get_optimizer, load_best_model,
                              plot_confusion_matrix, plot_roc_auc_multiclass,
                              pretty_print_detected_attacks)


class ClassificationModel:
    def __init__(self, model_number, model_name):
        self.model = load_best_model(model_number, model_name, model_type='cat')
        self.model_number = model_number
        self.model_name = model_name

        Path(const.METRICS_CLASSIFICATION_FOLDER_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)

    def train_categorical_model(
        self,
        learning_rate,
        optimizer,
        patience,
        epochs,
        batch_size,
        dropout,
        activation,
        momentum
    ):

        df = pd.read_csv(const.CAT_TRAIN_DATASET)
        trainX, trainY = format_data(df)
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

        num_categories = df.iloc[:,-1].nunique()

        model = Sequential()
        if self.model_name == 'mlp':
            model.add(Dense(1024, activation=activation, input_dim=trainX.shape[1]))
            model.add(Dropout(dropout))
            model.add(Dense(768, activation=activation))
            model.add(Dropout(dropout))
            model.add(Dense(num_categories, activation='softmax'))

        optimizer = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)

        model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        run = wandb.init(project="dp_cat", entity="tomasroncak")

        model.fit(
                trainX,
                trainY,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(valX, valY),
                callbacks=[get_callbacks(self.model_number, self.model_name, const.CLASSIFICATION_MODEL_PATH, patience)],
                verbose=1
        )

        #model.save(const.SAVE_CAT_MODEL_PATH.format(self.model_number, model_name) + 'model.h5')
        run.finish()

    def categorize_attacks(self, on_test_set, anomaly_detection_time):
        if on_test_set:
            test_df = pd.read_csv(const.CAT_TEST_DATASET)
            x, y = format_data(test_df)
        elif not anomaly_detection_time:
            print('No window found to classify on!')
            return
        else:
            df = pd.read_csv(const.WHOLE_DATASET, parse_dates=[const.TIME])
            windowed_data = df[(df[const.TIME] >= anomaly_detection_time[0]) & (df[const.TIME] <= anomaly_detection_time[1])]
            x, y = format_data(windowed_data)
        
        prob = self.model.predict(x, verbose=0)

        self.calculate_classification_metrics(y, prob, on_test_set)
        if not on_test_set:
            pretty_print_detected_attacks(prob)

    def calculate_classification_metrics(self, y, prob, on_test_set):
        y_pred = np.argmax(prob, axis=-1)
        if (y == 0).all():
            print('Selected window contains only benign traffic !')
            return
        elif (y_pred == 0).all():
            print('Prediction contains only benign traffic !')
            return

        all_classes = get_filtered_classes()
        classes_values = np.unique(y)
        present_classes = [all_classes[x] for x in classes_values]

        METRICS_PATH = const.MODEL_CLASSIFICATION_METRICS_TEST_PATH if on_test_set else const.MODEL_CLASSIFICATION_METRICS_WINDOW_PATH
        with open(METRICS_PATH.format(self.model_number), 'w') as f:
           f.write(classification_report(y, y_pred, labels=np.unique(y_pred), target_names=present_classes))
        plot_confusion_matrix(y, y_pred, self.model_number, on_test_set, present_classes)
        plot_roc_auc_multiclass(y, prob, self.model_number, on_test_set)

    def run_sweep(
        self,
        model_name,
        early_stop_patience,
        sweep_config_random
        ):
            sweep_id = wandb.sweep(sweep_config_random, project= 'categorical_' + model_name)
            wandb.agent(
                sweep_id, 
                function=lambda: self.wandb_train(early_stop_patience), 
                count=40
            )

    def wandb_train(self,early_stop_patience):
        run = wandb.init(project='dp_categorical', entity='tomasroncak')
        self.train_categorical_model(
                    wandb.config.learning_rate,
                    wandb.config.optimizer,
                    early_stop_patience,
                    wandb.config.epochs,
                    wandb.config.batch_size,
                    wandb.config.dropout,
                    wandb.config.activation,
                    wandb.config.momentum
                    )
        run.finish()