import os
import sys

import numpy as np

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from datetime import datetime as dt
from pathlib import Path

import absl.logging
from keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from keras.models import Sequential
from sklearn.metrics import classification_report

import constants as const
from models.functions import (get_attack_classes, get_callbacks, get_optimizer,
                              load_best_model, plot_confusion_matrix,
                              plot_roc_auc, pretty_print_detected_attacks)

absl.logging.set_verbosity(absl.logging.ERROR) # ignore warning ('Found untraced functions such as ...')

os.environ['WANDB_SILENT'] = 'true'

class ClassificationModel:
    def __init__(self, model_number, model_name, classification_handler, is_cat_multiclass):
        self.model_number = model_number
        self.model_name = model_name
        self.data_handler = classification_handler
        self.is_multiclass = is_cat_multiclass
        self.model_path = const.WHOLE_CLASSIFICATION_MULTICLASS_MODEL_PATH.format(model_number) \
                          if self.is_multiclass else const.WHOLE_CLASSIFICATION_BINARY_MODEL_PATH.format(model_number)
        self.all_windows_y = []
        self.all_windows_prob = []

    def train_categorical_model(
        self,
        learning_rate,
        optimizer,
        patience,
        epochs,
        batch_size,
        dropout,
        activation,
        momentum,
        blocks
    ):
        num_categories = len(np.unique(self.trainY))

        if num_categories > 2:  
            loss = SparseCategoricalCrossentropy()
            last_activation = 'softmax'
        else:  # Unikatne hodnoty su 1 a 0 -> binarna klasifikacia
            num_categories = 1
            loss = BinaryCrossentropy()
            last_activation = 'sigmoid'

        model = Sequential()
        if self.model_name == 'mlp':
            model.add(Dense(16, activation=activation, input_dim=self.trainX.shape[1]))
            model.add(Dropout(dropout))
            model.add(Dense(8, activation=activation))
            model.add(Dropout(dropout))
        elif self.model_name == 'cnn':
            model.add(Conv1D(filters=16, padding='same', kernel_size=2, activation=activation, input_shape=(self.trainX.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=8, padding='same', kernel_size=2, activation=activation))
            model.add(Dropout(dropout)),
            model.add(Flatten())
        elif self.model_name == 'lstm':
            model.add(LSTM(blocks, input_dim=self.trainX.shape[2]))
        elif self.model_name == 'gru':
            model.add(GRU(blocks, input_dim=self.trainX.shape[2]))
            model.add(Dropout(dropout))
        elif self.model_name == 'cnn_lstm':
            model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation=activation, input_shape=(self.trainX.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(blocks))
            model.add(Dropout(dropout))
        # elif self.model_name == 'rf':
        #     model = RandomForestClassifier(n_estimators = 200, n_jobs=-1, random_state=0, bootstrap=True)
        #     model.fit(self.trainX, self.trainY)
        #     save_rf_model(model, self.is_multiclass, self.model_number, self.model_name)
        #     return
        else:
            raise Exception('Nepodporovaný typ modelu !')
        
        model.add(Dense(num_categories, activation=last_activation))

        optimizer = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        run = wandb.init(
            project=('multiclass' if self.is_multiclass else 'binary') + '_classification',
            group=self.model_name,
            job_type='eval',
            entity='tomasroncak'
        )

        start = dt.now()
        model.fit(
                self.trainX,
                self.trainY,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.valX, self.valY),
                callbacks=[get_callbacks(
                            self.model_number,
                            self.model_name,
                            patience,
                            self.is_multiclass
                           )],
                verbose=1
        )
        print('Tréning modelu {0} prebiehal {1} sekúnd.'.format(self.model_name, (dt.now() - start).seconds))

        #model.save(const.save_model[self.is_multiclass].format(self.model_number, self.model_name) + 'model.h5')
        run.finish()

    def categorize_attacks(self, anomaly_detection_time=None, on_test_set=False, anomaly_count=None):
        self.model = load_best_model(self.model_number, self.model_name, model_type='cat', is_cat_multiclass=self.is_multiclass)

        if anomaly_detection_time:
            self.data_handler.handle_test_data(anomaly_detection_time, anomaly_count)
        x, y = self.data_handler.testX, self.data_handler.testY
        
        prob = self.model.predict(x, verbose=0)

        self.all_windows_y.extend(y)
        self.all_windows_prob.extend(prob.tolist())

        self.calculate_metrics(y, prob, on_test_set, anomaly_count)
        if not on_test_set:
            pretty_print_detected_attacks(prob, self.is_multiclass)

    def calculate_metrics_across_windows(self):
        self.calculate_metrics(self.all_windows_y, np.array(self.all_windows_prob), False, 'all')

    def calculate_metrics(self, y, prob, on_test_set, file_postfix):
        Path(const.metrics.path[on_test_set] \
            .format(self.model_path, self.model_name, file_postfix)) \
            .mkdir(parents=True, exist_ok=True)

        if self.is_multiclass:
            # Pre multiclass pouzivame Softmax aktivacnu funkciu, ktora vrati pravdepodobnost pre kazdu triedu
            # argmax funkcia vrati index s najvyssou hodnotou (pravdepodobnostou)
            if self.model_name == 'rf':
                y_pred = prob
            else:
                y_pred = np.argmax(prob, axis=-1)  
            all_classes = get_attack_classes()
            present_classes = [all_classes[x] for x in np.unique(y)]
        else:
            # Pre binary pouzivame Sigmoid aktivacnu funkciu, ktora vrati pravdepodobnost v intervale <0,1> preto staci len zaokruhlenie
            y_pred = np.round(prob, 0)
            present_classes = ['Benígne'] if len(np.unique(y)) == 1 and (y == 0).all() else ['Benígne', 'Malígne']

        with open(const.metrics.report[on_test_set].format(self.model_path, self.model_name, file_postfix), 'w') as f:
           f.write(classification_report(y, y_pred, labels=np.unique(y), target_names=present_classes, zero_division=0))
        
        if len(present_classes) > 1:
            plot_confusion_matrix(y, y_pred, self.model_number, self.is_multiclass,
                const.metrics.conf_m[on_test_set].format(self.model_path, self.model_name, file_postfix))
            plot_roc_auc(y, prob, self.model_number, self.data_handler.trainY, self.model_name,
                const.metrics.roc_auc[on_test_set].format(self.model_path, self.model_name, file_postfix))

    def run_sweep(self, early_stop_patience, sweep_config_random):
        project_name = 'multiclass' if self.is_multiclass else 'binary' + '_categorical'
        sweep_id = wandb.sweep(
            sweep_config_random, 
            project=project_name + '_sweep'
        )
        wandb.agent(
            sweep_id, 
            function=lambda: self.wandb_train(early_stop_patience, project_name), 
            count=20
        )

    def wandb_train(self, early_stop_patience, project_name):
        run = wandb.init(project=project_name, group=self.model_name, entity='tomasroncak')
        self.train_categorical_model(
                    wandb.config.learning_rate,
                    wandb.config.optimizer,
                    early_stop_patience,
                    wandb.config.epochs,
                    wandb.config.batch_size,
                    wandb.config.dropout,
                    wandb.config.activation,
                    wandb.config.momentum,
                    wandb.config.blocks
        )
        run.finish()