import os
import sys

import numpy as np
import pandas as pd

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import constants as const
from models.functions import (WARNING_TEXT, get_callbacks,
                              get_filtered_classes, get_optimizer,
                              load_best_model, parse_date_as_timestamp,
                              plot_confusion_matrix, plot_roc_auc,
                              pretty_print_detected_attacks,
                              reduce_normal_traffic, save_rf_model)

absl.logging.set_verbosity(absl.logging.ERROR) # ignore warning ('Found untraced functions such as ...')

os.environ['WANDB_SILENT'] = 'true'

class ClassificationModel:
    def __init__(self, model_number, model_name, is_cat_multiclass):
        self.model_number = model_number
        self.model_name = model_name
        self.is_cat_multiclass = is_cat_multiclass
        self.is_model_reccurent = self.model_name in ['lstm', 'gru']
        self.model_path = const.WHOLE_CLASSIFICATION_MULTICLASS_MODEL_PATH.format(model_number) \
                          if self.is_cat_multiclass else const.WHOLE_CLASSIFICATION_BINARY_MODEL_PATH.format(model_number)
        self.handle_data()

    def handle_data(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        to_delete = 'label' if self.is_cat_multiclass else 'attack_cat'
        features = [const.TIME, 'label', 'attack_cat']

        trainX, trainY = reduce_normal_traffic(pd.read_csv(const.CAT_TRAIN_VAL_DATASET), to_delete)
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, train_size=0.8, shuffle=True)
        self.trainX, self.trainY = self.scale_data(trainX, trainY, isTrain=True)
        self.valX, self.valY = self.scale_data(valX, valY)
        
        self.testDf = pd.read_csv(const.CAT_TEST_DATASET, parse_dates=[const.TIME], date_parser=parse_date_as_timestamp)
        self.testDf.drop(to_delete, axis=1, inplace=True)

        selected = [x for x in list(self.testDf.columns) if (x not in features)]
        self.testDf[selected], _ = self.scale_data(self.testDf[selected], None)

        if self.is_model_reccurent:  # Reshape -> [samples, time steps, features]
            self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
            self.valX = np.reshape(self.valX, (self.valX.shape[0], 1, self.valX.shape[1]))

    def scale_data(self, dataX, dataY, isTrain=False):
        if isTrain:
            dataX = self.minmax_scaler.fit_transform(dataX)
            dataX = self.standard_scaler.fit_transform(dataX)
            dataY = self.label_encoder.fit_transform(dataY)
        else:
            dataX = self.minmax_scaler.transform(dataX)
            dataX = self.standard_scaler.transform(dataX)
            if dataY is not None:
                dataY = self.label_encoder.transform(dataY)
        return dataX, dataY

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
        elif self.model_name == 'rf':
            model = RandomForestClassifier(n_estimators = 200, n_jobs=-1, random_state=0, bootstrap=True)
            model.fit(self.trainX, self.trainY)
            save_rf_model(model, self.is_cat_multiclass, self.model_number, self.model_name)
            return
        else:
            raise Exception('Nepodporovaný typ modelu !')
        
        model.add(Dense(num_categories, activation=last_activation))

        optimizer = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        run = wandb.init(
            project=('multiclass' if self.is_cat_multiclass else 'binary') + '_classification',
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
                            self.is_cat_multiclass
                           )],
                verbose=1
        )
        print('Tréning modelu {0} prebiehal {1} sekúnd.'.format(self.model_name, (dt.now() - start).seconds))

        #model.save(const.save_model[self.is_cat_multiclass].format(self.model_number, self.model_name) + 'model.h5')
        run.finish()

    def categorize_attacks(self, an_detect_time=None, on_test_set=False, anomaly_count=None):
        self.model = load_best_model(self.model_number, self.model_name, model_type='cat', is_cat_multiclass=self.is_cat_multiclass)

        if on_test_set:
            x, y = self.testX, self.testY
        elif an_detect_time:
            window_data = self.testDf[(self.testDf[const.TIME] >= an_detect_time[0]) & (self.testDf[const.TIME] <= an_detect_time[1])]
            data = window_data.drop(const.TIME, axis=1)
            x, y = data.iloc[:, :-1], data.iloc[:, -1]
            if x.empty:
                print(WARNING_TEXT + ': Klasifikačné dáta časového okna {0} - {1} neboli nájdené !'.format(an_detect_time[0], an_detect_time[1]))
                return
            if self.is_model_reccurent:  # Reshape -> [samples, time steps, features]
                x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        
        if self.model_name == 'rf':
            prob = self.model.predict(x)
        else:
            prob = self.model.predict(x, verbose=0)

        self.calculate_classification_metrics(y, prob, on_test_set, anomaly_count)
        if not on_test_set:
            pretty_print_detected_attacks(prob, self.is_cat_multiclass)

    def calculate_classification_metrics(self, y, prob, on_test_set, anomaly_count):
        Path(const.metrics.path[on_test_set] \
            .format(self.model_path, self.model_name, anomaly_count)) \
            .mkdir(parents=True, exist_ok=True)

        if self.is_cat_multiclass:  # Multiclass classification
            # Pre multiclass pouzivame Softmax aktivacnu funkciu, ktora vrati pravdepodobnost pre kazdu triedu
            # argmax funkcia vrati index s najvyssou hodnotou (pravdepodobnostou)
            if self.model_name == 'rf':
                y_pred = prob
            else:
                y_pred = np.argmax(prob, axis=-1)  
            all_classes = get_filtered_classes()
            present_classes = [all_classes[x] for x in np.unique(y)]
        else:
            # Pre binary pouzivame Sigmoid aktivacnu funkciu, ktora vrati pravdepodobnost v intervale <0,1> preto staci len zaokruhlenie
            y_pred = np.round(prob, 0)
            present_classes = ['Benígne', 'Malígne']  # Binary classification

        if (y == 0).all():
            print('Vybrané okno(á) obsahuje(ú) iba benígnu prevádzku !')
            return
        elif (y_pred == 0).all():
            print('Predikcia obsahuje len benígnu prevádzku !')
            return

        with open(const.metrics.report[on_test_set].format(self.model_path, self.model_name, anomaly_count), 'w') as f:
           f.write(classification_report(y, y_pred, labels=np.unique(y), target_names=present_classes, zero_division=0))
        
        plot_confusion_matrix(y, y_pred, self.model_number, self.is_cat_multiclass,
            const.metrics.conf_m[on_test_set].format(self.model_path, self.model_name, anomaly_count))
        plot_roc_auc(y, prob, self.model_number, self.trainY, self.model_name,
            const.metrics.roc_auc[on_test_set].format(self.model_path, self.model_name, anomaly_count))

    def run_sweep(self, early_stop_patience, sweep_config_random):
        project_name = 'multiclass' if self.is_cat_multiclass else 'binary' + '_categorical'
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