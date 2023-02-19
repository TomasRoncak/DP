import sys

import numpy as np
import pandas as pd

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from pathlib import Path

from keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.models import Sequential
from preprocess_data import format_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import constants as const
from models.functions import (get_callbacks, get_filtered_classes,
                              get_optimizer, load_best_model,
                              plot_confusion_matrix, plot_roc_auc,
                              pretty_print_detected_attacks)


class ClassificationModel:
    def __init__(self, model_number, model_name, is_cat_multiclass):
        self.model_number = model_number
        self.model_name = model_name
        self.is_cat_multiclass = is_cat_multiclass
        self.is_model_reccurent = self.model_name in ['lstm', 'gru']
        self.model_path = const.WHOLE_CLASSIFICATION_MULTICLASS_MODEL_PATH.format(model_number) \
                          if self.is_cat_multiclass else const.WHOLE_CLASSIFICATION_BINARY_MODEL_PATH.format(model_number)
        
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

        df = pd.read_csv(const.CAT_TRAIN_DATASET_PATH)
        trainX, trainY = format_data(df, self.is_cat_multiclass, self.is_model_reccurent)
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

        num_categories = df.iloc[:,-1].nunique()

        if num_categories > 2:  
            loss = SparseCategoricalCrossentropy()
            last_activation = 'softmax'
        else:  # Unikatne hodnoty su 1 a 0 -> binarna klasifikacia
            num_categories = 1
            loss = BinaryCrossentropy()
            last_activation = 'sigmoid'

        model = Sequential()
        if self.model_name == 'mlp':
            model.add(Dense(1024, activation=activation, input_dim=trainX.shape[1]))
            model.add(Dropout(dropout))
            model.add(Dense(768, activation=activation))
            model.add(Dropout(dropout))
            model.add(Dense(num_categories, activation=last_activation))
        elif self.model_name == 'cnn':
            model.add(Conv1D(filters=32, padding='same', kernel_size=2, activation=activation, input_shape=(trainX.shape[1],1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation=activation))
            model.add(Dropout(dropout)),
            model.add(Flatten())
            model.add(Dense(num_categories, activation=last_activation))
        elif self.model_name == 'lstm':
            model.add(LSTM(20, return_sequences=True, input_dim=trainX.shape[2]))
            model.add(LSTM(20, input_dim=trainX.shape[2]))
            model.add(Dropout(dropout))
            model.add(Dense(num_categories, activation=last_activation))
        elif self.model_name == 'gru':
            model.add(GRU(blocks, input_dim=trainX.shape[2]))
            model.add(Dropout(dropout))
            model.add(Dense(num_categories, activation=last_activation))
        elif self.model_name == 'cnn_lstm':
            model.add(Conv1D(filters=64, padding="same", kernel_size=2, activation=activation, input_shape=(trainX.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(blocks))
            model.add(Dropout(dropout))
            model.add(Dense(num_categories, activation=last_activation))
        else:
            raise Exception('Nepodporovaný typ modelu !')

        optimizer = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        run = wandb.init(project="dp_cat", entity="tomasroncak")

        model.fit(
                trainX,
                trainY,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(valX, valY),
                callbacks=[get_callbacks(
                            self.model_number, 
                            self.model_name, 
                            self.is_cat_multiclass, 
                            patience
                           )],
                verbose=1
        )

        #model.save(const.SAVE_CAT_MODEL_PATH.format(self.model_number, model_name) + 'model.h5')
        run.finish()

    def categorize_attacks(self, on_test_set, an_detect_time):
        self.model = load_best_model(self.model_number, self.model_name, model_type='cat', is_cat_multiclass=self.is_cat_multiclass)

        if on_test_set:
            test_df = pd.read_csv(const.CAT_TEST_DATASET_PATH)
            x, y = format_data(test_df, self.is_cat_multiclass, self.is_model_reccurent)
        elif not an_detect_time:
            print('Časové okno pre klasifikáciu nebolo nájdené !')
            return
        else:
            df = pd.read_csv(const.WHOLE_DATASET_PATH, parse_dates=[const.TIME])
            windowed_data = df[(df[const.TIME] >= an_detect_time[0]) & (df[const.TIME] <= an_detect_time[1])]
            x, y = format_data(windowed_data, self.is_model_reccurent)
        
        prob = self.model.predict(x, verbose=0)

        self.calculate_classification_metrics(y, prob, on_test_set)
        if not on_test_set:
            pretty_print_detected_attacks(prob)

    def calculate_classification_metrics(self, y, prob, on_test_set):
        if on_test_set:
            METRICS_PATH = const.METRICS_CLASSIFICATION_TEST_FOLDER_PATH
            REPORT_PATH = const.MODEL_CLASSIFICATION_METRICS_TEST_PATH
            CONFUSION_MATRIX_PATH = const.MODEL_CONF_TEST_MATRIX_PATH
            ROC_AUC_PATH = const.MODEL_METRICS_ROC_TEST_PATH
        else:
            METRICS_PATH = const.METRICS_CLASSIFICATION_WINDOW_FOLDER_PATH
            REPORT_PATH = const.MODEL_CLASSIFICATION_METRICS_WINDOW_PATH
            CONFUSION_MATRIX_PATH = const.MODEL_CONF_MATRIX_PATH
            ROC_AUC_PATH = const.MODEL_METRICS_ROC_PATH
        
        Path(METRICS_PATH.format(self.model_path, self.model_name)).mkdir(parents=True, exist_ok=True)

        if self.is_cat_multiclass:
            # Pre multiclass pouzivame Softmax aktivacnu funkciu, ktora vrati pravdepodobnost pre kazdu triedu
            # argmax funkcia vrati index s najvyssou hodnotou (pravdepodobnostou)
            y_pred = np.argmax(prob, axis=-1)  
        else:
            # Pre binary pouzivame Sigmoid aktivacnu funkciu, ktora vrati pravdepodobnost v intervale <0,1> preto staci len zaokruhlenie
            y_pred = np.round(prob, 0)

        if (y == 0).all():
            print('Vybrané okno(á) obsahuje(ú) iba benígnu prevádzku !')
            return
        elif (y_pred == 0).all():
            print('Predikcia obsahuje len benígnu prevádzku !')
            return

        if self.is_cat_multiclass:  # Multiclass classification
            all_classes = get_filtered_classes()
            classes_values = np.unique(y)
            present_classes = [all_classes[x] for x in classes_values]
        else:
            present_classes = ['Benígne', 'Malígne']  # Binary classification

        with open(REPORT_PATH.format(self.model_path, self.model_name), 'w') as f:
           f.write(classification_report(y, y_pred, labels=np.unique(y_pred), target_names=present_classes))
        
        plot_confusion_matrix(y, y_pred, self.model_number, present_classes, CONFUSION_MATRIX_PATH.format(self.model_path, self.model_name))
        plot_roc_auc(y, prob, self.model_number, self.is_cat_multiclass, ROC_AUC_PATH.format(self.model_path, self.model_name))

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

    def wandb_train(self, early_stop_patience):
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