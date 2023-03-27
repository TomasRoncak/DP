import csv
import math
import sys
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from pathlib import Path

from keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from tcn import TCN

import constants as const
from models.functions import (format_and_print_collective_anomaly,
                              get_callbacks, get_optimizer,
                              get_y_from_generator, load_best_model,
                              pretty_print_point_anomaly,
                              pretty_print_window_ok)


class AnomalyModel:
    def __init__(self, ts_handler, model_number, model_name, window_size, patience_limit):
        self.ts_handler = ts_handler
        self.n_features = len(ts_handler.features)
        self.whole_real_data = get_y_from_generator(self.n_features, ts_handler.attack_data_generator)
        self.model_number = model_number
        self.model_name = model_name
        self.patience_limit = patience_limit
        self.window_size = window_size
        self.collective_anomaly_count = 0
        self.an_detection_time = ()
        self.exceeding = 0
        self.normal = 0

        Path(const.MODEL_PREDICTIONS_BENIGN_PATH.format(model_number, model_name)).mkdir(parents=True, exist_ok=True)
        Path(const.anomaly_metrics[True].format(model_number, model_name)).mkdir(parents=True, exist_ok=True)
        Path(const.anomaly_metrics[False].format(model_number, model_name)).mkdir(parents=True, exist_ok=True)

    def train_anomaly_model(
        self,
        learning_rate,
        optimizer,
        early_stop_patience,
        epochs,
        dropout,
        blocks,
        activation,
        momentum
    ):
        n_steps = self.ts_handler.benign_train_generator.length
        n_features = self.n_features

        model = Sequential()
        if self.model_name == 'cnn':
            model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation=activation, input_shape=(n_steps, n_features)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout)),
            model.add(Flatten())
            model.add(Dense(25, activation=activation))
        elif self.model_name == 'lstm':
            model.add(LSTM(blocks, activation=activation, input_shape=(n_steps, n_features)))
        elif self.model_name == 'gru':
            model.add(GRU(blocks, input_shape=(n_steps, n_features)))
        elif self.model_name == 'cnn_lstm':
            model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation=activation, input_shape=(n_steps, n_features)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(blocks, activation=activation))
            model.add(Dropout(dropout))
        elif self.model_name == 'tcn':
            model.add(TCN(input_shape=(n_steps, n_features), nb_filters=256))
            model.add(Dropout(dropout))
        else:
            raise Exception('Nepodporovaný typ modelu !')
        model.add(Dense(n_features))
        
        optimizer_fn = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
        model.compile(optimizer=optimizer_fn, loss='mse')

        run = wandb.init(
            project='ts_prediction',
            group=self.model_name,
            entity='tomasroncak'
        )

        start = dt.now()
        model.fit(
            self.ts_handler.benign_train_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[get_callbacks(
                self.model_number, 
                self.model_name, 
                early_stop_patience
                )]
        )
        print('Tréning modelu {0} prebiehal {1} sekúnd.'.format(self.model_name, (dt.now() - start).seconds))

        #model.save(const.save_model[None].format(self.model_number, self.model_name) + 'model.h5')
        run.finish()

    def predict_ts(self, on_test_set, categorize_attacks_func=None):
        self.model = load_best_model(self.model_number, self.model_name, model_type='an')
        data_generator = self.ts_handler.benign_test_generator if on_test_set else self.ts_handler.attack_data_generator
        data_pred = []
        real_data_inversed = self.ts_handler.inverse_transform(self.whole_real_data)

        for i in range(len(data_generator)):
            x, y = data_generator[i]
            # x => conf.n_steps of data to be used for prediction - e.g. 0-4, 1-5, ...
            # y => real data at time step conf.n_steps + i - e.g. 5+0, 5+1, ...
            curr_time = self.ts_handler.time[i]
            curr_pred = self.model.predict(x, verbose=0)
            data_pred.append(curr_pred[0])

            if on_test_set:
                err = mse(curr_pred, y)
                pretty_print_window_ok(curr_time, err)
            elif self.is_anomaly_detected(curr_pred, y, curr_time, i):
                self.collective_anomaly_count += 1
                self.an_detection_time = format_and_print_collective_anomaly(self.first_an_detection_time, curr_time)
                categorize_attacks_func(an_detect_time=self.an_detection_time, anomaly_count=self.collective_anomaly_count)
                pred_data_inversed = self.ts_handler.inverse_transform(np.array(data_pred))
                self.save_plots(real_data_inversed, pred_data_inversed)

        predict_data_inversed = self.ts_handler.inverse_transform(data_pred)
        if on_test_set:
            self.save_benign_ts_plots(predict_data_inversed, show_full_data=False)
        self.calculate_regression_metrics(predict_data_inversed, on_test_set=on_test_set)

    def is_anomaly_detected(self, curr_data_pred, curr_data_real, curr_time, i):
        if not i:  # No data yet to detect on
            return False
        
        threshold = self.calculate_anomaly_threshold(i)
        err = mse(curr_data_pred, curr_data_real)

        if err <= threshold:
            self.normal += 1
            pretty_print_window_ok(curr_time, err)
        elif err > threshold:
            if self.exceeding == 0:  # Zaznamenaj prvu bodovu anomaliu
                self.first_an_detection_time = curr_time
                self.normal = 0
            self.exceeding += 1
            self.ts_handler.attack_data_generator.data[i] = curr_data_pred  # Replace real data with prediction of data

            pretty_print_point_anomaly(err, threshold, curr_time, self.window_size, self.exceeding, self.patience_limit)

        if self.normal >= 5:  # If after last exceeding comes 5 normals, reset exceeding to 0
            self.exceeding = 0
            
        if self.exceeding == self.patience_limit:
            self.exceeding = 0
            self.normal = 0
            return True
    
    def calculate_anomaly_threshold(self, i):
        q1, q3 = np.percentile(self.whole_real_data[:i], [25, 96])  # Calculate threshold according to whole data to the time point of 'i'
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        return upper_bound

    def calculate_regression_metrics(self, predict_data, on_test_set):
        real_data = get_y_from_generator(self.n_features, self.ts_handler.benign_test_generator) if on_test_set else self.whole_real_data
        real_data = self.ts_handler.inverse_transform(real_data)
        real_data = real_data[0:len(predict_data)]  # Slice data, if predicted data are shorter (detected anomaly stops prediction)

        with open(const.anomaly_metrics[on_test_set].format(self.model_number, self.model_name) + const.REPORT_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'mae', 'mape', 'mse', 'rmse'])   # Header
            for i in range(len(self.ts_handler.features)):
                mae_score = mae(real_data[:,i], predict_data[:,i])
                mape_score = mape(real_data[:,i], predict_data[:,i])
                mse_score = mse(real_data[:,i], predict_data[:,i])

                writer.writerow([
                    self.ts_handler.features[i], 
                    round(mae_score, 2), 
                    round(mape_score, 2), 
                    round(mse_score, 2), 
                    round(math.sqrt(mse_score), 2)
                ])
    
    def save_benign_ts_plots(self, prediction_data, show_full_data):
        train_data = get_y_from_generator(self.n_features, self.ts_handler.benign_train_generator)
        train_data = self.ts_handler.inverse_transform(train_data)

        test_data = get_y_from_generator(self.n_features, self.ts_handler.benign_test_generator)
        test_data = self.ts_handler.inverse_transform(test_data)

        if not show_full_data:                              # Display only half of the train data
            train_len = len(train_data)
            train_data = train_data[int(train_len/3)*2:]
            shortened_time = self.ts_handler.time[int(train_len/3)*2:train_len + len(train_data)]

        begin = len(train_data)                             # Beginning is where train data ends
        end = begin + len(test_data)                        # End is where predicted data ends

        whole_data = np.append(train_data, test_data)             
        whole_data = whole_data.reshape(-1, self.n_features)

        pred_data = np.empty_like(whole_data)
        pred_data[:, :] = np.nan                            # First : stands for first and the second : for the second dimension
        pred_data[begin:end, :] = prediction_data           # Insert predicted values

        self.save_plots(whole_data, pred_data, time=shortened_time)

    def save_plots(self, real_data, prediction_data, time=None):
        print('Ukladám grafy časových tokov...')
        sns.set_style('darkgrid')
        if self.an_detection_time != ():
            Path(const.MODEL_PREDICTIONS_ATTACK_PATH \
                .format(self.model_number, self.model_name, self.collective_anomaly_count)) \
                .mkdir(parents=True, exist_ok=True)
            time = self.ts_handler.time
            start = mdates.date2num(self.an_detection_time[0])
            fig_name = const.MODEL_PREDICTIONS_ATTACK_PATH
        else:
            fig_name = const.MODEL_PREDICTIONS_BENIGN_PATH
    
        for i in range(self.n_features): 
            real_feature = [item[i] for item in real_data]
            predict_feature = [item[i] for item in prediction_data]

            plt.rcParams['figure.figsize'] = (45, 15)
            plt.plot(time[:len(real_feature)], real_feature, label='Realita', color='#017b92', linewidth=3)
            plt.plot(time[:len(predict_feature)], predict_feature, label='Predikcia', color='#f97306', linewidth=3) 
            plt.xticks(rotation='vertical', fontsize=40)
            plt.yticks(fontsize=40)
            plt.legend(fontsize=40)
            
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            if self.an_detection_time != ():
                plt.title('{0} - ({1} - {2})'.format(
                    self.ts_handler.features[i], 
                    self.an_detection_time[0].strftime('%m-%d %H:%M'), 
                    self.an_detection_time[1].strftime('%m-%d %H:%M')
                    ), 
                    fontsize=50
                )
                plt.arrow(start, 0, 0, predict_feature[len(predict_feature)-1]*0.75, facecolor='red', width=0.003, head_length=np.mean(predict_feature)/6, length_includes_head=True)
                plt.savefig(fig_name.format(self.model_number, self.model_name, self.collective_anomaly_count) + self.ts_handler.features[i], bbox_inches='tight')
            else:
                plt.title(self.ts_handler.features[i], fontsize=50)
                plt.savefig(fig_name.format(self.model_number, self.model_name) + self.ts_handler.features[i], bbox_inches='tight')
            plt.close()

    def run_sweep(self, patience, sweep_config_random):
        wandb.agent(
            wandb.sweep(sweep_config_random, project='ts_prediction_sweep'), 
            function=lambda: self.wandb_train(patience), 
            count=40
        )

    def wandb_train(self, patience):
        run = wandb.init(project='ts_prediction', group=self.model_name, entity='tomasroncak')
        self.train_anomaly_model(
            wandb.config.learning_rate,
            wandb.config.optimizer,
            patience,
            wandb.config.epochs,
            wandb.config.dropout,
            wandb.config.blocks,
            wandb.config.activation,
            wandb.config.momentum
        )
        run.finish()