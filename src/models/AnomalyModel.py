import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from pathlib import Path

from keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.models import Sequential
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

import constants as const
from models.functions import (format_date, get_callbacks, get_optimizer,
                              get_y_from_generator, load_best_model,
                              pretty_print_collective_anomaly,
                              pretty_print_point_anomaly,
                              pretty_print_window_ok)


class AnomalyModel:
    def __init__(self, ts_handler, model_number, model_name, window_size, patience_limit):
        self.ts_handler = ts_handler
        self.n_features = ts_handler.n_features
        self.model = load_best_model(model_number, model_name, model_type='an')
        self.model_number = model_number
        self.model_name = model_name
        self.patience_limit = patience_limit
        self.window_size = window_size
        self.point_anomaly_detected = False
        self.anomaly_detection_time = ()
        self.exceeding = 0
        self.normal = 0

        Path(const.MODEL_PREDICTIONS_BENIGN_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)
        Path(const.MODEL_PREDICTIONS_ATTACK_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)
        Path(const.METRICS_REGRESSION_FOLDER_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)
        
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
            model.add(LSTM(16, input_shape=(n_steps, n_features), return_sequences=True))
            model.add(LSTM(8))
        elif self.model_name == 'gru':
            model.add(GRU(blocks, input_shape=(n_steps, n_features)))
        model.add(Dense(n_features))
        
        optimizer_fn = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
        model.compile(optimizer=optimizer_fn, loss='mse')

        run = wandb.init(project="dp_an", entity="tomasroncak")

        model.fit(
            self.ts_handler.benign_train_generator,
            epochs=epochs,
            verbose=0,
            callbacks=[get_callbacks(
                self.model_number, 
                self.model_name, 
                const.ANOMALY_MODEL_FOLDER, 
                early_stop_patience
                )]
        )

        #model.save(const.SAVE_ANOMALY_MODEL_PATH.format(model_number, model_name) + 'model.h5')
        run.finish()

    def predict_on_benign_ts(self):
        data_real = get_y_from_generator(self.n_features, self.ts_handler.benign_test_generator)
        data_generator = self.ts_handler.benign_test_generator
        data_pred = []
        time = self.ts_handler.time
        
        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            curr_time = time[i]
            curr_pred = self.model.predict(x, verbose=0)
            err = mse(data_real[i], curr_pred[0])
            
            pretty_print_window_ok(curr_time, err)

            data_pred.append(curr_pred[0])

        train_data = get_y_from_generator(self.n_features, self.ts_handler.benign_train_generator)
        test_data = get_y_from_generator(self.n_features, self.ts_handler.benign_test_generator)
        
        train_inversed = self.ts_handler.inverse_transform(train_data)
        test_inversed = self.ts_handler.inverse_transform(test_data)
        predict_inversed = self.ts_handler.inverse_transform(data_pred)

        self.calculate_regression_metrics(test_inversed, predict_inversed, on_test_set=True)
        self.create_radar_plot(on_test_set=True)
        self.save_benign_ts_plots(
            train_inversed, 
            test_inversed, 
            predict_inversed, 
            time, 
            show_full_data=False
        )

    def predict_on_attack_ts(self):
        data_real = get_y_from_generator(self.n_features, self.ts_handler.attack_data_generator)
        data_pred = []
        data_generator = self.ts_handler.attack_data_generator
        time = self.ts_handler.attack_time

        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            curr_time = time[i] 
            curr_pred = self.model.predict(x, verbose=0)
            
            data_pred.append(curr_pred[0])

            if self.detect_anomaly_ts(data_real, curr_pred, i, curr_time):
                start_time = format_date(self.first_anomaly_detection_time)
                stop_time = format_date(curr_time)
                self.anomaly_detection_time = (start_time, stop_time)

                pretty_print_collective_anomaly(start_time, stop_time)
                break

        attack_real_inversed = self.ts_handler.inverse_transform(data_real, attack_data=True)
        attack_predict_inversed = self.ts_handler.inverse_transform(np.array(data_pred), attack_data=True)

        self.calculate_regression_metrics(attack_real_inversed, attack_predict_inversed, on_test_set=False)
        self.create_radar_plot(on_test_set=False)
        self.save_plots(
                attack_real_inversed, 
                attack_predict_inversed, 
                time, 
                is_attack=True
        )

    def detect_anomaly_ts(self, real, pred, i, curr_time):
        if not i:  # No data yet to detect on
            return False
        
        threshold = self.calculate_anomaly_threshold(real, i)  # TODO Mozno nie real ale predict ?
        err = mse(real[i], pred[0])

        if err <= threshold:
            pretty_print_window_ok(curr_time, err)
            if self.point_anomaly_detected:
                self.normal += 1
        elif err > threshold:
            if self.exceeding == 0:
                self.first_anomaly_detection_time = curr_time
            self.exceeding += 1
            self.point_anomaly_detected = True

            pretty_print_point_anomaly(err, threshold, curr_time, self.window_size, self.exceeding, self.patience_limit)

        if self.normal == 5 and self.exceeding != 0:  # If after last exceeding comes 5 normals, reset exceeding to 0
            self.point_anomaly_detected = False
            self.exceeding = 0
            self.normal = 0
            
        return self.exceeding == self.patience_limit
    
    def calculate_anomaly_threshold(self, data, i):
        q1, q3 = np.percentile(data[:i],[25, 95])
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        return upper_bound

    def calculate_regression_metrics(self, real_data, predict_data, on_test_set):
        METRICS_PATH = const.MODEL_REGRESSION_TEST_METRICS_PATH if on_test_set \
                       else const.MODEL_REGRESSION_WINDOW_METRICS_PATH
        real_data = real_data[0:len(predict_data)]  # Slice data, if predicted data are shorter (detected anomaly stops prediction)
        
        with open(METRICS_PATH.format(self.model_number), 'w') as f:
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

    def create_radar_plot(self, on_test_set):
        #TODO presunut do functions.py lebo to nema nic spolocne s touto triedou
        PATH = const.MODEL_REGRESSION_TEST_METRICS_PATH if on_test_set else const.MODEL_REGRESSION_WINDOW_METRICS_PATH
        features = self.ts_handler.features.copy()
        angles = np.linspace(0,2*np.pi,len(features), endpoint=False)
        angles = np.concatenate((angles,[angles[0]]))

        # fig = go.Figure()
        # model_number = 1
        # while True:
        #     METRICS_PATH = PATH.format(model_number)
        #     try:
        #         metrics = pd.read_csv(METRICS_PATH)
        #     except:
        #         break
            
        #     metrics = metrics['mape'].to_list()
        #     fig.add_trace(go.Scatterpolar(
        #         r=metrics,
        #         theta=features,
        #         fill='toself',
        #         name='Product A'
        #     ))

        #     model_number += 1

        # fig.update_layout(
        #     polar=dict(
        #         radialaxis=dict(
        #         visible=True,
        #         range=[0, 0.6]
        #         )),
        #     showlegend=False
        # )

        # fig.write_image(const.MODEL_REGRESSION_RADAR_CHART_PATH.format('test.png'))

        features.append(features[0])
        model_number = 1
        fig = plt.figure(figsize=(6, 10))
        ax = fig.add_subplot(polar=True)
        while True:
            METRICS_PATH = PATH.format(model_number)
            try:
                metrics = pd.read_csv(METRICS_PATH)
            except:
                break
            metrics = metrics['mape'].to_list()
            metrics.append(metrics[0])

            ax.plot(angles, metrics, label=self.model_name)
            model_number += 1
        
        ax.set_thetagrids(angles * 180/np.pi, features)
        var = 'test' if on_test_set else 'window'
        plt.tight_layout()
        plt.legend(bbox_to_anchor = (1.4, 0.6), loc='center right')
        plt.title('MAPE skóre', fontsize=15)
        plt.savefig(const.MODEL_REGRESSION_RADAR_CHART_PATH.format(var), bbox_inches='tight')

    def save_benign_ts_plots(self, train_data, test_data, prediction_data, time, show_full_data):
        if not show_full_data:                              # Display only half of the train data
            train_len = len(train_data)
            train_data = train_data[int(train_len/3)*2:]    
            time = time[int(train_len/3)*2:train_len + len(train_data)]

        begin = len(train_data)                             # Beginning is where train data ends
        end = begin + len(test_data)                        # End is where predicted data ends

        whole_data = np.append(train_data, test_data)             
        whole_data = whole_data.reshape(-1, self.ts_handler.n_features)

        pred_data = np.empty_like(whole_data)
        pred_data[:, :] = np.nan                            # First : stands for first and the second : for the second dimension
        pred_data[begin:end, :] = prediction_data           # Insert predicted values

        self.save_plots(whole_data, pred_data, time, is_attack=False)

    def save_plots(self, train_data, prediction_data, time, is_attack):
        print('Ukladám grafy ...')
        fig = const.MODEL_PREDICTIONS_ATTACK_PATH if is_attack \
              else const.MODEL_PREDICTIONS_BENIGN_PATH
    
        for i in range(self.ts_handler.n_features): 
            train_feature = [item[i] for item in train_data]
            predict_feature = [item[i] for item in prediction_data]

            ax = plt.gca()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format(int(x), ',')))

            plt.rcParams["figure.figsize"] = (45, 15)
            plt.plot(time[:len(train_feature)], train_feature, label ='Realita', color="#017b92", linewidth=3)
            plt.plot(predict_feature, label ='Predikcia', color="#f97306", linewidth=3) 
            plt.xticks(time[::30], rotation='vertical', fontsize=40)
            plt.yticks(fontsize=40)
            plt.tight_layout()
            plt.title(self.ts_handler.features[i], fontsize=50)
            plt.legend(fontsize=40)
            plt.savefig(fig.format(self.model_number) + self.ts_handler.features[i], dpi=400, bbox_inches='tight')
            plt.close()
        print('Ukladanie hotové !')

    def run_sweep(
        self,
        model_name,
        n_steps,
        patience,
        blocks,
        sweep_config_random
    ):
        wandb.agent(
            wandb.sweep(sweep_config_random, project='anomaly_' + model_name), 
            function=lambda: self.wandb_train(
                                n_steps,
                                patience,
                                blocks
                            ), 
            count=40
        )

    def wandb_train(
        self,
        n_steps,
        patience,
        blocks
    ):
        run = wandb.init(project='dp_anomaly', entity='tomasroncak')
        self.train_anomaly_model(
            n_steps,
            wandb.config.learning_rate,
            wandb.config.optimizer,
            patience,
            wandb.config.epochs,
            wandb.config.dropout,
            blocks,
            wandb.config.activation,
            wandb.config.momentum
        )
        run.finish()