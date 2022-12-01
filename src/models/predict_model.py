from keras.models import load_model
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape
from os import path, makedirs
from collections import Counter

import sys
import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const
from preprocess_data import format_data, get_classes

class Prediction:
    def __init__(self, ts_handler, an_model_name, cat_model_name, patience_limit, model_number):
        self.anomaly_model = load_model(const.SAVE_ANOMALY_MODEL_PATH.format(model_number, an_model_name.lower()))
        self.category_model = load_model(const.SAVE_CAT_MODEL_PATH.format(model_number, cat_model_name.lower()))
        self.ts_handler = ts_handler
        self.model_number = model_number
        self.patience_limit = patience_limit
        self.exceeding = 0
        self.normal = 0
        self.point_anomaly_detected = False
        self.anomaly_detection_time = ()

        if not path.exists(const.MODEL_PREDICTIONS_BENIGN_PATH.format(model_number)):
            makedirs(const.MODEL_PREDICTIONS_BENIGN_PATH.format(model_number))
            makedirs(const.MODEL_PREDICTIONS_ATTACK_PATH.format(model_number))

        
    def get_y_from_generator(self, gen):
        y = None
        for i in range(len(gen)):
            batch_y = gen[i][1]
            y = batch_y if y is None else np.append(y, batch_y)
        return y.reshape((-1, (self.ts_handler.n_features + 1)))    # time is considered

    
    def inverse_transform(self, predict, attack_data=False):
        if attack_data:
            tmp = self.ts_handler.attack_stand_scaler.inverse_transform(predict)
            return self.ts_handler.attack_minmax_scaler.inverse_transform(tmp)
        else:
            tmp = self.ts_handler.stand_scaler.inverse_transform(predict)
            return self.ts_handler.minmax_scaler.inverse_transform(tmp)


    def categorize_attack(self):
        df = pd.read_csv(const.WHOLE_DATASET, parse_dates=['time'])
        data = df[(df['time'] >= self.anomaly_detection_time[0]) & (df['time'] <= self.anomaly_detection_time[1])]
        classes = get_classes()

        x, y = format_data(data)
        x = np.asarray(x).astype('float32')

        prob = self.category_model.predict(x)
        pred = np.argmax(prob, axis=-1)

        err = mse(y, pred)
        res_list = list(Counter(pred).items())
        res_list.sort(key=lambda a: a[1], reverse=True)
        res = [(classes[x[0]], x[1]) for x in res_list]

        print("Detected attacks:")
        for x in res:
            if x[0] == 'Normal':
                continue
            print(x)
        print("MSE Error:", err)

    
    def predict_benign(self):
        data_generator = self.ts_handler.benign_test_generator
        benign_predict = []
        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            x = np.asarray(x[:,:,1:]).astype('float32')     # remove time
            test_predict = self.anomaly_model.predict(x)
            benign_predict.append(test_predict[0])

        train_data = self.get_y_from_generator(self.ts_handler.benign_train_generator)
        test_data = self.get_y_from_generator(self.ts_handler.benign_test_generator)
        
        train_time = np.squeeze(train_data[:, :1], axis = 1) 
        test_time = np.squeeze(test_data[:, :1], axis = 1) 
        time =  np.concatenate((train_time, test_time), axis = 0)

        train_inversed = self.inverse_transform(train_data[:,1:])
        test_inversed = self.inverse_transform(test_data[:,1:])
        predict_inversed = self.inverse_transform(benign_predict)

        self.save_benign_plots(train_inversed, test_inversed, predict_inversed, time)


    def predict_attack(self):
        attack_real = self.get_y_from_generator(self.ts_handler.attack_data_generator)
        attack_real = attack_real[:,1:]
        attack_predict = []
        data_generator = self.ts_handler.attack_data_generator

        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            curr_time = np.squeeze(x[:,:,:1], axis = 0) 
            x = np.asarray(x[:,:,1:]).astype('float32') # remove time

            pred = self.anomaly_model.predict(x)
            attack_predict.append(pred[0])

            if self.detect_anomaly(attack_real, pred, i):
                detection_time = curr_time[len(curr_time)-1][0]
                begin_time = datetime.datetime.strptime(detection_time, '%Y-%m-%d %H:%M:%S')
                end_time = begin_time + datetime.timedelta(minutes=3)
                self.anomaly_detection_time = (begin_time, end_time)

                print('First anomaly occured in window {0} - {1}.' \
                        .format(begin_time.strftime('%d.%m %H:%M'), end_time.strftime('%d.%m %H:%M')))
                break

        attack_predict = np.array(attack_predict)
        
        self.calculate_metrics(attack_real, attack_predict)
        self.save_plots(
                self.inverse_transform(attack_real, True), 
                self.inverse_transform(attack_predict, True), 
                time, 
                attack=True
        )
        
        
    def detect_anomaly(self, real, pred, i):
        if i == 0:  # nemam este na com robit
            return False
        
        treshold = self.calculate_anomaly_threshold(real, i)
        err = mse(real[i], pred[0])

        if err > treshold:
            self.exceeding += 1
            self.point_anomaly_detected = True
        elif self.point_anomaly_detected:
            self.normal += 1
        
        if self.normal == 10 and self.exceeding != 0:     # if after last exceeding comes 10 normals, reset exceeding to 0
            self.point_anomaly_detected = False
            self.exceeding = 0
            self.normal = 0
            
        return self.exceeding == self.patience_limit

    
    def calculate_anomaly_threshold(self, data, i):
        q1, q3 = np.percentile(data[:i],[25,95]) # mozno nie attack_real ale attack_predict ?
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr) 
        return upper_bound


    def calculate_metrics(self, real_data, predict_data):
        real_data = real_data[0:len(predict_data)]  # slice data, if predicted data are shorter (detected anomaly stops prediction)
        with open(const.MODEL_METRICS_PATH.format(self.model_number), 'w') as f:
            for i in range(len(self.ts_handler.features)):
                mape_score = mape(real_data[:,i], predict_data[:,i])
                mse_score = mse(real_data[:,i], predict_data[:,i])

                f.write(self.ts_handler.features[i] + '\n')
                f.write('MAPE score:  {:.2f}\n'.format(mape_score))
                f.write('MSE score:   {:.2f}\n'.format(mse_score))
                f.write('RMSE score:  {:.2f}\n\n'.format(math.sqrt(mse_score)))


    def save_benign_plots(self, train_data, test_data, prediction_data, time):
        begin = len(train_data)                             # beginning is where train data ends
        end = begin + len(test_data)                        # end is where predicted data ends

        data = np.append(train_data, test_data)             # whole dataset
        data = data.reshape(-1, self.ts_handler.n_features)

        y_hat_plot = np.empty_like(data)
        y_hat_plot[:, :] = np.nan                           # first : stands for first and the second : for the second dimension
        y_hat_plot[begin:end, :] = prediction_data          # insert predicted values

        self.save_plots(data, y_hat_plot, time, attack=False)


    def save_plots(self, train_data, prediction_data, time, attack):
        fig = const.MODEL_PREDICTIONS_ATTACK_PATH if attack else const.MODEL_PREDICTIONS_BENIGN_PATH
    
        for i in range(self.ts_handler.n_features): 
            train_feature = [item[i] for item in train_data]
            predict_feature = [item[i] for item in prediction_data]

            plt.rcParams["figure.figsize"] = (25, 10)
            plt.plot(time, train_feature, label ='Reality', color="#017b92")
            plt.plot(predict_feature, label ='Prediction', color="#f97306") 

            plt.xticks(time[::25],  rotation='vertical')
            plt.tight_layout()
            plt.title(self.ts_handler.features[i])
            plt.legend()
            plt.savefig(fig.format(self.model_number) + self.ts_handler.features[i], dpi=400)
            plt.close()