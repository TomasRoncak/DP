from keras.models import load_model
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

class Prediction:
    def __init__(self, ts_handler, model_name, patience_limit, model_number):
        self.model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))
        self.ts_handler = ts_handler
        self.model_name = model_name
        self.model_number = model_number
        self.patience_limit = patience_limit
        self.exceeding = 0
        self.normal = 0
        self.point_anomaly_detected = False

        
    def get_y_from_generator(self, gen):
        y = None
        for i in range(len(gen)):
            batch_y = gen[i][1]
            y = batch_y if y is None else np.append(y, batch_y)
        return y.reshape((-1, self.ts_handler.n_features))

    
    def inverse_transform(self, predict, attack_data=False):
        if attack_data:
            tmp = self.ts_handler.attack_stand_scaler.inverse_transform(predict)
            return self.ts_handler.attack_minmax_scaler.inverse_transform(tmp)
        else:
            tmp = self.ts_handler.stand_scaler.inverse_transform(predict)
            return self.ts_handler.minmax_scaler.inverse_transform(tmp)

    
    def predict_benign(self):
        test_predict = self.model.predict(self.ts_handler.benign_test_generator)

        train_inversed = self.inverse_transform(self.get_y_from_generator(self.ts_handler.benign_train_generator))
        test_inversed = self.inverse_transform(self.get_y_from_generator(self.ts_handler.benign_test_generator))
        test_predict_inversed = self.inverse_transform(test_predict)

        self.save_prediction_plots(train_inversed, test_inversed, test_predict_inversed)

    
    def predict_attack(self):
        attack_real = self.get_y_from_generator(self.ts_handler.attack_data_generator)
        attack_predict = []
        data_generator = self.ts_handler.attack_data_generator

        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            pred = self.model.predict(x)
            attack_predict.append(pred[0])

            if self.detect_anomaly(attack_real, pred, i):
                print("ANOMALY !")
                break

        attack_predict = np.array(attack_predict)
        
        self.calculate_metrics(attack_real, attack_predict)
        self.save_attack_prediction_plots(
            self.inverse_transform(attack_real, True),
            self.inverse_transform(attack_predict, True)
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
                mse_score = mse(real_data[:, i], predict_data[:,i])

                f.write(self.ts_handler.features[i] + '\n')
                f.write('MAPE score:  {:.2f}\n'.format(mape_score))
                f.write('MSE score:   {:.2f}\n'.format(mse_score))
                f.write('RMSE score:  {:.2f}\n\n'.format(math.sqrt(mse_score)))


    def save_prediction_plots(self, train_data, test_data, prediction_data):
        begin = len(train_data)                             # beginning is where train data ends
        end = begin + len(test_data)                        # end is where predicted data ends

        data = np.append(train_data, test_data)             # whole dataset
        data = data.reshape(-1, self.ts_handler.n_features)

        y_plot = np.empty_like(data)                        # create empty np array with shape like given array
        y_plot[:, :] = np.nan                               # fill it with nan
        y_plot[begin:end, :] = test_data                    # insert real values

        y_hat_plot = np.empty_like(data)
        y_hat_plot[:, :] = np.nan                           # first : stands for first and the second : for the second dimension
        y_hat_plot[begin:end, :] = prediction_data          # insert predicted values

        for i in range(self.ts_handler.n_features): 
            train_column = [item[i] for item in data]
            reality_column = [item[i] for item in y_plot]
            predict_column = [item[i] for item in y_hat_plot]

            plt.rcParams["figure.figsize"] = (12, 3)
            plt.plot(train_column, label ='Train & Test data', color="#017b92")
            plt.plot(reality_column, color="#017b92") 
            plt.plot(predict_column, label ='Prediction', color="#f97306") 

            plt.title(self.ts_handler.features[i])
            plt.legend()
            plt.savefig(const.MODEL_PREDICTIONS_BENIGN_PATH \
                    .format(self.model_number) + self.ts_handler.features[i], dpi=400)
            plt.close()

            
    def save_attack_prediction_plots(self, train_data, prediction_data):
        for i in range(self.ts_handler.n_features): 
            reality = [item[i] for item in train_data]
            predict_column = [item[i] for item in prediction_data]

            plt.rcParams["figure.figsize"] = (12, 3)
            plt.plot(reality, label ='Reality', color="#017b92")
            plt.plot(predict_column, label ='Prediction', color="#f97306") 

            plt.title(self.ts_handler.features[i])
            plt.legend()
            plt.savefig(const.MODEL_PREDICTIONS_ATTACK_PATH \
                    .format(self.model_number) + self.ts_handler.features[i], dpi=400)
            plt.close()