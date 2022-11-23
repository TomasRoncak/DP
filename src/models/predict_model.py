from keras.models import load_model
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

def get_y_from_generator(gen, n_features):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        y = batch_y if y is None else np.append(y, batch_y)
    return y.reshape((-1, n_features))


def inverse_transform(predict, ts_handler):
    tmp = ts_handler.stand_scaler.inverse_transform(predict)
    return ts_handler.minmax_scaler.inverse_transform(tmp)

def attack_inverse_transform(predict, ts_handler):
    tmp = ts_handler.attack_stand_scaler.inverse_transform(predict)
    return ts_handler.attack_minmax_scaler.inverse_transform(tmp)


def calculate_threshold(data, i):
    q1, q3 = np.percentile(data[:i],[25,95]) # mozno nie attack_real ale attack_predict ?
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr) 
    return upper_bound

def predict(
    ts_handler,
    model_name,
    patience_limit,
    model_number
):
    model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))
    exceeding = normal = 0
    point_anomaly_detected = False

    train_inversed = inverse_transform(
        get_y_from_generator(ts_handler.benign_train_generator, ts_handler.n_features), 
        ts_handler
    )
    test_inversed = inverse_transform(
        get_y_from_generator(ts_handler.benign_test_generator, ts_handler.n_features), 
        ts_handler
    )
    test_predict = inverse_transform(
        model.predict(ts_handler.benign_test_generator),
        ts_handler
    )

    save_prediction_plots(
        train_inversed, 
        test_inversed,
        test_predict, 
        ts_handler.features, 
        ts_handler.n_features,
        model_number
    )

    attack_real = get_y_from_generator(ts_handler.attack_data_generator, ts_handler.n_features)
    attack_predict = []
    data_generator = ts_handler.attack_data_generator

    for i in range(len(data_generator)):
        x, _ = data_generator[i]
        pred = model.predict(x)
        attack_predict.append(pred[0])

        if i == 0:  # nemam este na com robit
            continue

        treshold = calculate_threshold(attack_real, i)
        err = mse(attack_real[i], pred[0])

        if err > treshold:
            exceeding += 1
            point_anomaly_detected = True
        elif point_anomaly_detected:
            normal += 1
        
        if normal == 10 and exceeding != 0:     # if after last exceeding comes 10 normals, reset exceeding to 0
            point_anomaly_detected = False
            exceeding = 0
            normal = 0
            
        if exceeding == patience_limit:
            print("ANOMALY !")
            break

    attack_predict = np.array(attack_predict)
    
    calculate_metrics(
        attack_real, 
        attack_predict, 
        ts_handler.features,
        model_number
    )
    
    save_attack_prediction_plots(
        attack_inverse_transform(attack_real, ts_handler),
        attack_inverse_transform(attack_predict, ts_handler),
        ts_handler.features, 
        ts_handler.n_features,
        model_number
    )


def calculate_metrics(real_data, predict_data, extracted_features, model_number):
    real_data = real_data[0:len(predict_data)]  # slice data, if predicted data are shorter (detected anomaly stops prediction)
    with open(const.MODEL_METRICS_PATH.format(model_number), 'w') as f:
        for i in range(len(extracted_features)):
            mape_score = mape(real_data[:,i], predict_data[:,i])
            mse_score = mse(real_data[:, i], predict_data[:,i])

            f.write(extracted_features[i] + '\n')
            f.write('MAPE score:  {:.2f}\n'.format(mape_score))
            f.write('MSE score:   {:.2f}\n'.format(mse_score))
            f.write('RMSE score:  {:.2f}\n\n'.format(math.sqrt(mse_score)))


def save_prediction_plots(train_data, test_data, prediction_data, extracted_features, n_features, model_number):
    begin = len(train_data)                         # beginning is where train data ends
    end = begin + len(test_data)                    # end is where predicted data ends

    data = np.append(train_data, test_data)          # whole dataset
    data = data.reshape(-1, n_features)

    y_plot = np.empty_like(data)                    # create empty np array with shape like given array
    y_plot[:, :] = np.nan                           # fill it with nan
    y_plot[begin:end, :] = test_data                # insert real values

    y_hat_plot = np.empty_like(data)
    y_hat_plot[:, :] = np.nan                       # first : stands for first and the second : for the second dimension
    y_hat_plot[begin:end, :] = prediction_data      # insert predicted values

    for i in range(n_features): 
        train_column = [item[i] for item in data]
        reality_column = [item[i] for item in y_plot]
        predict_column = [item[i] for item in y_hat_plot]

        plt.rcParams["figure.figsize"] = (12, 3)
        plt.plot(train_column, label ='Train & Test data', color="#017b92")
        plt.plot(reality_column, color="#017b92") 
        plt.plot(predict_column, label ='Prediction', color="#f97306") 

        plt.title(extracted_features[i])
        plt.legend()
        plt.savefig(const.MODEL_PREDICTIONS_PATH.format(model_number) + extracted_features[i], dpi=400)
        plt.close()

def save_attack_prediction_plots(train_data, prediction_data, extracted_features, n_features, model_number):
    for i in range(n_features): 
        reality = [item[i] for item in train_data]
        predict_column = [item[i] for item in prediction_data]

        plt.rcParams["figure.figsize"] = (12, 3)
        plt.plot(reality, label ='Reality', color="#017b92")
        plt.plot(predict_column, label ='Prediction', color="#f97306") 

        plt.title(extracted_features[i])
        plt.legend()
        plt.savefig(const.MODEL_PREDICTIONS_PATH.format(model_number) + extracted_features[i] + '_attack', dpi=400)
        plt.close()