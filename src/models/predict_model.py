from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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


def predict(
    ts_handler,
    model_name,
    model_number
):
    model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))

    train_predict = model.predict(ts_handler.train_generator)
    test_predict = model.predict(ts_handler.test_generator)

    train_real = get_y_from_generator(ts_handler.train_generator, ts_handler.n_features)
    test_real = get_y_from_generator(ts_handler.test_generator, ts_handler.n_features)

    attack_train_data = get_y_from_generator(ts_handler.attack_train, ts_handler.n_features)
    attack_test_data = get_y_from_generator(ts_handler.attack_test, ts_handler.n_features)

    calculate_metrics(
        train_real, 
        test_real, 
        train_predict, 
        test_predict, 
        ts_handler.features,
        model_number
    )

    train_data = inverse_transform(train_real, ts_handler)  # replace with this if want to plot against train data
    test_data = inverse_transform(test_real, ts_handler)

    save_prediction_plots(
        attack_train_data, 
        attack_test_data,
        inverse_transform(test_predict, ts_handler), 
        ts_handler.features, 
        ts_handler.n_features,
        model_number
    )
       

def calculate_metrics(train_real, test_real, train_predict, test_predict, extracted_features, model_number):
    with open(const.MODEL_METRICS_PATH.format(model_number), 'w') as f:
        for i in range(len(extracted_features)):
            mape_score = mean_absolute_percentage_error(test_real[:,i], test_predict[:,i])
            mse_test_score = mean_squared_error(test_real[:, i], test_predict[:,i])
            rmse_train_score = math.sqrt(mean_squared_error(train_real[:,i], train_predict[:,i]))   # [first_row:last_row,column_0] - all rows in column i
            rmse_test_score = math.sqrt(mse_test_score)

            f.write(extracted_features[i] + '\n')
            f.write('MAPE test score:  %.2f\n' % mape_score)
            f.write('MSE test score:   %.2f\n' % mse_test_score)
            f.write('RMSE train score: %.2f\n' % rmse_train_score)
            f.write('RMSE test score:  %.2f\n\n' % rmse_test_score)


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