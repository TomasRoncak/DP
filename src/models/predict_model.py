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
    model_number,
    save_plots
):
    extracted_features = ts_handler.features

    model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))

    train_real = get_y_from_generator(ts_handler.train_generator, len(extracted_features))
    train_predict = model.predict(ts_handler.train_generator)

    test_real = get_y_from_generator(ts_handler.test_generator, len(extracted_features))
    test_predict = model.predict(ts_handler.test_generator)

    metrics(train_real, test_real, train_predict, test_predict, model_number, extracted_features)

    if save_plots:
        train_real_inversed = inverse_transform(train_real, ts_handler)
        test_real_inversed = inverse_transform(test_real, ts_handler)
        test_predict_inversed = inverse_transform(test_predict, ts_handler)

        save_model_plots(train_real_inversed, test_real_inversed, test_predict_inversed, extracted_features, model_number)
       

def metrics(train_real, test_real, train_predict, test_predict, model_number, extracted_features):
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


def save_model_plots(train_y, y, y_hat, extracted_features, model_number):
    n_features = len(extracted_features)
    begin = len(train_y)                  # beginning is where train data ends
    end = begin + len(y_hat)              # end is where predicted data ends

    data = np.append(train_y, y)          # whole dataset
    data = data.reshape(-1, n_features)

    y_plot = np.empty_like(data)          # create empty np array with shape like given array
    y_plot[:, :] = np.nan                 # fill it with nan
    y_plot[begin:end, :] = y              # insert real values

    y_hat_plot = np.empty_like(data)
    y_hat_plot[:, :] = np.nan             # first : stands for first and the second : for the second dimension
    y_hat_plot[begin:end, :] = y_hat      # insert predicted values

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