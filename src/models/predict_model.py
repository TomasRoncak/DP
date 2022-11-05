from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from generate_time_series import generate_time_series, minmax_scaler, stand_scaler

import config as conf
import constants as const

def get_y_from_generator(gen):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        y = batch_y if y is None else np.append(y, batch_y)
    return y.reshape((-1, conf.n_featues))


def inverse_transform(predict):
    tmp = stand_scaler.inverse_transform(predict)
    return minmax_scaler.inverse_transform(tmp)


def predict(
    model_name,
    window_size,
    n_steps,
    n_featues,
    model_number,
    save_plots
):
    train_ts_generator, _, extracted_features = generate_time_series(window_size, n_steps, get_train=True)
    test_ts_generator, _, _ = generate_time_series(window_size, n_steps, get_train=False)

    model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))

    train_real = inverse_transform(get_y_from_generator(train_ts_generator))
    train_predict = model.predict(train_ts_generator)
    train_predict = inverse_transform(train_predict)

    test_real = inverse_transform(get_y_from_generator(test_ts_generator))
    test_predict = model.predict(test_ts_generator)
    test_predict = inverse_transform(test_predict)

    metrics(model, test_ts_generator, train_real, test_real, train_predict, test_predict, model_number, n_featues, extracted_features)

    if save_plots:
        save_plots(train_real, test_real, test_predict, n_featues, extracted_features, model_number)
       

def metrics(model, test_ts_generator, train_real, test_real, train_predict, test_predict, model_number, n_featues, extracted_features):
    with open(const.MODEL_PATH.format(model_number) + 'metrics.txt', 'w') as f:
        score = model.evaluate(test_ts_generator, verbose=0)
        f.write('Score on test set: {:0.2f}%\n\n'.format(float(score)*100))

        for i in range(n_featues):
            mape_score = mean_absolute_percentage_error(test_real[:,i], test_predict[:,i])
            rmse_train_score = math.sqrt(mean_squared_error(train_real[:,i], train_predict[:,i]))   # [first_row:last_row,column_0] - all rows in column i
            rmse_test_score = math.sqrt(mean_squared_error(test_real[:, i], test_predict[:,i]))

            f.write(extracted_features[i] + '\n')
            f.write('MAPE test score:  %.2f\n' % mape_score)
            f.write('RMSE train score: %.2f\n' % rmse_train_score)
            f.write('RMSE test score:  %.2f\n\n' % rmse_test_score)


def save_plots(train_y, y, y_hat, n_featues, extracted_features, model_number):
    begin = len(train_y)                  # beginning is where train data ends
    end = begin + len(y_hat)              # end is where predicted data ends

    data = np.append(train_y, y)          # whole dataset
    data = data.reshape(-1, n_featues)

    y_plot = np.empty_like(data)          # create empty np array with shape like given array
    y_plot[:, :] = np.nan                 # fill it with nan
    y_plot[begin:end, :] = y              # insert real values

    y_hat_plot = np.empty_like(data)
    y_hat_plot[:, :] = np.nan             # first : stands for first and the second : for the second dimension
    y_hat_plot[begin:end, :] = y_hat      # insert predicted values

    for i in range(n_featues): 
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