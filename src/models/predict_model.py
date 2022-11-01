from keras.models import load_model
from sklearn.metrics import mean_squared_error

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
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,conf.n_featues))
    return y

def predict(
    model_name,
    window_size,
    n_steps,
    n_featues,
    model_number,
    save_plots
):
    model = load_model(const.SAVE_MODEL_PATH.format(model_number, model_name))

    train_ts_generator, n_features, extracted_features = generate_time_series(window_size, n_steps, get_train=True)
    test_ts_generator, n_features, _ = generate_time_series(window_size, n_steps, get_train=False)

    score = model.evaluate(test_ts_generator, verbose=0)
    print("Score on test set: %.2f" % score)

    trainPredict = model.predict(train_ts_generator)
    testPredict = model.predict(test_ts_generator)

    trainPredict = stand_scaler.inverse_transform(trainPredict)
    testPredict = stand_scaler.inverse_transform(testPredict)

    trainPredict = minmax_scaler.inverse_transform(trainPredict)
    testPredict = minmax_scaler.inverse_transform(testPredict)

    trainY = get_y_from_generator(train_ts_generator)
    testY = get_y_from_generator(test_ts_generator)

    trainY = stand_scaler.inverse_transform(trainY)
    testY = stand_scaler.inverse_transform(testY)

    trainY = minmax_scaler.inverse_transform(trainY)
    testY = minmax_scaler.inverse_transform(testY)

    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    begin = len(trainY)
    end = begin + len(testPredict)

    data_org = np.append(trainY, testY)
    data_org = data_org.reshape(-1,10)

    testYPlot = np.empty_like(data_org)
    testYPlot[:, :] = np.nan
    testYPlot[begin:end, :] = testY

    testPredictPlot = np.empty_like(data_org)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[begin:end, :] = testPredict

    if save_plots:
        for i in range(n_featues): 
            data_column = [item[i] for item in data_org] # train data
            reality_column = [item[i] for item in testYPlot]
            predict_column = [item[i] for item in testPredictPlot]

            # plot baseline and predictions
            plt.rcParams["figure.figsize"] = (12,3)
            plt.plot(data_column, label ='Train & Test data', color="#017b92")
            plt.plot(reality_column, color="#017b92") 
            plt.plot(predict_column, label ='Prediction', color="#f97306") 
            plt.title(extracted_features[i])
            plt.legend()
            plt.savefig('models/model_{0}/predictions/{1}'.format(model_number, extracted_features[i]), dpi=400)
            plt.close()