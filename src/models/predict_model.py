import datetime
import math
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

from data.preprocess_data import get_classes

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from preprocess_data import format_data, get_classes

import constants as const


class Prediction:
    def __init__(self, ts_handler, an_model_name, cat_model_name, patience_limit, model_number, window_size):
        self.anomaly_model = load_best_model(model_number, an_model_name, model_type='an')
        self.category_model = load_best_model(model_number, cat_model_name, model_type='cat')
        self.ts_handler = ts_handler
        self.model_number = model_number
        self.patience_limit = patience_limit
        self.exceeding = 0
        self.normal = 0
        self.point_anomaly_detected = False
        self.anomaly_detection_time = ()
        self.window_size = window_size

        Path(const.MODEL_PREDICTIONS_BENIGN_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)
        Path(const.MODEL_PREDICTIONS_ATTACK_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)
        Path(const.METRICS_FOLDER.format(model_number)).mkdir(parents=True, exist_ok=True)
    

    ## Anomaly model prediction ##
    def predict_benign_ts(self):
        data_generator = self.ts_handler.benign_test_generator
        benign_predict = []
        
        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            predict = self.anomaly_model.predict(x)
            benign_predict.append(predict[0])

        train_data = self.get_y_from_generator(self.ts_handler.benign_train_generator)
        test_data = self.get_y_from_generator(self.ts_handler.benign_test_generator)
        
        train_inversed = self.inverse_transform(train_data)
        test_inversed = self.inverse_transform(test_data)
        predict_inversed = self.inverse_transform(benign_predict)

        self.calculate_regression_metrics(test_inversed, predict_inversed, is_test_set=True)
        self.save_benign_ts_plots(
            train_inversed, 
            test_inversed, 
            predict_inversed, 
            self.ts_handler.time, 
            show_full_data=False
        )


    def predict_attacks_ts(self):
        attack_real = self.get_y_from_generator(self.ts_handler.attack_data_generator)
        time = self.ts_handler.attack_time
        attack_predict = []
        data_generator = self.ts_handler.attack_data_generator

        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            curr_time = time[i] 

            pred = self.anomaly_model.predict(x, verbose = 0)
            attack_predict.append(pred[0])

            if self.detect_anomaly_ts(attack_real, pred, i, curr_time):
                start_time = datetime.datetime.strptime('2015-' + self.first_anomaly_detection_time+ ':00', const.FULL_TIME_FORMAT)
                stop_time = datetime.datetime.strptime('2015-' + curr_time + ':00', const.FULL_TIME_FORMAT)
                self.anomaly_detection_time = (start_time, stop_time)

                print('\033[91m\033[1mUpozornenie\033[0m: Kolektívna anomália detegovaná v okne {0} až {1}!'.format(
                    start_time.strftime(const.PRETTY_TIME_FORMAT),
                    stop_time.strftime(const.PRETTY_TIME_FORMAT)
                    )
                )
                break

        attack_predict = np.array(attack_predict)
        
        attack_real_inversed = self.inverse_transform(attack_real, attack_data=True)
        attack_predict_inversed = self.inverse_transform(attack_predict, attack_data=True)

        self.calculate_regression_metrics(attack_real_inversed, attack_predict_inversed, is_test_set=False)
        self.save_plots(
                attack_real_inversed, 
                attack_predict_inversed, 
                time, 
                is_attack=True
        )
        
        
    def detect_anomaly_ts(self, real, pred, i, curr_time):
        if i == 0:  # nemam este na com robit
            return False
        
        treshold = self.calculate_anomaly_threshold(real, i)
        err = mse(real[i], pred[0])

        if err <= treshold:
            print('Okno {0} - chyba {1:.2f} \033[92mOK\033[0m'.format(curr_time, err))
            if self.point_anomaly_detected:
                self.normal += 1
        elif err > treshold:
            if self.exceeding == 0:
                self.first_anomaly_detection_time = curr_time
            self.exceeding += 1
            self.point_anomaly_detected = True

            curr_time = datetime.datetime.strptime(curr_time, const.TIME_FORMAT)
            window_end_time = (curr_time + datetime.timedelta(seconds=self.window_size)).strftime(const.PRETTY_TIME_FORMAT)
            curr_time = curr_time.strftime(const.PRETTY_TIME_FORMAT)
            print('\033[93mBodová anomália detegovaná v časovom okne {0} - {1} \033[0m'.format(curr_time, window_end_time), end='- ')
            print('chyba {0:.2f} prekročila prah {1:.2f} (trpezlivosť={2}/{3})'.format(err, treshold, self.exceeding, self.patience_limit))

        if self.normal == 5 and self.exceeding != 0:  # if after last exceeding comes 5 normals, reset exceeding to 0
            self.point_anomaly_detected = False
            self.exceeding = 0
            self.normal = 0
            
        return self.exceeding == self.patience_limit

    
    def calculate_anomaly_threshold(self, data, i):
        q1, q3 = np.percentile(data[:i],[25,95])  # mozno nie attack_real ale attack_predict ?
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr) 
        return upper_bound


    ## Classification model prediction ##
    def categorize_attacks_on_window(self):
        if not self.anomaly_detection_time:
            print('No window found to classify on!')
            return

        train_df = pd.read_csv(const.CAT_TRAIN_DATASET)
        df = pd.read_csv(const.WHOLE_DATASET, parse_dates=[const.TIME])
        data = df[(df[const.TIME] >= self.anomaly_detection_time[0]) & (df[const.TIME] <= self.anomaly_detection_time[1])]
        classes = get_classes()

        x, y = format_data(data)
        _, trainY = format_data(train_df)
        x = np.asarray(x).astype('float32')

        prob = self.category_model.predict(x, verbose=0)
        self.calculate_classification_metrics(trainY, y, prob, is_test_set=False)

        res_list = list(Counter(np.argmax(prob, axis=-1)).items())
        res_list.sort(key=lambda a: a[1], reverse=True)
        res = [(classes[x[0]], x[1]) for x in res_list]

        print('\033[91m\033[1mUpozornenie\033[0m: Časové okno obsahuje útoky na sieť!')
        print('\033[1mDetegované kategórie útokov\033[0m:')
        for x in res:
            if x[0] == 'Normal':
                continue
            print('{0} ({1}x)'.format(x[0], x[1]))


    def categorize_attacks_on_test(self):
        test_df = pd.read_csv(const.CAT_TEST_DATASET)
        testX, testY = format_data(test_df)

        train_df = pd.read_csv(const.CAT_TRAIN_DATASET)
        _, trainY = format_data(train_df)
        prob = self.category_model.predict(testX)
        self.calculate_classification_metrics(trainY, testY, prob, is_test_set=True)


    ## Metrics ##
    def calculate_regression_metrics(self, real_data, predict_data, is_test_set):
        if is_test_set:
            METRICS_PATH = const.MODEL_REGRESSION_TEST_METRICS_PATH
        else:
            METRICS_PATH = const.MODEL_REGRESSION_WINDOW_METRICS_PATH

        real_data = real_data[0:len(predict_data)]  # slice data, if predicted data are shorter (detected anomaly stops prediction)
        
        with open(METRICS_PATH.format(self.model_number), 'w') as f:
            for i in range(len(self.ts_handler.features)):
                mae_score = mae(real_data[:,i], predict_data[:,i])
                mape_score = mape(real_data[:,i], predict_data[:,i])
                mse_score = mse(real_data[:,i], predict_data[:,i])

                f.write(self.ts_handler.features[i] + '\n')
                f.write('MAE score:   {:.2f}\n'.format(mae_score))
                f.write('MAPE score:  {:.2f}\n'.format(mape_score))
                f.write('MSE score:   {:.2f}\n'.format(mse_score))
                f.write('RMSE score:  {:.2f}\n\n'.format(math.sqrt(mse_score)))


    def calculate_classification_metrics(self, trainY, y, prob, is_test_set):
        if is_test_set:
            METRICS_PATH = const.MODEL_CLASSIFICATION_METRICS_TEST_PATH
            ROC_CURVE_PATH = const.MODEL_METRICS_ROC_TEST_PATH
            CONF_MATRIX_PATH = const.MODEL_CONF_TEST_MATRIX_PATH
        else:
            METRICS_PATH = const.MODEL_CLASSIFICATION_METRICS_WINDOW_PATH
            ROC_CURVE_PATH = const.MODEL_METRICS_ROC_PATH
            CONF_MATRIX_PATH = const.MODEL_CONF_MATRIX_PATH

        y_pred = np.argmax(prob, axis=-1)
        all_classes = get_classes()
        classes_values = np.unique(y)
        present_classes = [all_classes[x] for x in classes_values]

        if (y == 0).all():
            print('Selected window contains only benign traffic !')
            return
        elif (y_pred == 0).all():
            print('Prediction contains only benign traffic !')
            return

        labels = np.unique(y_pred)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', labels=labels)
        recall = recall_score(y, y_pred, average='weighted', labels=labels)
        f1 = f1_score(y, y_pred, average='weighted', labels=labels)
        report = classification_report(y, y_pred, labels=labels)

        with open(METRICS_PATH.format(self.model_number), 'w') as f:
            f.write('Accuracy:   {:.2f}\n'.format(accuracy))
            f.write('Precision:  {:.2f}\n'.format(precision))
            f.write('Recall:     {:.2f}\n'.format(recall))
            f.write('F1 score:   {:.2f}\n\n'.format(f1))
            f.write(report)

        plt.figure(figsize=(8, 5))
        cm = confusion_matrix(y, y_pred)
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
        ax.xaxis.set_ticklabels(present_classes, rotation = 90)
        ax.yaxis.set_ticklabels(present_classes, rotation = 0)
        plt.xlabel('Predikované',fontsize=15)
        plt.ylabel('Skutočné',fontsize=15)
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PATH.format(self.model_number), dpi=400)

        roc_auc_multiclass(trainY, y, prob, ROC_CURVE_PATH.format(self.model_number))


    ## Plots and other functions ##
    def get_y_from_generator(self, gen):
        y = None
        for i in range(len(gen)):
            batch_y = gen[i][1]
            y = batch_y if y is None else np.append(y, batch_y)
        return y.reshape((-1, (self.ts_handler.n_features)))    

    
    def inverse_transform(self, predict, attack_data=False):
        if attack_data:
            tmp = self.ts_handler.attack_stand_scaler.inverse_transform(predict)
            return self.ts_handler.attack_minmax_scaler.inverse_transform(tmp)
        else:
            tmp = self.ts_handler.stand_scaler.inverse_transform(predict)
            return self.ts_handler.minmax_scaler.inverse_transform(tmp)


    def save_benign_ts_plots(self, train_data, test_data, prediction_data, time, show_full_data):
        if not show_full_data:                              # display only half of the train data
            train_len = len(train_data)
            train_data = train_data[int(train_len/3)*2:]    
            time = time[int(train_len/3)*2:train_len + len(train_data)]

        begin = len(train_data)                             # beginning is where train data ends
        end = begin + len(test_data)                        # end is where predicted data ends

        whole_data = np.append(train_data, test_data)             
        whole_data = whole_data.reshape(-1, self.ts_handler.n_features)

        pred_data = np.empty_like(whole_data)
        pred_data[:, :] = np.nan                            # first : stands for first and the second : for the second dimension
        pred_data[begin:end, :] = prediction_data           # insert predicted values

        self.save_plots(whole_data, pred_data, time, is_attack=False)


    def save_plots(self, train_data, prediction_data, time, is_attack):
        fig = const.MODEL_PREDICTIONS_ATTACK_PATH if is_attack else const.MODEL_PREDICTIONS_BENIGN_PATH
    
        for i in range(self.ts_handler.n_features): 
            train_feature = [item[i] for item in train_data]
            predict_feature = [item[i] for item in prediction_data]

            ax=plt.gca()
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


def load_best_model(model_number, model_name, model_type):
    if model_type == 'an':
        dir = const.SAVE_ANOMALY_MODEL_PATH.format(model_number, model_name.lower())
    elif model_type == 'cat':
        dir = const.SAVE_CAT_MODEL_PATH.format(model_number, model_name.lower())

    if os.path.exists(dir):
        sub_dirs = os.listdir(dir)
        sub_dirs.sort()
        return load_model(dir + sub_dirs[0])
    return None


def roc_auc_multiclass(y_train, y_test, y_score, path):
    sns.set_style('darkgrid')        
    deep_colors = sns.color_palette('deep')
    classes = get_classes()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 9

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, color = deep_colors[i], label='{0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falošne pozitívne')
    plt.ylabel('Správne pozitívne')
    plt.legend(loc="lower right")
    plt.savefig(path)