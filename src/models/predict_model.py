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
    def __init__(self, ts_handler, an_model_name, cat_model_name, patience_limit, model_number):
        self.anomaly_model = load_best_model(model_number, an_model_name, model_type='an')
        self.category_model = load_best_model(model_number, cat_model_name, model_type='cat')
        self.ts_handler = ts_handler
        self.model_number = model_number
        self.patience_limit = patience_limit
        self.exceeding = 0
        self.normal = 0
        self.point_anomaly_detected = False
        self.anomaly_detection_time = ()

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

        self.save_benign_ts_plots(train_inversed, test_inversed, predict_inversed, self.ts_handler.time, show_full_data=False)


    def predict_attacks_ts(self):
        attack_real = self.get_y_from_generator(self.ts_handler.attack_data_generator)
        time = self.ts_handler.time
        attack_predict = []
        data_generator = self.ts_handler.attack_data_generator

        for i in range(len(data_generator)):
            x, _ = data_generator[i]
            curr_time = time[i] 

            pred = self.anomaly_model.predict(x)
            attack_predict.append(pred[0])

            if self.detect_anomaly_ts(attack_real, pred, i):
                begin_time = datetime.datetime.strptime(curr_time, const.TIME_FORMAT)
                end_time = begin_time + datetime.timedelta(minutes=3)
                self.anomaly_detection_time = (begin_time, end_time)

                print('First anomaly occured in window {0} - {1}.' \
                        .format(begin_time.strftime(const.PRETTY_TIME_FORMAT), end_time.strftime(const.PRETTY_TIME_FORMAT)))
                break

        attack_predict = np.array(attack_predict)
        
        self.calculate_regression_metrics(attack_real, attack_predict)
        self.save_plots(
                self.inverse_transform(attack_real, True), 
                self.inverse_transform(attack_predict, True), 
                time, 
                attack=True
        )
        
        
    def detect_anomaly_ts(self, real, pred, i):
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


    ## Classification model prediction ##
    def categorize_attacks_on_window(self):
        if not self.anomaly_detection_time:
            print('No window found to classify on!')
            return

        df = pd.read_csv(const.WHOLE_DATASET, parse_dates=[const.TIME])
        data = df[(df[const.TIME] >= self.anomaly_detection_time[0]) & (df[const.TIME] <= self.anomaly_detection_time[1])]
        classes = get_classes()

        x, y = format_data(data)
        x = np.asarray(x).astype('float32')

        prob = self.category_model.predict(x)
        y_pred = np.argmax(prob, axis=-1)

        self.calculate_classification_metrics(y, y, y_pred, x, is_test_set=False)

        res_list = list(Counter(y_pred).items())
        res_list.sort(key=lambda a: a[1], reverse=True)
        res = [(classes[x[0]], x[1]) for x in res_list]

        print("Detected attacks:")
        for x in res:
            if x[0] == 'Normal':
                continue
            print(x)


    def categorize_attacks_on_test(self):
        test_df = pd.read_csv(const.WHOLE_CAT_TEST_DATASET)
        testX, testY = format_data(test_df)

        train_df = pd.read_csv(const.WHOLE_CAT_TRAIN_DATASET)
        _, trainY = format_data(train_df)

        prob = self.category_model.predict(testX)
        self.calculate_classification_metrics(trainY, testY, prob, is_test_set=True)


    ## Metrics ##
    def calculate_regression_metrics(self, real_data, predict_data):
        real_data = real_data[0:len(predict_data)]  # slice data, if predicted data are shorter (detected anomaly stops prediction)
        with open(const.MODEL_REGRESSION_METRICS_PATH.format(self.model_number), 'w') as f:
            for i in range(len(self.ts_handler.features)):
                mape_score = mape(real_data[:,i], predict_data[:,i])
                mse_score = mse(real_data[:,i], predict_data[:,i])
                mae_score = mae(real_data[:,i], predict_data[:,i])

                f.write(self.ts_handler.features[i] + '\n')
                f.write('MAPE score:  {:.2f}\n'.format(mape_score))
                f.write('MAE score:   {:.2f}\n'.format(mae_score))
                f.write('MSE score:   {:.2f}\n'.format(mse_score))
                f.write('RMSE score:  {:.2f}\n\n'.format(math.sqrt(mse_score)))


    def calculate_classification_metrics(self, y_train, y, prob, is_test_set):
        if is_test_set:
            METRICS_PATH = const.MODEL_CLASSIFICATION_METRICS_TEST_PATH
            ROC_CURVE_PATH = const.MODEL_METRICS_ROC_TEST_PATH
            CONF_MATRIX_PATH = const.MODEL_CONF_TEST_MATRIX_PATH
        else:
            METRICS_PATH = const.MODEL_CLASSIFICATION_METRICS_WINDOW_PATH
            ROC_CURVE_PATH = const.MODEL_METRICS_ROC_PATH
            CONF_MATRIX_PATH = const.MODEL_CONF_MATRIX_PATH

        y_pred = np.argmax(prob, axis=-1)
        classes = list(get_classes().values())

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        report = classification_report(y, y_pred, target_names=classes)

        with open(METRICS_PATH.format(self.model_number), 'w') as f:
            f.write('Accuracy:   {:.2f}\n'.format(accuracy))
            f.write('Precision:  {:.2f}\n'.format(precision))
            f.write('Recall:     {:.2f}\n'.format(recall))
            f.write('F1 score:   {:.2f}\n\n'.format(f1))
            f.write(report)

        plt.figure(figsize=(8, 5))
        cm = confusion_matrix(y, y_pred)
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
        ax.xaxis.set_ticklabels(classes, rotation = 90)
        ax.yaxis.set_ticklabels(classes, rotation = 0)
        plt.xlabel('Predicted',fontsize=15)
        plt.ylabel('Actual',fontsize=15)
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PATH.format(self.model_number), dpi=400)

        roc_auc_multiclass(y_train, y, prob, ROC_CURVE_PATH.format(self.model_number))


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
            train_data = train_data[int(train_len/2):]    
            time = time[int(train_len/2):train_len + len(train_data)]

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

            plt.rcParams["figure.figsize"] = (35, 15)
            plt.plot(time.iloc[:len(train_feature)], train_feature, label ='Realita', color="#017b92", linewidth=3)
            plt.plot(predict_feature, label ='Predikcia', color="#f97306", linewidth=3) 

            plt.xticks(time[::30], rotation='vertical', fontsize=30)
            plt.yticks(fontsize=30)
            plt.tight_layout()
            plt.title(self.ts_handler.features[i], fontsize=40)
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(path)