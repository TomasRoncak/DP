import datetime
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data')

from preprocess_data import format_data, get_filtered_classes

import constants as const


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

def get_optimizer(learning_rate, optimizer, momentum = 0):
    switcher = {
        'sgd': SGD(learning_rate=learning_rate, name="SGD"),
        'sgd-momentum': SGD(learning_rate=learning_rate, momentum=momentum, name="SGD-Momentum"),
        'rms-prop': RMSprop(learning_rate=learning_rate, name="RMSprop"),
        'adam': Adam(learning_rate=learning_rate, name="Adam"),
        'adagrad': Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")
    }
    return switcher.get(optimizer)


def get_callbacks(model_number, model_arch, is_cat_multiclass, patience):
    if is_cat_multiclass is None:
        PATH = const.SAVE_ANOMALY_MODEL_PATH.format(str(model_number), model_arch)
    elif is_cat_multiclass:
        PATH = const.SAVE_CAT_MULTICLASS_MODEL_PATH.format(str(model_number), model_arch)
    else:
        PATH = const.SAVE_CAT_BINARY_MODEL_PATH.format(str(model_number), model_arch)

    cp_callback = ModelCheckpoint(filepath=PATH + 'loss-{loss:03f}.ckpt',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  save_freq=1,
                                  initial_value_threshold=None)
    early_stopping = EarlyStopping(monitor='loss', patience=patience)
    wandb_callback = wandb.keras.WandbCallback()

    return [cp_callback, early_stopping, wandb_callback]


def load_best_model(model_number, model_name, model_type, is_cat_multiclass=None):
    if model_type == 'an':
        dir = const.SAVE_ANOMALY_MODEL_PATH.format(model_number, model_name)
    elif model_type == 'cat':
        if is_cat_multiclass:
            dir = const.SAVE_CAT_MULTICLASS_MODEL_PATH.format(model_number, model_name)
        else:
            dir = const.SAVE_CAT_BINARY_MODEL_PATH.format(model_number, model_name)

    if os.path.exists(dir):
        sub_dirs = os.listdir(dir)
        sub_dirs.sort()
        return load_model(dir + sub_dirs[0])
    else:
        if model_type == 'an':
            model_type = 'Anomalytický'
        elif is_cat_multiclass:
            model_type = 'Viactriedny'
        else:
            model_type = 'Binárny'
        print('{0} model s číslom {1} nebol nájdený!'.format(model_type, model_number))
        quit()

def get_y_from_generator(n_features, gen):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        y = batch_y if y is None else np.append(y, batch_y)
    return y.reshape((-1, n_features))    


def plot_roc_auc(y_test, y_score, model_number, is_cat_multiclass, path):
    sns.set_style('darkgrid')        
    deep_colors = sns.color_palette('deep')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    train_df = pd.read_csv(const.CAT_TRAIN_DATASET_PATH)
    _, trainY = format_data(train_df, is_cat_multiclass)

    label_binarizer = LabelBinarizer().fit(trainY)
    y_onehot_test = label_binarizer.transform(y_test)
    n_classes = y_onehot_test.shape[1]

    if n_classes == 1:
        y_pred = np.round(y_score, 0)
        fpr, tpr, _ = roc_curve(y_onehot_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, lw=2, color = deep_colors[0], label='AUC = {0:0.2f}'.format(roc_auc))
    else:
        classes = get_filtered_classes()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 5))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, color = deep_colors[i], 
                label='{0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falošne pozitívne')
    plt.ylabel('Správne pozitívne')
    plt.legend(loc="lower right")
    plt.savefig(path.format(model_number))


def plot_confusion_matrix(y, y_pred, model_number, present_classes, path):
    plt.figure(figsize=(8, 5))
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
    ax.xaxis.set_ticklabels(present_classes, rotation = 90)
    ax.yaxis.set_ticklabels(present_classes, rotation = 0)
    plt.xlabel('Predikované', fontsize=15)
    plt.ylabel('Skutočné', fontsize=15)
    plt.tight_layout()
    plt.savefig(path.format(model_number), dpi=400)


def pretty_print_detected_attacks(prob):
    classes = get_filtered_classes()
    res_list = list(Counter(np.argmax(prob, axis=-1)).items())
    res_list.sort(key=lambda a: a[1], reverse=True)
    res = [(classes[x[0]], x[1]) for x in res_list]

    print(bcolors.FAIL + bcolors.BOLD + 'Upozornenie' + bcolors.ENDC + ': Časové okno obsahuje útoky na sieť!')
    print(bcolors.BOLD + 'Detegované kategórie útokov:' + bcolors.ENDC)
    
    for x in res:
        if x[0] == 'Normal':
            continue
        print('{0} ({1}x)'.format(x[0], x[1]))


def pretty_print_point_anomaly(err, threshold, curr_time, window_size, exceeding, patience_limit):
    curr_time = datetime.datetime.strptime(curr_time, const.TIME_FORMAT)
    window_end_time = (curr_time + datetime.timedelta(seconds=window_size)) \
                      .strftime(const.PRETTY_TIME_FORMAT)
    curr_time = curr_time.strftime(const.PRETTY_TIME_FORMAT)

    print(bcolors.WARNING + 'Bodová anomália detegovaná v časovom okne {0} - {1}' \
            .format(curr_time, window_end_time) + bcolors.ENDC, end='- ')
    print('chyba {0:.2f} prekročila prah {1:.2f} (trpezlivosť={2}/{3})' \
            .format(err, threshold, exceeding, patience_limit))


def pretty_print_collective_anomaly(start_time, stop_time):
    print(bcolors.FAIL + bcolors.BOLD + 'Upozornenie' + bcolors.ENDC + 
        ': Kolektívna anomália detegovaná v okne {0} až {1}!' \
        .format(start_time.strftime(const.PRETTY_TIME_FORMAT), stop_time.strftime(const.PRETTY_TIME_FORMAT))
    )


def format_date(time):
    return datetime.datetime.strptime('2015-' + time + ':00', const.FULL_TIME_FORMAT)


def pretty_print_window_ok(curr_time, err):
    print('Okno {0} - chyba {1:.2f} '.format(curr_time, err) + bcolors.OKGREEN + 'OK' + bcolors.ENDC)


def create_radar_plot(features, on_test_set, format):
        PATH = const.MODEL_REGRESSION_TEST_METRICS_PATH if on_test_set else const.MODEL_REGRESSION_WINDOW_METRICS_PATH
        var = 'test' if on_test_set else 'window'
        angles = np.linspace(0,2*np.pi,len(features), endpoint=False)
        angles = np.concatenate((angles,[angles[0]]))
        features.append(features[0])

        fig = go.Figure()
        model_number = 1
        while True:
            METRICS_PATH = PATH.format(model_number)
            try:
                metrics = pd.read_csv(METRICS_PATH)
            except:
                break

            path = os.path.dirname(const.WHOLE_ANOMALY_MODEL_PATH.format(model_number))
            arch_type = None
            for f_name in os.listdir(path):
                if f_name.startswith('savings_'):
                    arch_type = f_name.split('_')[1]
                    break
            
            metrics = metrics['mape'].to_list()
            metrics.append(metrics[0])
            fig.add_trace(go.Scatterpolar(
                r=metrics,
                theta=features,
                name=arch_type
            ))

            model_number += 1

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True
                )),
            showlegend=True,
            title_text='MAPE {0} skóre'.format(var),
            title_x=0.5
        )

        fig.write_image(const.MODEL_REGRESSION_RADAR_CHART_PATH.format(var + format))

        # fig = plt.figure(figsize=(6, 10))
        # ax = fig.add_subplot(polar=True)
        # while True:
        #     METRICS_PATH = PATH.format(model_number)
        #     try:
        #         metrics = pd.read_csv(METRICS_PATH)
        #     except:
        #         break

        #     path = os.path.dirname(const.WHOLE_ANOMALY_MODEL_PATH.format(model_number))
        #     arch_type = None
        #     for f_name in os.listdir(path):
        #         if f_name.startswith('savings_'):
        #             arch_type = f_name.split('_')[1]
        #             break
                
        #     metrics = metrics['mape'].to_list()
        #     metrics.append(metrics[0])

        #     ax.plot(angles, metrics, label=arch_type)
        #     model_number += 1
        
        # ax.set_thetagrids(angles * 180/np.pi, features)
        # plt.tight_layout()
        # plt.legend(bbox_to_anchor = (1.4, 0.6), loc='center right')
        # plt.title('MAPE skóre', fontsize=15)
        # plt.savefig(const.MODEL_REGRESSION_RADAR_CHART_PATH.format(var), bbox_inches='tight')
