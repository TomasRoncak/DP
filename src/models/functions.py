import datetime
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sklearn
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data')

import warnings

from ClassificationDataHandler import get_attack_classes

import constants as const

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


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


WARNING_TEXT_GREEN = bcolors.OKGREEN + bcolors.BOLD + 'Upozornenie' + bcolors.ENDC
WARNING_TEXT_RED = bcolors.FAIL + bcolors.BOLD + 'Upozornenie' + bcolors.ENDC
WARNING_TEXT_YELLOW = bcolors.WARNING + bcolors.BOLD + 'Upozornenie' + bcolors.ENDC


def get_optimizer(learning_rate, optimizer, momentum = 0):
    switcher = {
        'sgd': SGD(learning_rate=learning_rate, name='SGD'),
        'sgd-momentum': SGD(learning_rate=learning_rate, momentum=momentum, name='SGD-Momentum'),
        'rms-prop': RMSprop(learning_rate=learning_rate, name='RMSprop'),
        'adam': Adam(learning_rate=learning_rate, name='Adam'),
        'adagrad': Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
    }
    return switcher.get(optimizer)


def get_callbacks(model_number, model_name, patience, is_cat_multiclass=None):
    file_path = const.save_model[is_cat_multiclass].format(model_number, model_name) + 'loss-{loss:03f}.ckpt'
    cp_callback = ModelCheckpoint(filepath=file_path,
                                  monitor='loss',
                                  verbose=0,
                                  save_best_only=True,
                                  mode='min',
                                  save_freq='epoch',
                                  initial_value_threshold=None)
    early_stopping = EarlyStopping(monitor='loss', patience=patience)
    wandb_callback = wandb.keras.WandbCallback()

    return [cp_callback, early_stopping, wandb_callback]


def load_best_model(model_number, model_name, model_type, is_cat_multiclass=None):
    dir = const.save_model[is_cat_multiclass].format(model_number, model_name)

    if model_name == 'rf':
        return joblib.load(dir + const.RANDOM_FOREST_FILE)
    elif os.path.exists(dir):
        sub_dirs = os.listdir(dir)
        sub_dirs.sort()
        return load_model(dir + sub_dirs[0])
    else:
        model_type = {None: 'Anomalytický', True: 'Viactriedny', False: 'Binárny'}
        print('{0} model s číslom {1} nebol nájdený!'.format(model_type[is_cat_multiclass], model_number))
        quit()


def get_y_from_generator(n_features, gen):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        y = batch_y if y is None else np.append(y, batch_y)
    return y.reshape((-1, n_features))    


def plot_roc_auc(y_test, y_score, model_number, trainY, model_name, path):
    sns.set_style('darkgrid')        
    deep_colors = sns.color_palette('deep')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

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
        classes = get_attack_classes()
        if model_name == 'rf':
            y_score = label_binarizer.transform(y_score)
        for i in range(n_classes):
            if not (y_onehot_test[:, i] == 0).all():
                fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 5))
        for value in fpr:
            plt.plot(fpr[value], tpr[value], lw=2, color = deep_colors[value], 
                    label='{0} (AUC = {1:0.2f})'.format(classes[value], roc_auc[value]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falošne pozitívne')
    plt.ylabel('Správne pozitívne')
    plt.legend(loc='lower right')
    plt.savefig(path.format(model_number))
    plt.close()


def plot_confusion_matrix(y, y_pred, model_number, is_cat_multiclass, path):
    plt.figure(figsize=(8, 5))
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
    if is_cat_multiclass:
        all_classes = get_attack_classes()
        predicted_classes_names = [all_classes[x] for x in np.unique(y_pred)] 
        present_classes_names = [all_classes[x] for x in np.unique(y)] 
        classes_names = predicted_classes_names if len(predicted_classes_names) > len(present_classes_names) else present_classes_names
    else:
        classes_names = ['Benígne', 'Malígne']
        
    ax.set_xticks([x + 0.5 for x in range(len(classes_names))])
    ax.set_yticks([x + 0.5 for x in range(len(classes_names))])
    
    ax.xaxis.set_ticklabels(classes_names, rotation = 90)
    ax.yaxis.set_ticklabels(classes_names, rotation = 0)

    plt.xlabel('Predikované', fontsize=15)
    plt.ylabel('Skutočné', fontsize=15)
    plt.tight_layout()
    plt.savefig(path.format(model_number))
    plt.close()


def pretty_print_detected_attacks(prob, is_multiclass):
    if is_multiclass:
        classes = get_attack_classes()
        res_list = list(Counter(np.argmax(prob, axis=-1)).items())
        res_list.sort(key=lambda a: a[1], reverse=True)
        res = [(classes[x[0]], x[1]) for x in res_list]
        
        print(bcolors.BOLD + 'Detegované kategórie útokov:' + bcolors.ENDC)
        for x in res:
            if x[0] == 'Normal':
                continue
            print('{0} ({1}x)'.format(x[0], x[1]))
    else:
        y_pred = np.round(prob, 0)
        pred_attack_sum = (y_pred == 1).sum()
        if pred_attack_sum > 1:
            text = 'podozrivých tokov' if pred_attack_sum > 3 else 'podozrivé toky'
            print(WARNING_TEXT_RED + ': Znalostný model detegoval v časovom okne {0} {1}!'.format(pred_attack_sum, text))
        else:
            print(WARNING_TEXT_GREEN + ': Znalostný model nedetegoval v časovom okne žiadne podozrivé toky.')
         

def pretty_print_point_anomaly(err, threshold, curr_time, window_size, exceeding, patience_limit):
    window_end_time = (curr_time + datetime.timedelta(seconds=window_size)) \
                      .strftime(const.PRETTY_SHORT_TIME_FORMAT)
    curr_time = curr_time.strftime(const.PRETTY_TIME_FORMAT)

    print(WARNING_TEXT_YELLOW + ': Bola detegovaná bodová anomália v časovom okne {0}-{1}' \
            .format(curr_time, window_end_time) + bcolors.ENDC, end=' - ')
    print('chyba {0:.2f} > prah {1:.2f} (trpezlivosť={2}/{3})' \
            .format(err, threshold, exceeding, patience_limit))


def format_and_print_collective_anomaly(start_time, stop_time):
    print(WARNING_TEXT_RED + ': Kolektívna anomália detegovaná v okne {0}-{1}!' \
        .format(start_time.strftime(const.PRETTY_TIME_FORMAT), stop_time.strftime(const.PRETTY_SHORT_TIME_FORMAT))
    )
    return (start_time, stop_time)


def pretty_print_window_ok(curr_time, err):
    print('Okno {0} - chyba {1:.2f} '.format(curr_time, err) + bcolors.OKGREEN + 'OK' + bcolors.ENDC)


def save_rf_model(model, is_cat_multiclass, model_number, model_name):
    path = const.save_model[is_cat_multiclass].format(model_number, model_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path + const.RANDOM_FOREST_FILE)


def create_radar_plot(features, model_number, on_test_set, pic_format):
    var = 'Test' if on_test_set else 'Window'
    features.append(features[0])
    fig = go.Figure()

    for model_name in os.listdir(const.WHOLE_ANOMALY_MODEL_PATH.format(model_number)):
        if model_name.startswith('radar') or model_name == '.DS_Store':
            continue
        try:
            metrics = pd.read_csv(const.anomaly_metrics[on_test_set].format(model_number, model_name) + const.REPORT_FILE)
        except:
            print('{0} report modelu {1} nebol nájdený! Radar chart nemohol byť vytvorený.'.format(var, model_name))
            return

        metrics = metrics['mape'].to_list()
        metrics.append(metrics[0])

        fig.add_trace(go.Scatterpolar(
            r=metrics,
            theta=features,
            name=model_name.upper()
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title_text='MAPE {0} skóre'.format(var),
            title_x=0.5
        )

        fig.write_image(
            const.MODEL_REGRESSION_RADAR_CHART_PATH.format(model_number, var + '.' + pic_format), 
            format=pic_format
        )
    print('{0} radar chart bol úspešne vytvorený.'.format(var))