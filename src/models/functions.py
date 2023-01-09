import datetime
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def get_callbacks(model_number, model_arch, model_type, patience):
    checkpoint_path = 'models/models_' + str(model_number) + '/' + model_type + \
                      'savings/' + 'model_loss-{loss:03f}.ckpt'
    smallest_val_Loss = None

    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  period=1,
                                  initial_value_threshold=smallest_val_Loss)
    early_stopping = EarlyStopping(monitor='loss', patience=patience)
    wandb_callback = wandb.keras.WandbCallback()

    return [cp_callback, early_stopping, wandb_callback]


def load_best_model(model_number, model_name, model_type):
    if model_type == 'an':
        dir = const.SAVE_ANOMALY_MODEL_PATH.format(model_number, model_name)
    elif model_type == 'cat':
        dir = const.SAVE_CAT_MODEL_PATH.format(model_number, model_name)

    if os.path.exists(dir):
        sub_dirs = os.listdir(dir)
        sub_dirs.sort()
        return load_model(dir + sub_dirs[0])
    return None


def get_y_from_generator(n_features, gen):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        y = batch_y if y is None else np.append(y, batch_y)
    return y.reshape((-1, n_features))    


def plot_roc_auc_multiclass(y_test, y_score, model_number, is_test_set):
    PATH = const.MODEL_METRICS_ROC_TEST_PATH if is_test_set else const.MODEL_METRICS_ROC_PATH 
    sns.set_style('darkgrid')        
    deep_colors = sns.color_palette('deep')
    classes = get_filtered_classes()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    train_df = pd.read_csv(const.CAT_TRAIN_DATASET_PATH)
    _, trainY = format_data(train_df)

    label_binarizer = LabelBinarizer().fit(trainY)
    y_onehot_test = label_binarizer.transform(y_test)
    n_classes = y_onehot_test.shape[1]

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
    plt.savefig(PATH.format(model_number))


def plot_confusion_matrix(y, y_pred, model_number, is_test_set, present_classes):
    PATH = const.MODEL_CONF_TEST_MATRIX_PATH if is_test_set else const.MODEL_CONF_MATRIX_PATH
    plt.figure(figsize=(8, 5))
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
    ax.xaxis.set_ticklabels(present_classes, rotation = 90)
    ax.yaxis.set_ticklabels(present_classes, rotation = 0)
    plt.xlabel('Predikované', fontsize=15)
    plt.ylabel('Skutočné', fontsize=15)
    plt.tight_layout()
    plt.savefig(PATH.format(model_number), dpi=400)


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