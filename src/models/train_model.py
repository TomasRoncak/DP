import sys
import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

from os import path, makedirs
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam, Adagrad
from keras.layers import Dense, Conv1D, Conv2D, LSTM, GRU, Flatten, MaxPooling1D, MaxPooling2D, Dropout

def save_model(model, model_name, model_type):
    PATH = const.SAVE_ANOMALY_MODEL_PATH if model_type == 'anomaly' else const.SAVE_CAT_MODEL_PATH
    i = 1
    while path.exists(const.MODEL_PATH.format(i)):
        i += 1
    makedirs(const.MODEL_PATH.format(i))
    model.save(PATH.format(i, model_name.lower()))


def get_optimizer(learning_rate, optimizer, momentum = 0):
    switcher = {
        'sgd': SGD(learning_rate=learning_rate, name="SGD"),
        'sgd-momentum': SGD(learning_rate=learning_rate, momentum=momentum, name="SGD-Momentum"),
        'rms-prop': RMSprop(learning_rate=learning_rate, name="RMSprop"),
        'adam': Adam(learning_rate=learning_rate, name="Adam"),
        'adagrad': Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")
    }
    return switcher.get(optimizer)


"""
performs training on a specified neural network and saves trained model 

:param ts_handler: time series object containing data
:param model_name: string specifying type of neural network (cnn, ltsm, ...)
:param n_steps: integer specifying number of previous steps to be used for future prediction
:param learning_rate: integer specifying the speed of learning (speed of gradient descent)
:param optimizer: string specifying type of optimizer
:param patience: integer specifying dropout patience
:param epochs: integer specifying number of epochs to be trained
:param dropout_rate: integer specifying the probability of neurons dropout
:param blocks: number of blocks to be used in sequential neural networks
"""
def train_anomaly(
    ts_handler,
    model_name,
    n_steps,
    learning_rate,
    optimizer,
    patience,
    epochs,
    dropout,
    blocks
):
    n_features = ts_handler.n_features

    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        #model.add(Conv2D(filters=128, padding='same', kernel_size=2, activation='relu'))
        #model.add(Dropout(dropout)),
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
    elif model_name == 'LSTM':
        model.add(LSTM(blocks, input_shape=(n_steps, n_features)))
    elif model_name == 'GRU':
        model.add(GRU(blocks, input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    
    optimizer_fn = get_optimizer(learning_rate=learning_rate, optimizer=optimizer)
    model.compile(optimizer=optimizer_fn, loss='mse')

    run = wandb.init(project="dp", entity="tomasroncak")

    earlystop_callback = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    wandb_callback = wandb.keras.WandbCallback()

    model.fit(
        ts_handler.benign_train_generator,
        epochs=epochs,
        verbose=2,
        callbacks=[earlystop_callback, wandb_callback]
    )

    save_model(model, model_name, model_type='anomaly')
    run.finish()

def train_categorical(
    model_name,
    learning_rate,
    optimizer,
    patience,
    epochs,
    dropout
):
    pass