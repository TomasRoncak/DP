import sys
import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const
import config as conf

from generate_time_series import generate_time_series
from os import path, makedirs
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam, Adagrad
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D

def get_optimizer(learning_rate, optimizer, momentum = 0):
    switcher = {
        'sgd': SGD(learning_rate=learning_rate, name="SGD"),
        'sgd-momentum': SGD(learning_rate=learning_rate, momentum=momentum, name="SGD-Momentum"),
        'rms-prop': RMSprop(learning_rate=learning_rate, name="RMSprop"),
        'adam': Adam(learning_rate=learning_rate, name="Adam"),
        'adagrad': Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")
    }
    return switcher.get(optimizer)

def train(
    model_name,
    window_size,
    n_steps,
    learning_rate,
    optimizer,
    patience,
    epochs
):

    train_ts_generator, n_features, _ = generate_time_series(window_size, n_steps)

    model = None
    if model_name == 'CNN':
        model = Sequential()
        model.add(Conv2D(filters=64, padding='same', kernel_size=2, activation='relu', input_shape=(n_steps, n_features, 1)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=96, padding='same', kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_features))

    optimizer_fn = get_optimizer(learning_rate=learning_rate, optimizer=optimizer)
    model.compile(optimizer=optimizer_fn, loss='mse')

    run = wandb.init(project="dp", entity="tomasroncak")

    earlystop_callback = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    wandb_callback = wandb.keras.WandbCallback()

    model.fit(
        train_ts_generator, 
        epochs=epochs, 
        verbose=2, 
        callbacks=[earlystop_callback, wandb_callback]
    )

    i = 1
    while path.exists(const.SAVE_MODEL_PATH.format(i)):
        i += 1
    makedirs(const.MODEL_PATH.format(i))
    makedirs(const.MODEL_PREDICTIONS_PATH.format(i))
    model.save(const.SAVE_MODEL_PATH.format(i))
    run.finish()