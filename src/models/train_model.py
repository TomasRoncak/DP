import sys

import pandas as pd

import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from preprocess_data import format_data
from sklearn.model_selection import train_test_split

import constants as const


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
    checkpoint_path = 'models/models_' + str(model_number) + '/' + model_type + 'savings/' + model_arch + '/model_loss-{loss:03f}.ckpt'
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


"""
performs training on a specified neural network and saves trained model 

:param ts_handler: time series object containing data
:param model_name: string specifying type of neural network (cnn, ltsm, ...)
:param n_steps: integer specifying number of previous steps to be used for future prediction
:param learning_rate: integer specifying the speed of learning (speed of gradient descent)
:param optimizer: string specifying type of optimizer
:param patience: integer specifying dropout patience
:param epochs: integer specifying number of epochs to be trained
:param dropout: integer specifying the probability of neurons dropout
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
    blocks,
    model_number, 
    activation,
    momentum
):
    Path(const.WHOLE_ANOMALY_MODEL_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)

    n_features = ts_handler.n_features

    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv1D(filters=64, padding='same', kernel_size=2, activation=activation, input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout)),
        model.add(Flatten())
        model.add(Dense(25, activation=activation))
    elif model_name == 'LSTM':
        model.add(LSTM(16, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(LSTM(8))
    elif model_name == 'GRU':
        model.add(GRU(blocks, input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    
    optimizer_fn = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
    model.compile(optimizer=optimizer_fn, loss='mse')

    run = wandb.init(project="dp_an", entity="tomasroncak")

    model.fit(
        ts_handler.benign_train_generator,
        epochs=epochs,
        verbose=0,
        callbacks=[get_callbacks(model_number, model_name.lower(), const.ANOMALY_MODEL_PATH, patience)]
    )

    #model.save(const.SAVE_ANOMALY_MODEL_PATH.format(model_number, model_name) + 'model.h5')
    run.finish()


def train_categorical(
    model_name,
    learning_rate,
    optimizer,
    patience,
    epochs,
    batch_size,
    dropout,
    model_number,
    activation,
    momentum
):
    Path(const.WHOLE_CLASSIFICATION_MODEL_PATH.format(model_number)).mkdir(parents=True, exist_ok=True)   

    df = pd.read_csv(const.CAT_TRAIN_DATASET)
    trainX, trainY = format_data(df)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    num_categories = df.iloc[:,-1].nunique()

    model = Sequential()
    if model_name == 'MLP':
        model.add(Dense(1024, activation=activation, input_dim=trainX.shape[1]))
        model.add(Dropout(dropout))
        model.add(Dense(768, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(num_categories, activation='softmax'))

    optimizer = get_optimizer(learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)

    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    run = wandb.init(project="dp_cat", entity="tomasroncak")

    model.fit(
            trainX,
            trainY,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(valX, valY),
            callbacks=[get_callbacks(model_number, model_name.lower(), const.CLASSIFICATION_MODEL_PATH, patience)],
            verbose=1
    )

    #model.save(const.SAVE_CAT_MODEL_PATH.format(model_number, model_name) + 'model.h5')
    run.finish()


def run_sweep(
    ts_handler,
    model_name,
    n_steps,
    patience,
    dropout,
    blocks,
    model_number,
    model_type,
):
    sweep_config_random = {
        'method': 'random',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.3
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.3
            },
            'epochs': {
                'values': [100]
            },
            'optimizer': {
                'values': ['sgd', 'sgd-momentum', 'rms-prop', 'adam', 'adagrad']
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.99
            },
            'activation': {
                'values': ['relu', 'tanh', 'selu']
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config_random, project=model_type + '_' + model_name)

    wandb.agent(
        sweep_id, 
        function=lambda: wandb_train(
                                ts_handler,
                                model_name,
                                n_steps,
                                patience,
                                dropout,
                                blocks,
                                model_number,
                                model_type
                                ), 
        count=40
    )

def wandb_train(
    ts_handler,
    model_name,
    n_steps,
    patience,
    dropout,
    blocks,
    model_number,
    model_type
):
    run = wandb.init(project="dp" + model_type, entity="tomasroncak")
    if model_type == 'anomaly':
        train_anomaly(
                    ts_handler,
                    model_name,
                    n_steps,
                    wandb.config.learning_rate,
                    wandb.config.optimizer,
                    patience,
                    wandb.config.epochs,
                    wandb.config.dropout,
                    blocks,
                    model_number,
                    wandb.config.activation,
                    wandb.config.momentum
                    )
    elif model_type == 'categorical':
        train_categorical(
                    model_name,
                    wandb.config.learning_rate,
                    wandb.config.optimizer,
                    patience,
                    wandb.config.epochs,
                    wandb.config.batch_size,
                    wandb.config.dropout,
                    model_number,
                    wandb.config.activation,
                    wandb.config.momentum
                    )
    run.finish()