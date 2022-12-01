import sys
import wandb
import pandas as pd

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')
sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const
from preprocess_data import format_data

from os import path, makedirs
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam, Adagrad
from sklearn.model_selection import train_test_split
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Conv1D, Conv2D, LSTM, GRU, Flatten, MaxPooling1D, MaxPooling2D, Dropout

def save_model(model, name, number, type):
    MODEL_FOLDER_PATH = const.MODEL_PATH.format(number)
    MODEL_PATH = const.SAVE_ANOMALY_MODEL_PATH if type == 'anomaly' else const.SAVE_CAT_MODEL_PATH

    if not path.exists(MODEL_FOLDER_PATH):
        makedirs(MODEL_FOLDER_PATH)
    model.save(MODEL_PATH.format(number, name.lower()))


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
    blocks,
    model_number
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

    run = wandb.init(project="dp_an", entity="tomasroncak")

    earlystop_callback = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    wandb_callback = wandb.keras.WandbCallback()

    model.fit(
        ts_handler.benign_train_generator,
        epochs=epochs,
        verbose=2,
        callbacks=[earlystop_callback, wandb_callback]
    )

    save_model(model, model_name, model_number, type='anomaly')
    run.finish()

def train_categorical(
    model_name,
    learning_rate,
    optimizer,
    patience,
    epochs,
    batch_size,
    dropout,
    model_number
):
    df = pd.read_csv(const.WHOLE_CAT_TRAIN_DATASET)
    test_df = pd.read_csv(const.WHOLE_CAT_TEST_DATASET)

    num_categories = df.iloc[:,-1].nunique()
    input_shape = df.shape[1] - 1   # -  1 = category

    trainX, trainY = format_data(df)
    testX, testY = format_data(test_df)

    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    model = Sequential()
    if model_name == 'MLP':
        model.add(Flatten(input_shape=(input_shape, 1)))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(num_categories, activation='softmax'))

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = get_optimizer(learning_rate=learning_rate, optimizer=optimizer)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    run = wandb.init(project="dp_cat", entity="tomasroncak")

    wandb_callback = wandb.keras.WandbCallback()
    earlystop_callback = EarlyStopping(monitor='loss', patience=patience, verbose=1)

    model.fit(
            trainX,
            trainY,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(valX, valY),
            callbacks=[earlystop_callback, wandb_callback],
            verbose=1
    ) 

    test_scores = model.evaluate(testX, testY)
    print("Test scores:", test_scores)

    save_model(model, model_name, model_number, type='categorical')
    run.finish()