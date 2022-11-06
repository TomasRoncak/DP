from data.create_windowed_data import preprocess_dataset
from data.generate_time_series import generate_time_series, create_extracted_dataset
from models.train_model import train
from models.predict_model import predict
from features.build_features import perform_build_features

import config as conf

"""
clean and create time series dataset out of flow-based network capture dataset

:param window_size: integer specifying the length of sliding window to be used
:param include_attacks: boolean specifying if dataset should contain network attacks
:param save_plots: boolean specifying if protocol feature plots should saved
"""

#preprocess_dataset(conf.window_size, include_attacks=True, save_plots=True)
#preprocess_dataset(conf.window_size, include_attacks=False, save_plots=True)

"""
perform feature selection on created time series dataset (nonunique, randomness, colinearity, adfueller, peak cutoff)

:param window_size: integer specifying the length of sliding window to be used
:param print_steps: boolean specifying if selection steps should be described in more detail
"""

#perform_build_features(conf.window_size, print_steps=True)

"""
creates dataset suitable for training according to extracted features (on data without attacks)

:param window_size: number specifying which dataset to use according to window size
"""
#create_extracted_dataset(conf.window_size)

"""
performs training on a specified neural network and saves trained model 

:param model_name: string specifying type of neural network (cnn, ltsm, ...)
:param window_size: integer specifying the type of dataset to be used according to size of window
:param n_steps: integer specifying number of previous steps to be used for future prediction
:param learning_rate: integer specifying the speed of learning (speed of gradient descent)
:param optimizer: string specifying type of optimizer
:param patience: integer specifying dropout patience
:param epochs: integer specifying number of epochs to be trained
:param dropout_rate: integer specifying the probability of neurons dropout
:param blocks: number of blocks to be used in sequential neural networks
:param stl_decomposition: boolean specifying if STL decomposition should be applied on dataset
"""

train(
    conf.model_name,
    conf.window_size, 
    conf.n_steps, 
    conf.learning_rate, 
    conf.optimizer, 
    conf.patience, 
    conf.epochs,
    conf.dropout_rate,
    conf.blocks,
    conf.stl_decomposition
)

"""
loads saved model, performs prediction on test data calculates metrics and optionaly saves prediction plots

:param model_name: string specifying type of neural network (cnn, ltsm, ...)
:param window_size: integer specifying the type of dataset to be used according to size of window
:param n_steps: integer specifying number of previous steps to be used for future prediction
:param stl_decomposition: boolean specifying if STL decomposition should be applied on dataset
:param model_number: integer specifying the number of model on which to predict
:param save_plots: boolean specifying if prediction plots should be saved
"""

predict(
    conf.model_name,
    conf.window_size,
    conf.n_steps,
    conf.stl_decomposition,
    model_number=4,
    save_plots=True
)