import sys
import wandb

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/data/')

from generate_time_series import generate_time_series
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D
from keras.callbacks import EarlyStopping

window_size = 180.0
n_steps = 2
train_ts_generator, test_ts_generator, n_features = generate_time_series(window_size, n_steps)

model = Sequential()
model.add(Conv2D(filters=64, padding='same', kernel_size=2, activation='relu', input_shape=(n_steps, n_features, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=96, padding='same', kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')

run = wandb.init(project="dp", entity="tomasroncak")

earlystops = EarlyStopping(monitor='loss', patience=10, verbose=1)
wandb_callback = wandb.keras.WandbCallback()

model.fit(
    train_ts_generator, 
    epochs=200, 
    verbose=2, 
    callbacks=[earlystops, wandb_callback]
)

run.finish()