from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 1024
timesteps = 20
nb_classes = 10
train = True

def train(encoded):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(1024, return_sequences=False,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
    
    print("fitting")
    model.fit(x_train, y_train, batch_size=64, nb_epoch=5, validation_data=(x_val, y_val))
    model.save_weights("weights_RNN.dat", True)

    
def predict(encoded):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(1024, return_sequences=False, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.load_weights("weights_RNN.dat")
    model.predict(encoded)
    