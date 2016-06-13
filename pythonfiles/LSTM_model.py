from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def train(x_train, y_train, x_test, y_test):
    # expected input data shape: (batch_size, timesteps, data_dim)
    timesteps = 10
    data_dim = len(x_train[0])
    
    
    model = Sequential()
    model.add(LSTM(data_dim, return_sequences=False,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
    
    print("fitting")
    model.fit(x_train, y_train, batch_size=64, nb_epoch=5, validation_data=(x_test, y_test))
    model.save_weights("weights_RNN.dat", True)
    
    return model
    
def predict():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(1024, return_sequences=False, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.load_weights("weights_RNN.dat")
    return model
    