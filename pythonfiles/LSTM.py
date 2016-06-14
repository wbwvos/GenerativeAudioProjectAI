from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from keras.layers.wrappers import TimeDistributed
import numpy as np
import openWav

#x_train, y_train, x_test, y_test, sr = openWav.lstmData()
#model = train(x_train, y_train, x_test, y_test)

def train(x_train, y_train, x_test, y_test):
    # expected input data shape: (batch_size, timesteps, data_dim)
    timesteps = 10
    data_dim = len(x_train[0])
    batchsize = 10    
    num_hidden_dimensions = data_dim/2
    num_frequency_dimensions = data_dim
    print(num_hidden_dimensions)
    print(num_frequency_dimensions)
    
    #model = Sequential()
    #model.add(LSTM(data_dim, return_sequences=False, batch_input_shape=(10, data_dim, 1)))  
    #model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
    
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(LSTM(num_hidden_dimensions, return_sequences=True, stateful=False))

    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.summary()
    exit
    #print("fitting")
    #model.fit(x_train, y_train, batch_size=10, nb_epoch=5, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, batch_size=10, nb_epoch=5, verbose=1, validation_split=0.0)
    model.save_weights("weights_RNN.dat", True)
    return model
    
def predict():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(1024, return_sequences=False, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.load_weights("weights_RNN.dat")
    return model

x_train, y_train, x_test, y_test, sr = openWav.lstmData()
model = train(x_train, y_train, x_test, y_test)    
