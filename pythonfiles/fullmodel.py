from keras.layers import Input, Dense, LSTM, Convolution1D, AveragePooling1D, UpSampling1D, TimeDistributed
from keras.models import Model, Sequential
import openWav
import numpy as np
import pickle

input_length = 64 
x_train, y_train = openWav.loadDrums2(input_length)
input_sample = Input(shape=(input_length, 1))

x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 16, border_mode='same', activation="tanh")(x)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(1, 8, border_mode='same', activation="tanh")(x)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)

# This layer converts frequency space to hidden space
x = TimeDistributed(Dense(128))(x)
x = LSTM(128, return_sequences=True, stateful=False)(x)
# This layer converts hidden space back to frequency space
x = TimeDistributed(Dense(128))(x)

x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
output = Convolution1D(1, 32, border_mode='same', activation="tanh")(x)
    
    
generator = Model(input_sample, output)
generator.compile(optimizer='adadelta', loss='mean_squared_error')
weights_filename = 'fullmodel_weights.dat'
train = False
if train:
    generator.fit(x_train, y_train, nb_epoch=50,batch_size=10,shuffle=True,validation_data=(x_test, y_test))
    generator.save_weights(weights_filename, True)
#else:       
    #generator.load_weights(weights_filename)
generator.summary()
          

