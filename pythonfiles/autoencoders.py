from keras.layers import Input, Dense, Convolution1D, AveragePooling1D, UpSampling1D
from keras.models import Model
import numpy as np
import time
import os.path

def getConvAutoEncoderModel(input_length, x_train, x_test):
    x_train = np.resize(x_train, (x_train.shape[0], input_length, 1)).astype(np.float32)
    x_test = np.resize(x_test, (x_test.shape[0], input_length, 1)).astype(np.float32)
    
    input_sample = Input(shape=(input_length, 1))
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(32, 16, border_mode='same', activation="tanh")(x)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(1, 8, border_mode='same', activation="tanh")(x)
    encoded = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    #output = 128,1
    x = UpSampling1D(length=2)(encoded)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    decoded = Convolution1D(1, 32, border_mode='same', activation="tanh")(x)
    
    autoencoder = Model(input_sample, decoded)
    autoencoder.summary()
    encoder = Model(input=input_sample, output=encoded)
    encoder.summary()
    encoded_input = Input(shape=(128,))
    decoder_layers = autoencoder.layers[9:]
    decoder = Model(input=encoded_input, output=decoder_layers(encoded_input))
    decoder.summary()
    
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    print autoencoder.summary()
    weights_filename = 'weights_conv.dat'
    if os.path.isfile(weights_filename):
        print 'Loading the model...'
        autoencoder.load_weights(weights_filename)
    else:
        print 'Training the model...'
        trainStart = time.time()
        autoencoder.fit(x_train, x_train,
                    nb_epoch=500,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))
        trainEnd = time.time()
        print 'Trained the model in', trainEnd - trainStart, 'seconds'
        print 'Saving the model...'
        autoencoder.save_weights(weights_filename, True)
    return encoder, decoder
    
def getSimpleAutoEncoderModel(input_length, x_train, x_test, encoding_dim=1024):

    # this is our input placeholder
    input = Input(shape=(input_length,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_length, activation='tanh')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input, output=decoded)
    
    # this model maps an input to its encoded representation
    encoder = Model(input=input, output=encoded)
    
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    print autoencoder.summary()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    weights_filename = 'weights_simple.dat'
    if os.path.isfile(weights_filename):
        print 'Loading the model...'
        autoencoder.load_weights(weights_filename)
    else:
        print 'Training the model...'
        trainStart = time.time()
        autoencoder.fit(x_train, x_train,
                        nb_epoch=50, 
                        batch_size=10, 
                        shuffle=True, 
                        validation_data=(x_test, x_test))
        trainEnd = time.time()
        print 'Trained the model in', trainEnd - trainStart, 'seconds'
        print 'Saving the model...'
        autoencoder.save_weights(weights_filename, True)
    return encoder, decoder

def getEncoder(input_length):
    input_sample = Input(shape=(input_length, 1))
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(32, 16, border_mode='same', activation="tanh")(x)
    x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(1, 8, border_mode='same', activation="tanh")(x)
    encoded = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    encoder = Model(input_sample, encoded)
    encoder.summary()
    return encoder

def getDecoder(encoded_length):
    encoded = Input(shape=(encoded_length, 1))
    x = UpSampling1D(length=2)(encoded)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    decoded = Convolution1D(1, 32, border_mode='same', activation="tanh")(x)
    decoder = Model(encoded, decoded)
    decoder.summary()
    return decoder
