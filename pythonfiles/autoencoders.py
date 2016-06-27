from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.noise import GaussianDropout
from keras.models import Model
from keras.utils.visualize_util import plot
import numpy as np
import time
import os.path
import openWav

def getConvAutoEncoderModel(input_length, x_train = None, x_test=None):
    #x_train = np.resize(x_train, (x_train.shape[0], input_length, 1)).astype(np.float32)
    #x_test = np.resize(x_test, (x_test.shape[0], input_length, 1)).astype(np.float32)   
    input_sample = Input(shape=(input_length, 1))
    x = Convolution1D(32, 11, border_mode='same', activation="tanh")(input_sample)
    #x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    #x = GaussianDropout(0.1)(x)
    x = Convolution1D(32, 5, border_mode='same', activation="tanh")(x)
    x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    encoded = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    
    x = UpSampling1D(length=2)(encoded)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    #x = GaussianDropout(0.25)(x)
    x = Convolution1D(32, 5, border_mode='same', activation="tanh")(x)
    #x = UpSampling1D(length=2)(x)
    #x = GaussianDropout(0.1)(x)
    decoded = Convolution1D(32, 11, border_mode='same', activation="tanh")(x)
    
    autoencoder = Model(input_sample, decoded)
    #autoencoder.summary()
    #encoder = Model(input=input_sample, output=encoded)
    #encoder.summary()
    #encoded_input = Input(shape=(128,))
    #decoder_layers = autoencoder.layers[9:]
    #decoder = Model(input=encoded_input, output=decoder_layers(encoded_input))
    #decoder.summary()
    
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.summary()
    #plot(autoencoder, to_file='conv_autoencoder.png', show_shapes=True)
    weights_filename = 'weights_conv_maxpooling_exp1.dat'
    if os.path.isfile(weights_filename):
        print 'Loading the model...'
        autoencoder.load_weights(weights_filename)
    else:
        print 'Training the model...'
        trainStart = time.time()
        autoencoder.fit(x_train, x_train,
                    nb_epoch=100,
                    batch_size=100,
                    shuffle=True) #,
                    #validation_data=(x_test, x_test))
        trainEnd = time.time()
        print 'Trained the model in', trainEnd - trainStart, 'seconds'
        print 'Saving the model...'
        autoencoder.save_weights(weights_filename, True)
    return autoencoder
   
def trainConvAutoEncoder():
    from os import listdir
    import librosa
    rootdir = '../data/drums/'

    y, sr = librosa.load(rootdir+listdir(rootdir)[0], sr=None)
    print(sr)
    data = y[:(len(y)/512)*512]
    print(data.shape)
    data = np.reshape(data, (data.shape[0]/512, 512, 1))
    #data = y[:-512]
    #print(data.shape)
    #for i in range(0, 512):
    #    print(i)
    #    data = np.vstack([data, y[i*4 :-(512-i)]])
    #print(data.shape)
    getConvAutoEncoderModel(512, data, data)

trainConvAutoEncoder()

 
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

def getEncoderModel(input_length, ae_weights=0):
    input_sample = Input(shape=(input_length, 1))
    x = Convolution1D(32, 11, border_mode='same', activation="tanh")(input_sample)
    #x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(32, 5, border_mode='same', activation="tanh")(x)
    x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    encoded = MaxPooling1D(pool_length=2, stride=None, border_mode="valid")(x)
    encoder = Model(input_sample, encoded)
    #plot(encoder, to_file='conv_encoder.png', show_shapes=True)
    return encoder

def getDecoderModel(encoded_length, ae_weights=0):
    encoded = Input(shape=(encoded_length, 1))
    x = UpSampling1D(length=2)(encoded)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(64, 3, border_mode='same', activation="tanh")(x)
    x = UpSampling1D(length=2)(x)
    x = Convolution1D(32, 5, border_mode='same', activation="tanh")(x)
    #x = UpSampling1D(length=2)(x)
    decoded = Convolution1D(32, 11, border_mode='same', activation="tanh")(x)
    decoder = Model(encoded, decoded)
    #plot(decoder, to_file='conv_decoder.png', show_shapes=True)
    return decoder

def getSplitConvAutoEncoder():
    input_dim = 512
    latent_dim = 32
    ae = getConvAutoEncoderModel(input_dim)
    encoder = getEncoderModel(input_dim)
    decoder = getDecoderModel(latent_dim)
    e_nb_layers = len(encoder.layers)
    d_nb_layers = len(decoder.layers)
    ae_nb_layers = len(ae.layers)
    #print('Encoder:......', e_nb_layers)
    #print('Decoder:......', d_nb_layers)
    #print('Autoencoder:..', ae_nb_layers)
    for i in range(e_nb_layers):
        encoder.layers[i].set_weights(ae.layers[i].get_weights())
    
    for i in range(1, d_nb_layers): #we start at 1 so skip the first layer, this is the added input layer for the decoder
        ae_index = i + e_nb_layers - 1
        #print 'd_index', i, 'ae_index', ae_index
        decoder.layers[i].set_weights(ae.layers[ae_index].get_weights())    
    
    encoder.compile(optimizer='adadelta', loss='mean_squared_error')
    decoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return encoder, decoder

#encoder, decoder = getSplitConvAutoEncoder()



























