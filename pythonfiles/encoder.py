from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import openWav 
import pickle

def encoder(x_train, x_test, train, encode, ):
    # this is the size of our encoded representations
    encoding_dim = 1024  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    # this is our input placeholder
    input = Input(shape=(sr,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(sr, activation='tanh')(encoded)
    
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
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print x_train.shape
    print x_test.shape
    
    if train:
        print("fitting")
        autoencoder.fit(x_train, x_train,nb_epoch=50, batch_size=10, shuffle=True, validation_data=(x_test, x_test))
        Model.save_weights("weights_1.dat", True)
        
    Model.load_weights("weights_1.dat")
                    
    # encode and decode some digits
    # note that we take them from the *test* set
    if encode:
        encoded_imgs = encoder.predict(x_test)
        return encoded_imgs
    else:
        decoded_imgs = decoder.predict(encoded_imgs)
        return decoded_imgs
