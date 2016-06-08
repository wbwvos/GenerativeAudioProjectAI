from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import openWav 

x_train, x_test, sr = openWav.loadData()

# this is the size of our encoded representations
encoding_dim = 128  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

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

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))
                
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
#import matplotlib.pyplot as plt

#n = 10  # how many digits we will display
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i)
#    plt.imshow(x_test[i].reshape(210, 210))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    # display reconstruction
#    ax = plt.subplot(2, n, i + n)
#    plt.imshow(decoded_imgs[i].reshape(210, 210))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
