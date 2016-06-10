from keras.layers import Input, Dense, Convolution1D, AveragePooling1D, UpSampling1D
from keras.models import Model
import openWav
import numpy as np
import pickle
import time

train = False

loadStart = time.time()
x_train, x_test, sr = openWav.loadData()

x_train = np.resize(x_train, (x_train.shape[0], sr, 1)).astype(np.float32)
x_test = np.resize(x_test, (x_test.shape[0], sr, 1)).astype(np.float32)

print(x_train.shape)
print(x_test.shape)
loadEnd = time.time()
print 'Loading the data in' , loadEnd - loadStart, 'seconds'
input_sample = Input(shape=(sr, 1))

x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 16, border_mode='same', activation="tanh")(x)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(1, 8, border_mode='same', activation="tanh")(x)
encoded = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)

x = UpSampling1D(length=2)(encoded)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
decoded = Convolution1D(1, 32, border_mode='same', activation="tanh")(x)

autoencoder = Model(input_sample, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

if train:
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
    autoencoder.save_weights("weights_conv.dat", True)
else:
    print 'Loading the model...'
    autoencoder.load_weights("weights_conv.dat")
                
decoded_samples = autoencoder.predict(x_test)



#pickle.dump(x_test, open('conv_x_test.p', 'wb'))
#pickle.dump(decoded_samples, open('conv_decoded.p', 'wb'))
