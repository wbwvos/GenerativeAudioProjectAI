from keras.layers import Input, Dense, Convolution1D, AveragePooling1D, UpSampling1D
from keras.models import Model
import openWav
import numpy as np

x_train, x_test, sr = openWav.loadData()

x_train = np.resize(x_train, (x_train.shape[0], sr, 1)).astype(np.float32)
x_test = np.resize(x_test, (x_test.shape[0], sr, 1)).astype(np.float32)

print(x_train.shape)
print(x_test.shape)
input_sample = Input(shape=(1, sr))

x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(input_sample)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(32, 16, border_mode='same', activation="tanh")(input_sample)
x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
x = Convolution1D(1, 8, border_mode='same', activation="tanh")(input_sample)
encoded = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)

x = UpSampling1D(length=2)(encoded)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(32, 32, border_mode='same', activation="tanh")(x)
decoded = Convolution1D(1, 32, border_mode='same', activation="tanh")(x)

autoencoder = Model(input_sample, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))
                
decoded_samples = autoencoder.predict(x_test)

print decoded_samples[1]