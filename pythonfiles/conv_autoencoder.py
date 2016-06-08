from keras.layers import Input, Dense, Convolution1D, AveragePooling1D, UpSampling1D
from keras.models import Model
import openWav

x_train, x_test, sr = openWav.loadData()

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