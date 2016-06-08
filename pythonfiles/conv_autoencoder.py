from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import openWav

x_train, x_test, sr = openWav.loadData()

input_sample = Input(shape=(1, sr))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_sample)
x = MaxPooling2D((2, 2), border_mode='valid')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='valid')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='valid')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='tanh', border_mode='same')(x)

autoencoder = Model(input_sample, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')