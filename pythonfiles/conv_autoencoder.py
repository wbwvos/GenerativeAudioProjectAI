from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import librosa
import openWav


train = True
x_test, x_train, sr = openWav.loadData()

model = Sequential()
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh", input_shape=(sr, 1)))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(32, 16, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(1, 8, border_mode='same', activation="tanh"))
# model.add(MaxPooling1D(pool_length=2, stride=None, border_mode="valid"))
#
# model.add(Convolution1D(4, 3, border_mode='same', activation="tanh"))
# model.add(UpSampling1D(length=2))
# model.add(Convolution1D(2, 3, border_mode='same', activation="tanh"))
model.add(UpSampling1D(length=2))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(UpSampling1D(length=2))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(UpSampling1D(length=2))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(Convolution1D(1, 32, border_mode='same', activation="tanh"))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
if train:
    print("NOW FITTING")
    model.fit(x_train, x_train, nb_epoch=500, batch_size=64)
    model.save_weights("weights_1.dat", True)

model.load_weights("weights_1.dat")


predictions = model.predict_on_batch(x_train)
error = mean_squared_error(np.resize(x_train, (len(x_train), sr)), np.resize(predictions, (len(predictions), sr)))
print("Train Error: %.4f" % error)
for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (sr,))
    librosa.output.write_wav('../data/decoded/%d.wav', prediction, sr, norm=True)
    # to_plot = wavfile.read("train_predictions/prediction_%d.wav" % (indices[i+100]+1))[1]
    # plt.plot(to_plot[:100])
    # plt.show()
    # print("hi")


predictions = model.predict_on_batch(x_test)
error = mean_squared_error(np.resize(x_test, (len(x_test), sr)), np.resize(predictions, (len(predictions), sr)))
print("Test Error: %.4f" % error)

for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (sr,))
    librosa.output.write_wav('../data/decoded/%dt.wav', prediction, sr, norm=True)

# one = wavfile.read("data/train/381.wav")[1]
# plt.plot(np.linspace(1,100))
# plt.plot(y[0][:500] * max)
# plt.plot(one[:5000])
# plt.show()