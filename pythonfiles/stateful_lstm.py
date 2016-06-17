'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import openWav

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 80
epochs = 10
# number of elements ahead that are used to make the prediction
lahead = 1

x_train, y_train, x_test, y_test, sr = openWav.lstmDataStream('../data/drums/128bpm hip1.wav')

#def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
#    """Generates an absolute cosine time series with the amplitude
#    exponentially decreasing
#
#    Arguments:
#        amp: amplitude of the cosine function
#        period: period of the cosine function
#        x0: initial x of the time series
#        xn: final x of the time series
#        step: step of the time series discretization
#        k: exponential rate
#    """
#    cos = np.zeros(((xn - x0) * step, 1, 1))
#    for i in range(len(cos)):
#        idx = x0 + i * step
#        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
#        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
#    return cos
#
#
#print('Generating Data')
#cos = gen_cosine_amp()
#print('Input shape:', cos.shape)
#
#expected_output = np.zeros((len(cos), 1))
#for i in range(len(cos) - lahead):
#    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])
#
#print('Output shape')
#print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(x_test, batch_size=batch_size)

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
        model.save_weights(weights_filename, True)

import pickle 
expected_output = y_test
pickle.dump(predicted_output, open('predicted.p','wb'))
pickle.dump(expected_output, open('expected.p','wb'))
#
#print('Ploting Results')
#plt.subplot(2, 1, 1)
#plt.plot(expected_output)
#plt.title('Expected')
#plt.subplot(2, 1, 2)
#plt.plot(predicted_output)
#plt.title('Predicted')
#plt.show()
