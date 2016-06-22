'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import openWav
import time
import os.path
import autoencoders as ae
# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 1
epochs = 5
dim = 32
# number of elements ahead that are used to make the prediction
sr = 44100
conv_pack_input_size = 512
x_train, y_train = openWav.loadDrumsConv(tsteps, conv_pack_input_size, sr)
encoder, decoder = ae.getSplitConvAutoEncoder()
x_train_e = openWav.encodeDrums(x_train, encoder)
y_train_e = openWav.encodeDrums(y_train, encoder)
#x_train = x_train[0]
#y_train = y_train[0]

max_timesteps = (x_train_e.shape[0]/batch_size)*batch_size
x_train_e = x_train_e[:max_timesteps]
y_train_e = np.reshape(y_train_e[:max_timesteps], (max_timesteps, dim))
print(x_train_e.shape)
print(y_train_e.shape)
print('Creating Model')
model = Sequential()
model.add(LSTM(128,
               batch_input_shape=(batch_size, tsteps, dim),
               return_sequences=True,
               stateful=True))
model.add(LSTM(128,
               #batch_input_shape=(batch_size, tsteps, dim),
               return_sequences=False,
               stateful=True))
model.add(Dense(32))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

weights_filename = 'weights_conv_sequence_lstm.dat'
if os.path.isfile(weights_filename):
    print('Loading the model...')
    model.load_weights(weights_filename)
else:
    print('Training the model...')
    trainStart = time.time()
    for i in range(epochs):
        print('Epoch', i+1, '/', epochs)
        model.fit(x_train_e,
                  y_train_e,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()
    trainEnd = time.time()
    print('Trained the model in', trainEnd - trainStart, 'seconds')
    print('Saving the model...')
    model.save_weights(weights_filename, True)



print('Predicting')
prime = x_train_e[:batch_size*80]
print('prime shape:', prime.shape)
generations = batch_size*80
print('Predicting prime')
predicted_output = model.predict(prime, batch_size=batch_size, verbose=True)
print(predicted_output.shape)
#print(predicted_output)
#print(len(predicted_output))
total = predicted_output
print('total generations:', generations)
for i in range(generations):
    print('totalshape:', total.shape)
    last_batch = total[-batch_size]
    print('last batch shape:', last_batch.shape)
    last_batch = np.reshape(last_batch, (batch_size, dim, 1))
    print('last batch shape:', last_batch.shape)
    predicted_output_batch = model.predict(last_batch, batch_size=batch_size, verbose=True)
    predicted_value = predicted_output_batch[-1]
    total = np.vstack([total, predicted_value])
    #if i % 64 == 0:
    #print(i, 'predicted:', predicted_value)
    #model.reset_states()
    #print(total[-10:])
     

print('total.shape:', total.shape)


total = np.reshape(total, (total.shape[0], total.shape[1] , 1))
print('total.shape', total.shape)
sound = openWav.decodeDrums(total, decoder)

print(sound.shape)

import pickle 
#expected_output = y_train
#pickle.dump(predicted_output, open('predicted.p','wb'))
#print('Saved predicted_output to predicted.p')
#pickle.dump(prime, open('expected.p','wb'))
#print('Saved expected_output to expected.p')
pickle.dump(sound, open('generated_sound.p','wb'))
print('Saved generated_output to generated_sound.p')

#
#print('Ploting Results')
#plt.subplot(2, 1, 1)
#plt.plot(expected_output)
#plt.title('Expected')
#plt.subplot(2, 1, 2)
#plt.plot(predicted_output)
#plt.title('Predicted')
#plt.show()
