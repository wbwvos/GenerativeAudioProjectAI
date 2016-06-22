'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Convolution1D, AveragePooling1D, TimeDistributed, Flatten
import openWav
import time
import os.path

# since we are using stateful rnn tsteps can be set to 1
tsteps = 64*4
batch_size = tsteps
epochs = 10
# number of elements ahead that are used to make the prediction
lahead = 1
sr = 2048
outputsize = 1
x_train, y_train = openWav.loadDrums2(tsteps, sr)
x_train = x_train[0]
y_train = y_train[0]
print(x_train[0].shape)
print(y_train[0].shape)

print('Creating Model')
model = Sequential()

model.add(Convolution1D(32, 32, border_mode='same', activation="tanh", batch_input_shape=(batch_size, tsteps, 1)))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(32, 16, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(8, 1, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))

model.add(LSTM(50,
               
               return_sequences=False,
               stateful=False))
               
model.add(Dense(outputsize))

model.compile(loss='mse', optimizer='rmsprop')
model.summary()
weights_filename = 'weights_fullmodel.dat'
if os.path.isfile(weights_filename):
    print('Loading the model...')
    model.load_weights(weights_filename)
else:
    print('Training the model...')
    trainStart = time.time()
    for i in range(epochs):
        print('Epoch', i+1, '/', epochs)
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()
    trainEnd = time.time()
    print('Trained the model in', trainEnd - trainStart, 'seconds')
    print('Saving the model...')
    model.save_weights(weights_filename, True)



total = []
print('Predicting')
prime = x_train[:batch_size]
print('prime shape:', prime.shape)
generations = batch_size*10
print('Predicting prime')
i=0
predicted_output = model.predict(prime, batch_size=batch_size, verbose=True)

while(i<1000):
    if i == 0:
        prime = np.resize(predicted_output, (1, predicted_output.shape[0], 1))
    else:
       prime = np.append(prime, predicted_output)
       prime = prime[1:]
    
    print(prime.shape)
    
    predicted_output = model.predict(prime, batch_size=batch_size, verbose=True)
    
    total = np.append(total, predicted_output)
    i+=1

print(predicted_output)
print(predicted_output.shape)
#print(predicted_output)
#print(len(predicted_output))
#total = predicted_output
print('total generations:', generations)
#for i in range(generations):
#    last_batch = total[-batch_size*batch_size:]
#    #print('last batch shape:', last_batch.shape)
#    last_batch = np.resize(last_batch, (last_batch.shape[0]/batch_size, batch_size, 1))
#    #print('last batch shape:', last_batch.shape)
#    predicted_output_batch = model.predict(last_batch, batch_size=batch_size)
#    print(predicted_output_batch)
#    break
#    predicted_value = predicted_output_batch[-1]
#    total = np.append(total, predicted_value)
#    if i % 64 == 0:
#        print(i, 'predicted:' ,predicted_value)
    #print(total[-10:])
     

import pickle 
#expected_output = y_train
#pickle.dump(predicted_output, open('predicted.p','wb'))
#print('Saved predicted_output to predicted.p')
#pickle.dump(prime, open('expected.p','wb'))
#print('Saved expected_output to expected.p')
pickle.dump(total, open('fullmodel_generated.p','wb'))
#print('Saved generated_output to generated.p')

#
#print('Ploting Results')
#plt.subplot(2, 1, 1)
#plt.plot(expected_output)
#plt.title('Expected')
#plt.subplot(2, 1, 2)
#plt.plot(predicted_output)
#plt.title('Predicted')
#plt.show()
