from __future__ import print_function

import numpy as np
import librosa
#import matplotlib.pyplot as plt
#import numpy as np
import random



def loadData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    totalSamples = 1000
    testPercentage = 0.10
    trainPercentage = 0.90
    testSamples = testPercentage * totalSamples
    trainSamples = trainPercentage * totalSamples
    sr = 2048
    y, sr = librosa.load(audiofile, sr=sr)
    B = random.sample(range(totalSamples), int(testSamples))
    X = np.reshape(y, (totalSamples, sr))
    X_test = X[B,:]
    X_train = np.delete(X, B, axis=0)
    
    return X_train, X_test, sr
  

def lstmData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    totalSamples = 1000
    testPercentage = 0.10
    trainPercentage = 0.90
    testSamples = testPercentage * totalSamples
    trainSamples = trainPercentage * totalSamples
    sr = 2048
    y, sr = librosa.load(audiofile, sr=sr)
    
    X = np.reshape(y, (totalSamples, sr))
    test = X[trainSamples:,:]
    train = X[:trainSamples,:]

    seed = np.zeros(train[0].shape)
    
    x_train = np.vstack(seed, train)
    y_train =  np.vstack(train, seed)
    
    x_test = np.vstack(seed, test)
    y_test =  np.vstack(test, seed)    
    
    return x_train, y_train, x_test, y_test, sr