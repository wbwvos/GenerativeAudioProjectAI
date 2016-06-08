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
    sr = 1764
    y, sr = librosa.load(audiofile, sr=sr)
    B = random.sample(range(totalSamples), int(testSamples))
    X = np.reshape(y, (totalSamples, sr))
    X_test = X[B,:]
    X_train = np.delete(X, B, axis=0)
    print(X_test.shape)
    print(X_train.shape)
    
    return X_test, X_train, sr
  

