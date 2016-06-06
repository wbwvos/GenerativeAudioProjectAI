from __future__ import print_function
import librosa
#import matplotlib.pyplot as plt
import numpy as np



def loadData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    y, sr = librosa.load(audiofile, sr=None)
    testInterval = 100*44100
    #trainInterval = y.shape[0]- testInterval
    
    y_test = np.reshape(y[0:testInterval], (100, 44100))
    y_train = np.reshape(y[testInterval:], (900, 44100))
    print(y_test.shape)
    print(y_train.shape)
    return y_test#, y_train

loadData()