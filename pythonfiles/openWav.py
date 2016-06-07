from __future__ import print_function
import librosa
#import matplotlib.pyplot as plt
import numpy as np
import random



def loadData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    y, sr = librosa.load(audiofile, sr=None)
    B = random.sample(range(0, 999), 100)
    X = np.reshape(y, (1000, 44100))
    X_test = X[B,:]
    X_train = np.delete(X, B, axis=0)
    print(X_test.shape)
    print(X_train.shape)
    
    return X_test, X_train
    