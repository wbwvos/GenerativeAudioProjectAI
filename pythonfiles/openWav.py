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
    test = test[:-1,:]
    train = train[:-1,:]
    seed = np.zeros(train[0].shape)
    
    x_train = np.vstack((seed, train))
    y_train =  np.vstack((train, seed))
    
    x_test = np.vstack((seed, test))
    y_test =  np.vstack((test, seed))    
    
    x_train = np.resize(x_train, (x_train.shape[0], sr, 1)).astype(np.float32)
    y_train = np.resize(y_train, (y_train.shape[0], sr, 1)).astype(np.float32)
    #x_test = np.resize(x_test, (x_test.shape[0], sr, 1)).astype(np.float32)
    #y_test = np.resize(y_test, (y_test.shape[0], sr, 1)).astype(np.float32)
    
 
    return x_train, y_train, x_test, y_test, sr

def lstmDataStream():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    totalSamples = 1000
    testPercentage = 0.10
    trainPercentage = 0.90
    testSamples = testPercentage * totalSamples
    sr = 2048
    trainSamples = int(trainPercentage * totalSamples*sr)
    y, sr = librosa.load(audiofile, sr=sr)
    #print(y.shape)
    test = y[trainSamples:]
    train = y[:trainSamples]

    seed = np.zeros(train[0].shape)
    
    x_train = np.append(seed, train[:-1,])
    y_train = train
    
    x_test = np.append(seed, test[:-1,])
    y_test = test   
    x_train = np.resize(x_train, (x_train.shape[0], 1, 1)).astype(np.float32)
    y_train = np.resize(y_train, (y_train.shape[0], 1, 1)).astype(np.float32)
    x_test = np.resize(x_test, (x_test.shape[0], 1, 1)).astype(np.float32)
    y_test = np.resize(y_test, (y_test.shape[0], 1, 1)).astype(np.float32)
    return x_train, y_train, x_test, y_test, sr
    
def sequenceData(sr = 2048, xsize = 128, ysize = 1):
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    y, sr = librosa.load(audiofile, sr=sr)
    x_train = []
    y_train = []

    for i in range(0, len(y)-xsize-ysize):
        x_train.append(y[i:i+xsize])
        y_train.append(y[i+xsize:i+xsize+ysize])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test = x_train[-127992:]
    y_test = y_train[-127992:]
    
    x_train = x_train[:1919880]
    y_train = y_train[:1919880]  
    
    return x_train, y_train, x_test, y_test, sr
x_train, y_train, x_test, y_test, sr = lstmDataStream()