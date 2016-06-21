from __future__ import print_function
print('importing openWav...')

import numpy as np
import librosa
import matplotlib.pyplot as plt
#import numpy as np
import random
import sys
import os.path

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

def lstmDataStream(audiofile = '../data/toy_data_sines_44_1khz.wav'):
    totalSamples = 1000
    testPercentage = 0.10
    trainPercentage = 0.90
    testSamples = testPercentage * totalSamples
    sr = 2048*4
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
    #y_train = np.resize(y_train, (y_train.shape[0], 1, 1)).astype(np.float32)
    x_test = np.resize(x_test, (x_test.shape[0], 1, 1)).astype(np.float32)
    #y_test = np.resize(y_test, (y_test.shape[0], 1, 1)).astype(np.float32)
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
#x_train, y_train, x_test, y_test, sr = lstmDataStream()
    
def loadDrums(batchsize, sr = 2048):
    from os import listdir
    rootdir = '../data/drums/'
    #os.path.abspath(
    
    audiofiles = [];
    
    
    for i, file in enumerate(listdir(rootdir)):
        y, sr = librosa.load(rootdir+file, sr=sr)
        audiofiles.append(y)

    train = audiofiles
    x_train = []
    y_train = []
    seed = np.zeros(batchsize)
    for i in range(0, len(train)): 
        #trainexample = np.append(seed, train[i])
        #trainexample = trainexample[:-batchsize]
        trainexample = train[i]
        trainexample = np.resize(trainexample, (trainexample.shape[0], 1, 1)).astype(np.float32)
        x_train.append(trainexample)
        trainexampleY = np.resize(train[i], (train[i].shape[0], 1)).astype(np.float32)
        y_train.append(trainexampleY)
        
    return x_train, y_train

def loadDrums2(timesteps, sr = 2048):
    from os import listdir
    rootdir = '../data/drums/'
   
    

    audiofiles = []
    #audiofiles = [[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]]
    
    for i, file in enumerate(listdir(rootdir)):
        y, sr = librosa.load(rootdir+file, sr=sr)
        audiofiles.append(y)
        break
        
    
    #train = audiofiles
    x_train = []
    y_train = []
    trainexample = []
    exampleY = []
    xy_dim = timesteps+1
    
    for i in range(0, len(audiofiles)):
        for j in range(0, xy_dim):
            if(j == xy_dim-1): 
                exampleY = audiofiles[i][j:(len(audiofiles[i])-timesteps)+j]
            else:
                trainexample.append( audiofiles[i][j:(len(audiofiles[i])-timesteps)+j])
        trainexample = np.array(trainexample)
        trainexample = np.resize(trainexample, (trainexample.shape[0], trainexample.shape[1],1)).astype(np.float32)
        exampleY = np.array(exampleY)
        exampleY = np.resize(exampleY, (exampleY.shape[0],1)).astype(np.float32)
        x_train.append(np.swapaxes(trainexample, 0,1))
        y_train.append(exampleY)
        trainexample = []
        exampleY = []
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train
   

def loadDrumsConv(timesteps, pack_size = 512, sr = 44100):
    from os import listdir
    rootdir = '../data/drums/'

    audiofiles = []
    for i, file in enumerate(list(root)):
        y, sr = librosa.load(rootdit+file, sr=sr)
        audiofiles.append(y)
        break

    print audiofiles[0].shape
















 
