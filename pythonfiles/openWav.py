from __future__ import print_function
import librosa
import matplotlib.pyplot as plt


def loadData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    y, sr = librosa.load(audiofile, sr=None)
    y_test = reshape(y[1:44100*50], (50, 44100))
    y_train = reshape(y[44100*50:], (450, 44100))
    return y_test, y_train
