from __future__ import print_function
import librosa
import matplotlib.pyplot as plt


def loadData():
    audiofile = '../data/toy_data_sines_44_1khz.wav'
    y, sr = librosa.load(audiofile, sr=None)
    return y[1:44100*50], y[44100*50:]
