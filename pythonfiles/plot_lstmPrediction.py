import matplotlib.pyplot as plt
import pickle
import librosa
#e_name = "expected"
#p_name = "predicted"
#g_name = "generated"
#expected = pickle.load(open(e_name + '.p'))
#predicted = pickle.load(open(p_name + '.p'))
#generated = pickle.load(open(g_name + '.p'))
#
#generated.shape
#
#def plotData(expected, predicted):
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(expected[1000:3000])
#    plt.figure(1)
#    #plt.subplot(212)
#    plt.plot(generated[1000:3000], 'r')
#    
#        # display reconstruction
#        #ax = plt.subplot(2, n, i + n)
#        #plt.imshow(decoded_imgs[i])
#        #plt.gray()
#        #ax.get_xaxis().set_visible(False)
#        #ax.get_yaxis().set_visible(False)
#        #plt.show()
#    
#plotData(expected, predicted)
#sr = 8192
#librosa.output.write_wav(str(sr) + '_gen_' + e_name + '.wav', expected, sr)
#librosa.output.write_wav(str(sr) + '_gen_' + p_name + '.wav', predicted, sr)
#librosa.output.write_wav(str(sr) + '_gen_' + g_name + '.wav', generated, sr)
#


filename = '200_generated'
generated = pickle.load(open(filename + '.p'))
startgen = generated.flatten().shape[0]/4

plt.plot(generated.flatten())
plt.axvline(x=440320, color='r', linewidth=2)
plt.axis([0, 880640, -1, 1])



#librosa.output.write_wav(filename + '.wav', generated.flatten(), 44100)