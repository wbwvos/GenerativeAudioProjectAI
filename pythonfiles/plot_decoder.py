import matplotlib.pyplot as plt
import pickle

x_test = pickle.load(open("x_test.p"))
decoded_imgs = pickle.load(open("decoded_imgs.p"))

def plotData(x_test, decoded_imgs):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x_test[0])
    plt.figure(2)
    plt.subplot(212)
    plt.plot(decoded_imgs[0])
        # display reconstruction
        #ax = plt.subplot(2, n, i + n)
        #plt.imshow(decoded_imgs[i])
        #plt.gray()
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        #plt.show()
    
plotData(x_test, decoded_imgs)
