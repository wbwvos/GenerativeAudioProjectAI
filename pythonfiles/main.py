def main():
    x_train, x_test, sr = openWav.loadData()
    trainAutoencoder(x_train, x_test)
    
    
def trainAutoencoder(x_train, x_test):
    encoder, decoder = conv_autoencoder2.convEncoder(input_sample, True, x_train, x_test)
def train():
    x_train, x_test, sr = openWav.loadData()
    encoded = encoder.encoder(x_train, x_test, False, True)
    sequence = LSTM.train(True, encoded)
    
def predict():
    x_train, x_test, sr = openWav.loadData()
    encoded = encoder.encoder(x_train, x_test, False, True)
    sequence = LSTM.predict(True, encoded)
    decoded = encoder.encoder(x_train, x_test, False, False)