def main():
    

def train():
    x_train, x_test, sr = openWav.loadData()
    encoded = encoder.encoder(x_train, x_test, False, True)
    sequence = LSTM.train(True, encoded)
    
def predict():
    x_train, x_test, sr = openWav.loadData()
    encoded = encoder.encoder(x_train, x_test, False, True)
    sequence = LSTM.predict(True, encoded)
    decoded = encoder.encoder(x_train, x_test, False, False)