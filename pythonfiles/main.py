import autoencoders as ae
import LSTM_model
import openWav

def main(switch):
    if switch == 1:
        x_train, x_test, sr = openWav.loadData()
        encoder, decoder = ae.getConvAutoEncoderModel(sr, x_train, x_test)
    else: 
        if switch == 2:
            x_train, y_train, x_test, y_test, sr = openWav.lstmData()
            lstm, trainLSTM(x_train, y_train, x_test, y_test, encoder)
        else: 
            if switch == 3:
              predict(seed)  

def trainLSTM(x_train, y_train, x_test, y_test, encoder):
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    lstm = LSTM_model.train(encoded_train, encoded_test)
    return lstm
    
def predict(seed):
    encoded = encoder.predict(seed)
    sequence = lstm.predict(encoded)
    decoded = decoder.predict(sequence)