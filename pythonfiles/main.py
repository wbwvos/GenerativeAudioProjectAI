import autoencoders as ae
import LSTM_model
import openWav

def main():
    #if switch == 1:
    x_train, x_test, sr = openWav.loadData()
    encoder, decoder = ae.getConvAutoEncoderModel(sr, x_train, x_test)
    print(encoder)
    print(decoder)
    #else: 
    #    if switch == 2:
    #x_train, y_train, x_test, y_test, sr = openWav.lstmData()
    #lstmModel = trainLSTM(x_train, y_train, x_test, y_test, encoder)
    #    else: 
    #        if switch == 3:
    #lsmtModel.predict(seed)  

def trainLSTM(x_train, y_train, x_test, y_test, encoder):
    x_train_encoded = encoder.predict(x_train)
    y_train_encoded = encoder.predict(y_train)
    x_test_encoded = encoder.predict(x_test)
    y_test_encoded = encoder.predict(y_test)
    lstm = LSTM_model.train(x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded)
    return lstm
    
def predict(seed):
    encoded = encoder.predict(seed)
    sequence = lstm.predict(encoded)
    decoded = decoder.predict(sequence)

main()
