from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error
#from keras.callbacks import EarlyStopping
from keras.layers import ConvLSTM2D

from keras.layers import LSTM, Flatten

def lstm_model_1_(SEQ_SIZE, NUM_CLASSES):

    model = Sequential()
    model.add(LSTM(64, input_shape = (None, SEQ_SIZE)))
    model.add(Dense(32))
    model.add(Dense(1))

    return model

def lstm_model_2_(SEQ_SIZE, NUM_CLASSES):

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape = (None, SEQ_SIZE)))
    model.add(LSTM(50, activation = "relu"))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def lstm_model_3_(SEQ_SIZE, NUM_CLASSES):

    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape = (None, SEQ_SIZE))))
    model.add(Dense(1))
    return model