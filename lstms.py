
#!/usr/bin/env python

# Do *not* edit this script.

import sys
from lstms import lstm_model_2_
import numpy as np
import pandas as pd

from pandas import read_csv
import math
from utils import *

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#from keras.callbacks import EarlyStopping
from keras.layers import ConvLSTM2D

from keras.layers import LSTM, Flatten


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model. \n python train_model.py "data/StandFord_2020.csv" "./model"')

    data_directory = sys.argv[1]
    model_directory = sys.argv[2]

    df = pd.read_csv(data_directory)
    df['Dates'] = pd.to_datetime(df["Dates"])
    df.set_index("Dates", inplace = True)
    dataset = df.values
    dataset = dataset.astype('float32') #COnvert values to float
    scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.66)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    seq_size = 10 # Number of time steps to look back 
    #Larger sequences (look further back) may improve forecasting.
    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)

    print("trainX: ",trainX.shape)
    print("trainY: ",trainY.shape)
    print("testX: ",testX.shape)
    print("testY: ",testY.shape)

    """# LSTMs """

    ################################################################################
    # Reshape for LSTM: [# signals, time steps, feat]

    print("trainX.shape: ", trainX.shape)

    X_train = np.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
    X_test = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))

    print("X_train.shape LSTM: ",X_train.shape)
    print("X_test.shape LSTM: ",X_test.shape)

    model = lstm_model_2_(seq_size, 1)

    model.compile(loss = "mean_squared_error", optimizer = "adam")
    model.summary()
    print("Model ready")

    model.fit(X_train, trainY, 
            validation_data = (X_test, testY),
            verbose =2,
            epochs = 100)

    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # invert predictions back to prescaled values
    #This is to compare with original input values
    #SInce we used minmaxscaler we can now use scaler.inverse_transform
    #to invert the transformation.
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    #we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict
