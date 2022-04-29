import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation
from keras.layers import concatenate, add, Lambda
from keras.models import Model, Sequential
import keras.backend as K

from esgflib.data.datasets import get_melbourne_data, split_train_test_data, create_test_data
from matplotlib import pyplot as plt
from esgflib.features.features import create_training_points

import pickle

def rse(y_true, y_pred):
    #
    # The formula is:
    #           K.sqrt(K.sum(K.square(y_true - y_pred)))     
    #    RSE = -----------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)))       
    #
    #           K.sqrt(K.sum(K.square(y_true - y_pred))/(N-1))
    #        = ----------------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)/(N-1)))
    #
    #
    #           K.sqrt(K.mean(K.square(y_true - y_pred)))
    #        = ------------------------------------------
    #           K.std(y_true)
    #
    num = K.sqrt(K.mean(K.square(y_true - y_pred), axis=None))
    den = K.std(y_true, axis=None)
    
    return num / den


def corr(y_true, y_pred):
    #
    # This function calculates the correlation between the true and the predicted outputs
    #
    num1 = y_true - K.mean(y_true, axis=0)
    num2 = y_pred - K.mean(y_pred, axis=0)
    
    num  = K.mean(num1 * num2, axis=0)
    den  = K.std(y_true, axis=0) * K.std(y_pred, axis=0)
    
    return K.mean(num / den)


def LSTNetmodel(hidC = 30,Ck = 3,hidR = 20,hidS = 30,skip = 15,loss = "mse",pt =3):
    # x is the number of time-series
    x = Input(shape=(history_days, 1))

    # CNN
    c = Conv1D(hidC, Ck, activation='tanh', use_bias = True, kernel_regularizer = keras.regularizers.L1(0.005))(x)

    # RNN
    # GRU
    # Apply a GRU layer (with activation set to 'tanh' as per the paper) and take the returned states as result
    r = GRU(hidR, kernel_regularizer = keras.regularizers.L1(0.005),activation='tanh')(c)
    r = Lambda(lambda k: K.reshape(k, (-1, hidR)))(r)

    # skip-GRU
    if skip > 0:
        # Reshape the data to easily setup the skip-RNN
        # Calculate the number of values to use which is equal to the window divided by how many time values to skip
        s = Lambda(lambda k: k[:, int(-pt*skip):, :])(c)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, skip, hidC)))(s)
        s = Lambda(lambda k: K.permute_dimensions(k, (0,2,1,3)))(s)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, hidC)))(s)

        s = GRU(hidS, kernel_regularizer = keras.regularizers.L1(0.005))(s)
        s = Lambda(lambda k: K.reshape(k, (-1, skip*hidS)))(s)
        # Concatenate the outputs of GRU and SkipGRU
        r = concatenate([r,s])
    # Dense layer
    res = Dense(horizon_days, kernel_regularizer = keras.regularizers.L1(0.005))(r)
    
    res = Activation("relu")(res)
    # Generate Model
    model = Model(inputs=x, outputs=res)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics = [keras.metrics.mse])
    return model

melbourne_data = get_melbourne_data()


test_years = [1987,1988,1989]
test_months = [3,6,12]

eval={}

for year in test_years:
    eval[year] = []
    for month in test_months:
        print("Year:", year, "prediction of the next", month, "months")
        # Will evaluate the model on the next 30*month 
        horizon_days = month *30
        #Models will be trained on 2 times the history to make sure to encounter the patterns to predicts
        history_days = horizon_days * 2
        
        
        # Get data
        train_data, _ = split_train_test_data(melbourne_data, split_year=year)
        X_test, Y_test = create_test_data(melbourne_data, history_days=history_days, horizon_days=horizon_days, year = year)
        # Index
        train_data.set_index('Date', inplace=True)

        # Create test points

        X_train, Y_train = create_training_points(train_data, history_days=history_days, horizon_days=horizon_days)
        
        
        #Configuration of the model
        
        model = LSTNetmodel(hidC = 30,Ck = 3,hidR = 20,hidS = 30,skip = 15,loss = "mse",pt =int((history_days-3)/15))
        model.fit(X_train, Y_train, epochs = horizon_days * 10, batch_size = X_train.shape[0], verbose = False)
        eval[year].append(model.evaluate(X_test, Y_test)[1])
