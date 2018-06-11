#!/usr/bin/env python

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.layers import TimeDistributed, Dense, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import cwrnn # clockwork rnn
from matplotlib.ticker import FormatStrFormatter


# different model builds
def build_model(input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(LSTM(10, input_shape=(input_dim_1, input_dim_2)))
    model.add(Dropout(0.5))
    model.add(Dense(y_input_dim_1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def build_timedistributed_model(input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(LSTM(100, input_shape=(input_dim_1, input_dim_2), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(y_input_dim_1, activation='softmax')))
    #model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_time_distributed_stateful_model(batch_size,input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(LSTM(64,
                   batch_input_shape=(batch_size, input_dim_1, input_dim_2),
                   stateful=True,
                   return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(y_input_dim_1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def build_stacked_model(input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(LSTM(100, input_shape=(input_dim_1, input_dim_2),
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, input_shape=(input_dim_1, input_dim_2)))
    model.add(Dropout(0.2))
    model.add(Dense(y_input_dim_1))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_stateful_model(batch_size,input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(LSTM(64,
                   batch_input_shape=(batch_size, input_dim_1, input_dim_2),
                   stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(y_input_dim_1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_cwrnn_model(input_dim_1, input_dim_2, y_input_dim_1):
    model = Sequential()
    model.add(cwrnn.ClockworkRNN(output_dim= y_input_dim_1,
                                 input_shape=(input_dim_1, input_dim_2),
                                 period_spec=[1,2,4,16]))
    model.add(Dropout(0.2))

    model.add(Dense(y_input_dim_1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
