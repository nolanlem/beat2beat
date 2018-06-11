#!/usr/bin/env python

import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
np.set_printoptions(precision=2,suppress=True)
from numpy import newaxis

import scipy
from librosa import display
import os
import glob
import tensorflow as tf
#tf.python.control_flow_ops = tf

import time
import datetime

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import cwrnn # clockwork rnn
from matplotlib.ticker import FormatStrFormatter

# set default matplotlib font
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

import warnings
warnings.filterwarnings('ignore')
# config file
import config.nn_config as nn_config
import models.model_builds as model_builds

nn_params = nn_config.get_nn_params()
model_type = "stateful-lstm"


xtr = np.load(nn_params["training data dir"] + nn_params["drummer"] + "_X.npy")
ytr = np.load(nn_params["training data dir"] + nn_params["drummer"] + "_Y.npy")


print "x tr shape:", xtr.shape # for metallica (173402), and batch_size needs to % 0
print "y tr shape:", ytr.shape

# to make xtr(dim_2) % batch_size == 0
xtr = xtr[:-2,:,:]
ytr = ytr[:-2,:]

print 'x tr shape lopped off:', xtr.shape

loss = []
batch_size = 10
num_iters = 20
num_epochs = 100

model = model_builds.build_stateful_model(
    batch_size,
    input_dim_1 = xtr.shape[1],
    input_dim_2 = xtr.shape[2],
    y_input_dim_1 = ytr.shape[1]
    )

# call back function to save weights after every epoch (or iteration through batch here)
filepath="./model-weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
filepath = nn_params["weightsdir"] + model_type + "_weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



for i in range(num_iters):
    print '... running batch %r/%r...' %(i,num_iters)
    history = model.fit(xtr, ytr, nb_epoch=num_epochs, batch_size=batch_size,
             verbose=2, shuffle=False, callbacks=callbacks_list)
    loss.append(history.history['loss'])
    model.reset_states()

ts = time.gmtime()
modelsavefn = model_type + "-" + time.strftime("%c", ts).replace(' ', '_')


loss = np.array(loss)
np.save(nn_params["loss directory"] + "epochs:" + str(num_epochs) + "-" + modelsavefn + ".npy", loss)
model.save_weights(nn_params["weightsdir"] + modeltype + "_weights-improvement-{epoch:02d}-{loss:.2f}.h5")

print modelsavefn
model.save(nn_params["models dir"] + modelsavefn + "-{epoch:02d}-{loss:.2f}" + ".h5")
