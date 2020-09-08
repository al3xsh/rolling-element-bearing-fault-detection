"""
train_rnn_wdcnn_snr.py

training script for my model development with an rnn + wdcnn model

this merges the ideas behind lstm-fcn (karim, et. al. 2017) with a wdcnn 
model inspired by (Zhang, et al. 2017), and adds a few tweaks and 
modifications to improve generalisation performance

this file explores the model behaviour under different signal-to-noise ratios

author: alex shenfield
date:   20/04/2020
"""

# suppress loggoing of info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# imports
import argparse

import numpy as np

from keras.optimizers import SGD
from keras.utils import to_categorical

from models.rnn_wdcnn_model import generate_model

from utils.clr.clr_callback import CyclicLR

from utils.data.data_utils import awgn
from utils.data.cwru_data_loader import CWRUBearingData

#
# main code
#

# measure some timings
import time
t_start = time.time()
print(t_start)

#
# take input arguments
#

# set up the argument parser (allowing inputs from a file)
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# training paramters
parser.add_argument('-n_epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=10)

# clr parameters
parser.add_argument('-clr_method', default='triangular2')
parser.add_argument('-base_lr', type=float, default=1e-5)
parser.add_argument('-max_lr', type=float, default=1e-3)
parser.add_argument('-step_size', type=int, default=8800)

parser.add_argument('-lr_find', default=False, action='store_true')

# model parameters
parser.add_argument('-rnn_type', default='gru')
parser.add_argument('-n_cells', type=int, default=16)
parser.add_argument('-kernel_size', type=int, default=256)
parser.add_argument('-recurrent_dropout', type=float, default=0.1)

# data parameters
parser.add_argument('-data_path', required=True)

# ...
# we can include other data parameters here - e.g. normalisation, variables, 
# etc

# parse the arguments
args = parser.parse_args()

#
# load the data
#

# progress message ...
print('loading data ...')

# define the length of the data window
window_length = 4096

# pick the variables to use (use both channels for these experiments)
variables = ['DE_time', 'FE_time']

# load, normalise, and split the data
data_path  = args.data_path
experiment = '48k_drive_end_fault'
normalisation = 'robust-zscore'
data = CWRUBearingData(data_path, experiment, [1,2,3], 
                       normalisation=normalisation)
data.change_variables(variables)
x_train, y_train, x_test, y_test = data.split_data(360000, 
                                                   train_fraction=0.5,
                                                   window_step=64, 
                                                   window_length=window_length,
                                                   verbose=False)

# get the train - val split
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size=0.1, 
                                                  random_state=42,
                                                  stratify=y_train)

# one hot encode the labels 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val  = to_categorical(y_val)

#
# create and train the model
#

# set the model parameters
n_class = y_train.shape[1]
n_epochs   = args.n_epochs
batch_size = args.batch_size

# ^ small batch sizes yield better generalisation (apparently ...) ^

# use the rnn-fcn model
rnn_type = args.rnn_type
ncells   = args.n_cells
kernel_1 = args.kernel_size
rec_drop = args.recurrent_dropout
model    = generate_model(n_class, x_train.shape[1], x_train.shape[2],
                          rnn_type=rnn_type, ncells=ncells, rec_drop=rec_drop,  
                          first_kernel=kernel_1)

#
# set the optimiser and the CLR strategy
#

# the number of clr cycles can be calculated by n_epochs / step_size / 2
#
# we don't want too many cycles or too few cycles*, and we can effectively set
# this using the step size parameter
#
# * particularly when using triangular2 with a limited number of epochs, 
# as the triangles decay and therefore the model starts learning less
# and stagnates a bit towards the tail end of the training process
#
# i'm aiming to set the step size to ensure there are 3 cycles over the 
# course of the training

# cyclical learning rate scheduler callback

# sgd based clr
opt = SGD(lr=0.0, momentum=0.9, nesterov=True)
clr = CyclicLR(
 	mode=args.clr_method, 
 	base_lr=args.base_lr,
 	max_lr=args.max_lr,
 	step_size=args.step_size
)

callbacks = [clr]

# compile the model
#opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

# ##########

# run the learning rate finder
find_lr = args.lr_find
if find_lr:
    from utils.clr.learningratefinder import LearningRateFinder
    import sys
    lrf = LearningRateFinder(model, stopFactor=100)
    lrf.find([x_train, y_train],
    		1e-10, 1e+1,
            epochs=1,
    		stepsPerEpoch=np.ceil((len(x_train) / float(batch_size))),
    		batchSize=batch_size)
    
    # i *think* stopFactor has to be quite large because our loss gets quite 
    # low and therefore we need to wait for it to go up a load
    
    # plot the loss for the various learning rates
    lrf.plot_loss()
    sys.exit(0)

# ##########

# progress message ...
print('training model ...')

#
# training
#

# fit the model
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    shuffle=True,
                    verbose=1)

#
# process the results
#

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# plot the training results
plot = True
if plot:
    
    # import matplotlib to plot the results
    import matplotlib.pyplot as plt
    
    # accuracy
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.text(0.4, 0.05, 
             ('validation accuracy = {0:.3f}'.format(best_accuracy)), 
             ha='left', va='center', 
             transform=ax1.transAxes)
    plt.show()
    
    # loss
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.text(0.4, 0.05, 
             ('validation loss = {0:.3f}'
              .format(min(history.history['val_loss']))), 
             ha='right', va='top', 
             transform=ax2.transAxes)
    plt.show()
    
    # plot the learning rate history
    nval = np.arange(0, len(clr.history["lr"]))
    plt.figure()
    plt.plot(nval, clr.history["lr"])
    plt.title("cyclical learning rate (CLR)")
    plt.xlabel("training iterations")
    plt.ylabel("learning rate")

#
# test against different snrs
#

# specifiy the snr list to test against
snrs = [-4, -2, 0, 2, 4, 6, 8, 10, None]

# iterate over the snr list adding noise to the test data and then testing 
# the model with that noisy data
scores_list = list()
for snr in snrs:
    
    # create the noisy data
    x_noisy = list()
    if snr != None:
        for x in x_test:
            x_noisy.append(awgn(x,snr))
        x_noisy = np.array(x_noisy)
    else:
        x_noisy = x_test
        
    # test wdcnn
    score = model.evaluate(x_noisy, y_test, verbose=0)[1]*100
    print('snr is {0}, accuracy is {1}'.format(snr, score))
    scores_list.append(score)


# get final timings
t_end = time.time()
print(t_end)
print('elapsed time = {0}'.format(t_end - t_start))
