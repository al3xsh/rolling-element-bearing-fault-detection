"""
train_rnn_wdcnn_loads.py

training script for my model development with an rnn + wdcnn model using the
cwru_data_loader

this merges the ideas behind lstm-fcn (karim, et. al. 2017) with a wdcnn 
model inspired by (Zhang, et al. 2017), and adds a few tweaks and 
modifications to improve generalisation performance

this uses the cyclical learning rate approach from Smith (2017) for training
(which results in measurable, but marginal, gains over adam)

author: alex shenfield
date:   13/05/2020
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
parser.add_argument('-rnn_type', default='lstm')
parser.add_argument('-n_cells', type=int, default=128)
parser.add_argument('-kernel_size', type=int, default=256)
parser.add_argument('-recurrent_dropout', type=float, default=0.1)

# data parameters
parser.add_argument('-data_path', required=True)
parser.add_argument('-source', type=int, action='append')
parser.add_argument('-target', type=int, action='append')

# ...
# we can include other data parameters here - e.g. normalisation, variables, 
# etc

# parse the arguments
args = parser.parse_args()

#
# set the source and target domains for the experiment (so we don't have to 
# go through changing bits and pieces in the code)
#

# get them from the arguments
source = args.source
target = args.target

#
# load the data
#

# progress message ...
print('loading data ...')

# define the length of the data window
window_length = 4096

# pick the variables to use (use both channels for these experiments)
variables = ['DE_time', 'FE_time']

# load, normalise, and split the source data
data_path = args.data_path
experiment = '48k_drive_end_fault'
normalisation = 'robust-zscore'
source_data = CWRUBearingData(data_path, experiment, source, 
                              normalisation=normalisation)
source_data.change_variables(variables)
x_data, y_data, _, _ = source_data.split_data(360000, 
                                              train_fraction=1.0,
                                              window_step=64, 
                                              window_length=window_length,
                                              verbose=False)

# get the train - test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=y_data)

# one hot encode the labels 
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

#
# create and train the model
#

# set the model parameters
n_class    = y_train.shape[1]
n_epochs   = args.n_epochs
batch_size = args.batch_size

# ^ small batch sizes yield better generalisation ^

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
    # low and therefore we need to wait for it to go up a fair amount
    
    # plot the loss for the various learning rates
    lrf.plot_loss()
    sys.exit(0)

# ##########

# progress message ...
print('training model ...')

# fit the model
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=1)

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# plot the training results
plot = True
if plot:
    
    # import matplotlib to plot the results
    import matplotlib.pyplot as plt
    
    # plot accuracy
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
    
    # plot loss
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
    plt.show()


# test against different targets loads         
for l in target:
    
    # get the data (using testing sets - i.e. non overlapping windows)
    t_data = CWRUBearingData(data_path, experiment, [l], normalisation=normalisation)
    t_data.change_variables(variables)
    _, _, x_data_t, y_data_t = t_data.split_data(360000,  
                                                 train_fraction=0.0,
                                                 window_length=window_length,
                                                 verbose=False)
    x_t = x_data_t
    y_true_t = y_data_t
    y_t = to_categorical(y_data_t)
    
    # evaluate
    scores_t = model.evaluate(x=x_t, y=y_t, batch_size=128)
    print('training on {0}hp testing on {1}hp ='.format(source, l))
    print(scores_t)   
    
    # what do we get wrong?
    if plot:    
        from utils.visualisation import visualisation_utils as myvis
        y_pred_t = model.predict(x_t)
        f3 = plt.figure()
        myvis.plot_confusion_matrix(y_true_t, np.argmax(y_pred_t, axis=1))
        plt.show()    


# get final timings
t_end = time.time()
print(t_end)
print('elapsed time = {0}'.format(t_end - t_start))