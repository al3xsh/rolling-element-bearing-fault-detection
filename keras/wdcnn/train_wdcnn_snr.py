"""
train_wdcnn_snr.py

training script for the wdcnn model from Zhang, et al. 2017 to use
in comparison with our novel rnn-wdcnn-fcn model

this uses the cyclical learning rate approach from Smith (2017)

author: alex shenfield
date:   11/06/2020
"""

# suppress loggoing of info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# imports
import argparse

import numpy as np

from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

from models.my_wdcnn_model import generate_model

from utils.clr_callback import CyclicLR

from utils.data_utils import awgn
from utils.cwru_data_loader import CWRUBearingData

#
# main code
#

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

parser.add_argument('-clr_method', default='triangular2')
parser.add_argument('-base_lr', type=float, default=1e-5)
parser.add_argument('-max_lr', type=float, default=1e-3)
parser.add_argument('-step_size', type=int, default=8800)

# model parameters
parser.add_argument('-kernel_size', type=int, default=64)

# data parameters
parser.add_argument('-data_path', required=True)
# ...
# can include other data parameters here - e.g. normalisation, variables, etc

# parse the arguments
args = parser.parse_args()

#
# load the data
#

# define the length of the data window
window_length = 4096

# pick the variables to use
variables = ['DE_time', 'FE_time']

# load, normalise, and split the data
data_path  = args.data_path
experiment = '48k_drive_end_fault'
normalisation = 'robust-zscore'
data = CWRUBearingData(data_path, experiment, [1,2,3], normalisation=normalisation)
data.change_variables(variables)
x_train, y_train, x_test, y_test = data.split_data(360000, 
                                                   train_fraction=0.5,
                                                   window_step=64, 
                                                   window_length=window_length,
                                                   verbose=True)

# get the train - test split
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size=0.25, 
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

# use the wdcnn model
kernel_1 = args.kernel_size
model    = generate_model(n_class, x_train.shape[1], x_train.shape[2],
                          first_kernel=kernel_1)

#
# set the optimiser and the CLR strategy
#

# the number of clr cycles can be calculated by n_epochs / step_size / 2
# we don't want too many cycles or too few cycles*, and we can effectively set
# this using the step size parameter
#
# * particularly when using triangular2 with a limited number of epochs, 
# as the triangles decay and therefore the model starts learning less
# and stagnates a bit towards the tail end of the training process

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

# plot the results

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


# get timings
t_end = time.time()
print(t_end)
print('elapsed time = {0}'.format(t_end - t_start))
