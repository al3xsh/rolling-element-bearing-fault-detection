"""
validate_rnn_wdcnn_snr_clr_sgd.py

cross validation script for the novel rnn fcn model with wide convolutions
after the input using the cwru_data_loader

this merges the ideas behind lstm-fcn (karim, et. al. 2017) with a wdcnn 
model inspired by (Zhang, et al. 2017), and adds a few tweaks and 
modifications to improve generalisation performance

so if we use the 48khz data, we should use a superwide first conv (e.g. 256),
if we use the data that has been resampled to 12khz we should use a less wide
kernel (e.g. 64). this is because the signal is a factor of 4 larger in the 
unsampled data!

author: alex shenfield
date:   20/05/2020
"""

# suppress loggoing of info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# imports
import time
import pickle
import numpy as np

import pandas as pd

import argparse

from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

# my imports
from models.rnn_wdcnn_model import generate_model

from utils.data_utils import awgn
from utils.clr_callback import CyclicLR
from utils.cwru_data_loader import CWRUBearingData

#
# main code
#

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

#
# take input arguments
#

# set up the argument parser (allowing inputs from a file)
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# training paramters
parser.add_argument('-n_epochs', type=int, default=25)
parser.add_argument('-batch_size', type=int, default=10)

parser.add_argument('-clr_method', default='triangular2')
parser.add_argument('-base_lr', type=float, default=1e-5)
parser.add_argument('-max_lr', type=float, default=1e-3)
parser.add_argument('-step_size', type=int, default=8800)

# model parameters
parser.add_argument('-rnn_type', default='gru')
parser.add_argument('-n_cells', type=int, default=16)
parser.add_argument('-kernel_size', type=int, default=256)
parser.add_argument('-recurrent_dropout', type=float, default=0.1)

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

# load, normalise, and split the 3hp data
data_path = args.data_path
experiment = '48k_drive_end_fault'
normalisation = 'robust-zscore'
data_3hp = CWRUBearingData(data_path, experiment, [1,2,3], normalisation=normalisation)
data_3hp.change_variables(variables)
x_data, y_data, _, _ = data_3hp.split_data(360000, 
                                           train_fraction=1.0,
                                           window_step=64, 
                                           window_length=window_length,
                                           verbose=False)

#
# set the model parameters
#

# model structure
n_class  = len(np.unique(y_data))
rnn_type = args.rnn_type
ncells   = args.n_cells
kernel_1 = args.kernel_size
rec_drop = args.recurrent_dropout

# training parameters
n_epochs   = args.n_epochs
batch_size = args.batch_size

# ^ small batch sizes yield better generalisation (apparently ...) ^

# set up the clr callback scheduler
clr_method = args.clr_method
base_lr = args.base_lr
max_lr = args.max_lr
step_size = args.step_size
clr = CyclicLR(
	mode=clr_method, 
	base_lr=base_lr,
	max_lr=max_lr,
	step_size=step_size
)

#
# note: I have precalculated the base_lr, max_lr, and step_size based on 
# previous experiments for adam - with the step size representing 3 cycles
# over 25 epochs
#

# set the root directory for results
results_dir = ('./weights/{0}_wdcnn_snr_clr/' +
               'cross_validation_{1}/').format(rnn_type, 
                                               time.strftime("%Y%m%d_%H%M"))
                                               
# write the parameters into a text file so we can see what experiment we ran
os.makedirs(results_dir, exist_ok=True)
with open('{0}parameters.txt'.format(results_dir), 'w') as f:
    f.write('data = {0}\n'.format(data_path))
    f.write('data set = {0}\n'.format(experiment))
    f.write('variables used = {0}\n'.format(str(variables)))
    f.write('normalisation = {0}\n'.format(normalisation))
    f.write('batch size = {0}\n'.format(batch_size))
    f.write('number of epochs = {0}\n'.format(n_epochs))
    f.write('clr method = {0} / {1} / {2} / {3} / sgd\n'.format(clr_method, 
                                                                base_lr, 
                                                                max_lr, 
                                                                step_size))
    f.write('window length = {0}\n'.format(window_length))
    f.write('rnn type = {0}\n'.format(rnn_type))
    f.write('wdcnn first kernel = {0}\n'.format(kernel_1))
    f.write('number of rnn cells = {0}\n'.format(ncells))
    f.write('recurrent dropout = {0}\n'.format(rec_drop))

#
# initialise lists for storing data
#

# store some data from the crossvalidation training process
xval_history = list()
final_accuracy = list()
final_loss = list()

snr_results_list = list()

#
# do stratified k-fold crossvalidation on the model
#

# progress ...
print('doing cross validation ...')

# create the kfold object with 10 splits
n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# run the crossvalidation
fold = 0
for train_index, test_index in kf.split(x_data, y_data):
  
    # progress ...
    print('evaluating fold {0}'.format(fold))
    
    # set up a model checkpoint callback (including making the directory where  
    # to save our weights)
    directory = results_dir + 'fold_{0}/'.format(fold)
    os.makedirs(directory, exist_ok=True)
    filename  = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=directory+filename, 
                                   verbose=0, 
                                   save_best_only=True)    
    # get train and test data
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_test, y_test   = x_data[test_index], y_data[test_index]     
    
    # one hot encode the labels 
    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)
    
    # build and compile the model
    model = generate_model(n_class, x_data.shape[1], x_data.shape[2],
                           rnn_type=rnn_type, ncells=ncells, rec_drop=rec_drop,  
                           first_kernel=kernel_1)
    
    # (re)-initialise the optimiser and compile the model
    opt = SGD(lr=0.0, momentum=0.9, nesterov=True)
    clr._reset()
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, metrics=['acc'])

    # train the model
    history = model.fit(x_train, 
                        y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer, clr])    

    # save the final model
    model.save(directory + 'final_model.h5')

    # store the training history
    xval_history.append(history.history)
    
    # print the validation result
    final_loss.append(history.history['val_loss'][-1])
    final_accuracy.append(history.history['val_acc'][-1])
    print('validation loss is {0} and accuracy is {1}'.format(final_loss[-1],
          final_accuracy[-1]))
    
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
        
    # record the snr results for this fold
    snr_results_list.append(scores_list)
        
    # next fold ...
    fold = fold + 1

#
# tidy up ...
#
    
# log our results ...

# print the experiments we were running
print('results for the novel wide rnn fcn model')

# get the data statistics formatted in a dataframe
df = pd.DataFrame([np.mean(snr_results_list, axis=0), 
                   np.std(snr_results_list, axis=0)]).T
df.index = snrs
df.columns = ['mean', 'std']

# print the generalisation performance
print('noise rejection performance:')
print(df) 

# save those results to the parameters file
with open('{0}parameters.txt'.format(results_dir), 'a') as f:
    f.write('\n')
    f.write('results for the novel wide rnn fcn model\n')
    f.write('overall performance:\n')
    f.write('{0:.5f}% (+/- {1:.5f}%)\n'.format(np.mean(final_accuracy), np.std(final_accuracy)))
    f.write('noise rejection performance:\n')
    f.write(str(df))
    f.write('\n')
    f.write('all done!\n')

# pickle the entire cross validation history so we can use it later
with open(results_dir + 'xval_history.pickle', 'wb') as file:
    pickle.dump(xval_history, file)

# save the accuracies for each snr for each fold
np.savez(results_dir + 'cross_validation_results_snr.npz', 
         accuracies=np.array(snr_results_list))

# we're all done!
print('all done!')
