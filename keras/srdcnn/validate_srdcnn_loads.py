"""
validate_srdcnn_loads.py

validation script for the srdcnn model from Zhuang, et al. 2019 to use
in comparison with our novel rnn-wdcnn-fcn model

this uses the cyclical learning rate approach from Smith (2017)

srdcnn seems really difficult to train - the learning rate chooser shows a 
really weird loss plot (for the model without dropout) with multiple peaks
and troughs

it seems a bit easier to train with plain old adam though (so maybe we 
should use that ...)

author: alex shenfield
date:   11/06/2020
"""

# suppress loggoing of info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# imports
import time
import pickle
import numpy as np

import argparse

from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

# my imports
from models.my_srdcnn_model import generate_model

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

# data parameters
parser.add_argument('-data_path', required=True)
parser.add_argument('-source', type=int, action='append')
parser.add_argument('-target', type=int, action='append')
# ...
# can include other data parameters here - e.g. normalisation, variables, etc

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

# define the length of the data window
window_length = 4096

# pick the variables to use
variables = ['DE_time', 'FE_time']

# load, normalise, and split the 3hp data
data_path = args.data_path
experiment = '48k_drive_end_fault'
normalisation = 'robust-zscore'
data_1hp = CWRUBearingData(data_path, experiment, source, normalisation=normalisation)
data_1hp.change_variables(variables)
x_data, y_data, _, _ = data_1hp.split_data(360000, 
                                           train_fraction=1.0,
                                           window_step=64, 
                                           window_length=window_length,
                                           verbose=False)

#
# we're using the testing set of one of the target domains here as our 
# validation set just to make sure we're behaving sensibly during training
#

# target domain [0]         
         
# get the data (using testing sets - i.e. non overlapping windows)
val_data = CWRUBearingData(data_path, experiment, [target[0]], 
                           normalisation=normalisation)
val_data.change_variables(variables)
_, _, x_data_v, y_data_v = val_data.split_data(360000,  
                                               train_fraction=0.0,
                                               window_length=window_length,
                                               verbose=False)
y_data_v  = to_categorical(y_data_v)

#
# this just gives us a sanity check as we train :)
#

#
# set the model parameters
#

# model structure
n_class  = len(np.unique(y_data))

# training parameters
n_epochs   = args.n_epochs
batch_size = args.batch_size

# ^ small batch sizes yield better generalisation (apparently ...) ^

# set up the clr callback scheduler
clr_method = args.clr_method
base_lr = args.base_lr
max_lr = args.max_lr
step_size = args.step_size

#
# note: I have precalculated the base_lr, max_lr, and step_size based on 
# previous experiments for adam - with the step size representing 3 cycles
# over 25 epochs
#

# set the root directory for results
results_dir = ('./weights/srdcnn_loads_{0}_to_{1}/' +
               'cross_validation_{2}/').format(''.join([str(s) for s in source]),
                                               ''.join([str(t) for t in target]),
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

#
# initialise lists for storing data
#

# store some data from the crossvalidation training process
xval_history = list()
final_accuracy = list()
final_loss = list()

# store the accuracy results
accuracies = [list() for i in range(len(target))] 

# store the ground truths and predictions for target load(s) (for each fold) 
# so we can produce confusion plots later
ground_truth = [list() for i in range(len(target))] 
predictions  = [list() for i in range(len(target))] 

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
    model = generate_model(n_class, x_data.shape[1], x_data.shape[2])
    
    # (re)-initialise the optimiser and compile the model
    opt = SGD(lr=0.0, momentum=0.9, nesterov=True)
    clr = CyclicLR(
    	mode=clr_method, 
    	base_lr=base_lr,
    	max_lr=max_lr,
    	step_size=step_size
    )
    clr._reset()
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, metrics=['acc'])

    # train the model
    history = model.fit(x_train, 
                        y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(x_data_v, y_data_v),
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
    
    # get the model performance at different loads (to look at it's 
    # generalisation ability)
    
    # test against all target loads         
    for ix, l in enumerate(target):
        
        # get the data (using testing sets - i.e. non overlapping windows)
        t_data = CWRUBearingData(data_path, experiment, [l], normalisation=normalisation)
        t_data.change_variables(variables)
        _, _, x_data_t, y_data_t = t_data.split_data(360000,  
                                                     train_fraction=0.0,
                                                     window_length=window_length,
                                                     verbose=False)
        
        # reformat the data for use in predictions
        x_t = x_data_t
        y_true_t = y_data_t
        y_t = to_categorical(y_data_t)
        
        # evaluate
        scores_t = model.evaluate(x=x_t, y=y_t, batch_size=128, verbose=0)
        print('training on {0}hp testing on {1}hp = {2}'.format(source, 
                                                                l,
                                                                scores_t[1]))
        
        # what do we get wrong?
        y_pred_t = model.predict(x_t)
        
        # store the results
        accuracies[ix].append(scores_t[1])
        ground_truth[ix].append(y_t)
        predictions[ix].append(y_pred_t)
        
    
    # display the results for this fold so we can see how we are doing ...
    
    # done testing generalisation ...
        
    # next fold ...
    fold = fold + 1

#
# tidy up ...
#
    
# log our results ...

# print the experiments we were running
print('results for the srdcnn model')

# print the final results
print('overall performance:')
print('{0:.5f}% (+/- {1:.5f}%)'.format(
        np.mean(final_accuracy), 
        np.std(final_accuracy))
     ) 
    
# print the generalisation performance
print('generalisation performance:')
for ix, l in enumerate(target):
    print('{0}hp -> {1}hp = {2:.5f}% (+/- {3:.5f}%)'.format(
          source, l,
          np.mean(accuracies[ix]), 
          np.std(accuracies[ix]))
         ) 

# save everything i think might be at all important ...

# save those results to the parameters file
with open('{0}parameters.txt'.format(results_dir), 'a') as f:
    f.write('\n')
    f.write('results for the srdcnn model\n')
    f.write('overall performance:\n')
    f.write('{0:.5f}% (+/- {1:.5f}%)\n'.format(np.mean(final_accuracy), np.std(final_accuracy)))
    f.write('generalisation performance:\n')
    for ix, l in enumerate(target):
        f.write('{0}hp -> {1}hp = {2:.5f}% (+/- {3:.5f}%)\n'.format(
                source, l,
                np.mean(accuracies[ix]), 
                np.std(accuracies[ix]))
               )
    f.write('all done!\n')

# pickle the entire cross validation history so we can use it later if we want
with open(results_dir + 'xval_history.pickle', 'wb') as file:
    pickle.dump(xval_history, file)

# save the predictions, ground truths, and accuracies for the different loads
# and the different models we have trained in different folds ...
np.savez(results_dir + 'cross_validation_results.npz', 
         source=source,
         target=target,
         y_pred=np.array(predictions), 
         gtruth=np.array(ground_truth),
         accuracy=np.array(accuracies))

# we're all done!
print('all done!')

