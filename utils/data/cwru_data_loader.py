"""
cwru_data_loader.py

a class to handle loading and splitting the cwru bearing data

we will use the windowed training data for cross validation and then the 
held out data (x_test) for blind fold testing

i have cleaned up and renamed the original cwru datafiles as there were
some errors / confusing bits in them. i have made some notes in 'notes' 
files in the appropriate data directory as to what i've changed

author: alex shenfield
date:   17/04/2020
"""

import re 

import numpy as np
import scipy.io as sio

from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# implementation of robust z-score normalisation - where we center and scale
# to have median = 0, and median absolute deviation = 1
def normalise(x, axis=0):
    
    # calculate the center and scale
    x_center = np.median(x, axis=axis)
    x_scale = np.median(np.abs(x - x_center), axis=axis)
    
    # normalise our data and return
    x_norm = (x - x_center) / x_scale
    return x_norm
    

# filter the keys from the matlab file to just match the ones we are 
# interested in
def filter_key(keys, variables):
    
    # iterate over all the file keys
    fkeys = list()
    for key in keys:
        
        # use the experiment number that matches the RPM key
        match = re.match( r'(.*)RPM', key, re.M|re.I)
        if match:
            fkeys.append(match.group(1))
    
    # if we have multiple keys matching the data then we have a problem and
    # need to clean up the data
    if(len(fkeys)>1):
        print('multiple data keys in file - please check {0}'.format(keys))
        
    # return the keys relating to the variables we are interested in
    var_keys = [(fkeys[0] + '_' + v) for v in variables]
    return var_keys


# dictionary mapping class names (i.e. faults) to integers
faults_idx = {
    'normal': 0,
    'b007':   1,
    'b014':   2,
    'b021':   3,
    'ir007':  4,
    'ir014':  5,
    'ir021':  6,
    'or007c': 7,
    'or014c': 8,
    'or021c': 9,
}


# this is the main class for preprocessing and loading the data
class CWRUBearingData:
    
    """Load the CWRU bearing data.
    
    This class loads the specified bearing data matlab files, pulls out the 
    variables we are interested in, and then divides the data into train and 
    test sets.
    
    Arguments:
        root_dir       - the base directory where the data can be found
        experiment     - the subset of the data to use (e.g. 48k_drive_end_fault)
        loads          - the specific load conditions to read in (can be a list 
                         of multiple load conditions)
        normalisation  - which normalisation method to use - can be: 
                         'robust-zscore' (zero median and unit m.a.d)
                         'standard-scaler' (zero mean and unit std)
                         'robust-scaler' (uses the robust scaler from scikit-learn)
                         None (no normalisation applied)
    """
    
    # fields of interest within the data
    variable_fields = ['FE_time', 'DE_time']
    baseline_dir = '48k_normal_baseline'
    
    def __init__(self, root_dir, experiment, loads, 
                 normalisation='robust-zscore'):
        
        # get the paths
        self.loads = loads
        self.root_dir = root_dir
        self.data_dir = Path(root_dir, experiment)   
        
        # build a list of all the files
        filelist = list()
        for l in self.loads:
            filelist.extend(list((Path(self.root_dir, self.baseline_dir) / 
                                 '{0}hp'.format(l)).iterdir()))
            filelist.extend(list((self.data_dir / 
                                  '{0}hp'.format(l)).iterdir()))
        filelist.sort()
        self.files = filelist
        
        
        if normalisation:
            if normalisation == 'robust-zscore':
                self.transform = lambda x : normalise(x)
            elif normalisation == 'standard-scaler':
                self.scaler = StandardScaler()
                self.transform = lambda x : self.scaler.fit_transform(x)
            elif normalisation == 'robust-scaler':
                self.scaler = RobustScaler()
                self.transform = lambda x : self.scaler.fit_transform(x)
        else:
            self.transform = lambda x : x
                
    
    # retrieve the label from the file name
    def _get_class(self, f):        
        c = None
        for k in faults_idx.keys():
            if k in str(f):
                c = k
        return faults_idx[c]
    
      
    # change the variables we are interested in (before extracting the data!)
    def change_variables(self, variable_list):
        self.variable_fields = variable_list
        
        
    # extract the data and split it into train and test - we need to know the
    # minimum length of all the time serieses in our data set to use in the
    # splitting
    #
    # in the cwru 48k sampled data this is 381890 for ir014 @ 1hp load
    # in the cwru 12k sampled data this is 121265 for ir007 @ 0hp load
    #
    # we also need to specify the fraction of the data to window for use in 
    # training (remember, the test data is not windowed!)
    def split_data(self, data_length, train_fraction=0.5, window_length=2048, 
                   window_step=64, verbose=False):        
        
        # get the training end point and testing start point
        train_end = np.int_(np.around(data_length * train_fraction))
        
        # get the train and test splits using a fraction of the data for 
        # training (which we will window) and half for testing (which we wont)
        train_splits = np.arange(0, train_end, window_step)
        test_splits  = np.arange(train_end, data_length, window_length)
        
        # information
        if verbose:
            print('we are using {0} training samples and {1} testing samples '
                  'from each fault category'.format(len(train_splits), 
                                                    len(test_splits)))
        
        # initialise arrays for our training and testing data
        x_train = np.zeros((0, window_length, len(self.variable_fields)))
        x_test  = np.zeros((0, window_length, len(self.variable_fields)))        
        y_train = list()
        y_test  = list()
        
        # for all the data files we are interested in
        for f in self.files:
            
            # print the file we are processing
            if verbose:
                print(str(f))
            
            # load the raw data from the matlab file
            data = sio.loadmat(f)
            
            # pull out the variables of interest and turn the data into a 'n' 
            # dimensional timeseries
            keylist = filter_key(data.keys(), self.variable_fields)
            ts_data = np.hstack([data[key] for key in keylist])
            
            # normalise using specified scaling / normalisation method
            # --wrt z score using the median scaling--
            #ts_data = normalise(ts_data)
            ts_data = self.transform(ts_data)
            
            # really we should fit the scaler to the training set and then 
            # use those calculated values to transform the data - but tbh the
            # training set and the testing set look pretty much identical as 
            # they are gathered the same way from different parts of the same 
            # signal
            
            # print the timeseries shape so we can see how many samples and 
            # how many variables we have
            if verbose:
                print(ts_data.shape)
            
            # now window the data for our training set (checking we are
            # actually supposed to be producing training data ...)
            if train_splits.size:
                samples = list()
                for start in train_splits:
                    samples.append(ts_data[start:start+window_length])        
                x_train = np.vstack((x_train, np.stack(samples, axis=0)))
            
                # and finally generate our train labels
                y_train.extend([self._get_class(f)] * len(train_splits))
            
            # now get our test set in much the same way but without overlaps
            # in our windows (first checking we are actually supposed to be 
            # producing test data ...)
            if test_splits.size:
                samples = list()
                for start in test_splits:
                    samples.append(ts_data[start:start+window_length])        
                x_test = np.vstack((x_test, np.stack(samples, axis=0)))
            
                # and finally generate our test labels
                y_test.extend([self._get_class(f)] * len(test_splits))
            
        # return our training and testing data sets
        y_train = np.array(y_train)
        y_test  = np.array(y_test)
        return x_train, y_train, x_test, y_test