"""
clr_callback.py

this implements the cyclical learning rate strategies from:
    
L. Smith (2017), "Cyclical Learning Rates for Training Neural Networks"
https://arxiv.org/abs/1506.011861

This is just a slight simplification of https://github.com/bckenstler/CLR

author: alex shenfield
date:   17/04/2020
"""

import numpy as np
import keras.backend as K

from keras.callbacks import Callback

class CyclicLR(Callback):
    
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, 
                 mode='triangular', gamma=1.):
            
        super().__init__()

        # set initial parameters
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        # set the clr mode
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1/(2.**(x-1))
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: gamma**(x)
            self.scale_mode = 'iterations'
            
        # initialise internal variables
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}

        # reset everything
        self._reset()
        
    
    # reset the iterations
    def _reset(self):        
        self.clr_iterations = 0
        
        
    # get the learning rate to use based on which clr strategy has been chosen
    def clr(self):
        
        # which clr cycle are we on? (a cycle is a complete up and down
        # of the learning rate)
        cycle = np.floor(1 + (self.clr_iterations / (2 * self.step_size)))
        x = np.abs((self.clr_iterations / self.step_size) - (2 * cycle + 1))
        
        # find the learning rate based on which clr stragey we are using and 
        # where we are at in the cycle
        if self.scale_mode == 'cycle':
            return self.base_lr + \
                (self.max_lr - self.base_lr) * np.maximum(0, (1-x)) * \
                    self.scale_fn(cycle)
        else:
            return self.base_lr + \
                (self.max_lr - self.base_lr) * np.maximum(0, (1-x)) * \
                    self.scale_fn(self.clr_iterations)
        
        
    #
    # keras callback methods
    #
    
    # start of training (i.e. called at the beginning of 'fit')
    def on_train_begin(self, logs=None):
        logs = logs or {}

        # set the learning rate for the optimiser based on where we are in
        # the clr process
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr()) 
            
          
    # called at the end of processing a batch of data
    def on_batch_end(self, epoch, logs=None):
        
        # keep track of where we are up to
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        # keep a history log of the learning rate and iterations (so we can 
        # plot it afterwards!)
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        # maintain the elements of the logs
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        # set the learning rate for the optimiser based on where we are in
        # the clr process
        K.set_value(self.model.optimizer.lr, self.clr())