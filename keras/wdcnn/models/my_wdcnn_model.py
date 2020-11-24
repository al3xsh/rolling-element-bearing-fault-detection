"""
my_wdcnn_model.py

from Zhang, et. al (2017) "A new deep learning model for fault diagnosis
with good anti-noise and domain adaptation ability on raw vibration signals"

note - 02/06/2020
i've added the stride of 16 in the first convolutional layer as per the paper
though if anything it hampers performance

author: alex shenfield
date:   14/04/2020
"""

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import MaxPooling1D

from keras.layers import Dropout


# build the wdcnn model
def generate_model(n_class, n_timesteps, n_variables, first_kernel=64):
    
    # set up the shape of the input
    ip = Input(shape=(n_timesteps, n_variables))

    # convolutional layers
    #y = Conv1D(16, (first_kernel), padding='same')(ip)
    y = Conv1D(16, (first_kernel), strides=16, padding='same')(ip)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(2, strides=2, padding='same')(y)
    
    y = Conv1D(32, (3), padding='same')(y)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(2, strides=2, padding='same')(y)
    
    y = Conv1D(64, (3), padding='same')(y)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(2, strides=2, padding='same')(y)
    
    y = Conv1D(64, (3), padding='same')(y)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(2, strides=2, padding='same')(y)
    
    y = Conv1D(64, (3), padding='same')(y)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(2, strides=2, padding='same')(y)
    
    # flatten
    y = Flatten()(y)
    
    # dense
    y = Dense(100)(y)
    y = BatchNormalization()(y)

    # add the softmax classification outpuy
    out = Dense(n_class, activation='softmax')(y)
    
    # join the input and the output and return the model
    model = Model(ip, out)
    return model
