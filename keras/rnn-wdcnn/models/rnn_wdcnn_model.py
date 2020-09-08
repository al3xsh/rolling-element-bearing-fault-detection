"""
rnn_wdcnn_model.py

this merges the idea of using a split convolutional and recurrent path from
lstm-fcm (karim, et. al. 2017) with a wide first convolutional layer to 
compress the time series signal (from Zhang, et al. 2017)

author: alex shenfield
date:   11/04/2020
"""

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import MaxPooling1D

from keras.layers import Permute
from keras.layers import Dropout

from keras.layers import concatenate

# import the supported models

from keras.layers import GRU
from keras.layers import LSTM

from .mylayers.attention_gru import AttentionGRU
from .mylayers.attention_lstm import AttentionLSTM


# build the attention based lstm-fcn model
def generate_model(n_class, n_timesteps, n_variables, 
                   rnn_type='lstm', ncells=16, rec_drop=0.1, first_kernel=64):
    
    # set up the shape of the input
    ip = Input(shape=(n_timesteps, n_variables))

    # feed to the attention based rnn layer via a wide 1d convolutional kernel
    # and apply dropout (at 50% - from the paper)
    
    # wide convolution as layer 1 on rnn pathway (same as fcn pathway)
    x = Conv1D(16, (first_kernel), padding='same')(ip)
    x = Permute((2, 1))(x)
    
    # select the type of rnn to use ...
    if rnn_type == 'agru':
        x = AttentionGRU(ncells, recurrent_dropout=rec_drop)(x)
    elif rnn_type == 'alstm':
        x = AttentionLSTM(ncells, recurrent_dropout=rec_drop)(x)
    elif rnn_type == 'gru':
        x = GRU(ncells, recurrent_dropout=rec_drop)(x)
    elif rnn_type == 'lstm':
        x = LSTM(ncells, recurrent_dropout=rec_drop)(x)
    else:
        raise ValueError('only (a)lstm or (a)gru are currently supported!')
        
    # apply (lots of) dropout
    x = Dropout(0.8)(x)
    #x = Dropout(0.5)(x)
    
    # fcnn pathway

    # convolutional layers
    y = Conv1D(16, (first_kernel), padding='same')(ip)
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

    # so, dropout is not mentioned in the wdcnn paper but seems to make a 
    # significant impact on accuracies when the signal is noisy ...
    y = Dropout(0.5)(y)
    
    #
    # i have added it to help with the noise rejection - we'll see what effect
    # it has on the load domain adaption stuff ...
    # 

    # concatenate the lstm path and the fcn path
    x = concatenate([x, y])

    # add the softmax classification output
    out = Dense(n_class, activation='softmax')(x)
    
    # join the input and the output and return the model
    model = Model(ip, out)
    return model
