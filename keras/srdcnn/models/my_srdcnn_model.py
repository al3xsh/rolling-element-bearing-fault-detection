"""
srdcnn_model.py

my implementation of the srdcnn model using inspiration from wavenet / tcns

author:   alex shenfield
date:     15/04/2020
"""

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Flatten

from keras.regularizers import l2

from keras import layers


# the gate structure activations that are also used in srdcnn
def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return layers.multiply([tanh_out, sigm_out])


# define the residual dilated convolutional block from Zhuang, et al. (2019)
def rdconv_block(x, s, i, nb_filters, kernel_size):
    
    # input tensor
    original_x = x
    
    # dilated convolution + wavenet activation (stealing the gate structure 
    # from lstm)
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='causal',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    x = wave_net_activation(conv)

    # dropout - not in the paper but (sometimes) works well    
    x = layers.SpatialDropout1D(0.05)(x)

    # residual connection
    res_x = Conv1D(nb_filters, 1, padding='same')(original_x)
    res_x = layers.add([res_x, x])
    
    return res_x


# build the srdcnn model
def generate_model(n_class, n_timesteps, n_variables):
    
    # set up the shape of the input
    ip = Input(shape=(n_timesteps, n_variables))

    # i'm not sure if the dilations should be [0, 1, 2, 3] or [1, 2, 3, 4]
    # but [1, 2, 3, 4] seems to perform better
    #dilations = [0, 1, 2, 3]
    dilations = [1, 2, 3, 4]

    # srdcnn path (from paper ...)
    y = rdconv_block(ip, 1, dilations[0], 32, 64)
    y = Activation('relu')(y)

    y = rdconv_block(y, 1, dilations[1], 32, 32)
    y = Activation('relu')(y)
        
    y = rdconv_block(y, 1, dilations[2], 64, 16)
    y = Activation('relu')(y)
    
    y = rdconv_block(y, 1, dilations[3], 64, 8)
    y = Activation('relu')(y)
    
    # flatten
    y = Flatten()(y)
    
    # dense
    y = Dense(100)(y)

    # add the softmax classification output
    out = Dense(n_class, activation='softmax')(y)
    
    # join the input and the output and return the model
    model = Model(ip, out)
    return model