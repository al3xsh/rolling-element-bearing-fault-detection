"""
data_utils.py

some useful functions to manipulate signals such as adding gaussian white 
noise at a specified signal to noise ratio

author: alex shenfield
date:   16/04/2020
"""

import numpy as np


# create additive white gaussian noise and add to a signal
def awgn(signal, snr):
    
    # convert snr to linear scale
    snr = 10 ** (snr / 10.0)
    
    # measure the signal power
    signal_power = np.sum(np.absolute(signal) ** 2.0, axis=0) / signal.shape[0]
    
    # calculate the noise power needed to meet the specified snr and then 
    # create a noise signal
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power) * np.random.standard_normal(signal.shape)
    
    # add the noise to the signal and return
    return signal + noise


# test case
if __name__ == '__main__':
    
    from scipy import signal
    import matplotlib.pyplot as plt
    
    # generate a saw tooth signal
    t = np.linspace(0, 10, 101)
    s = signal.sawtooth(2 * np.pi * (1/2/np.pi) * t)
    
    # add some noise
    ns = awgn(s, -4)
    
    # plot
    plt.plot(t, s)
    plt.plot(t, ns)
    plt.show()
