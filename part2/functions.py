import math
import matplotlib.pyplot as plt
from scipy.special import expit
import scipy.io as sio
import numpy as np

def add_zero_feature(X, axis=1):
    return np.append(np.ones((X.shape[0], 1) if axis else (1, X.shape[1])), X, axis=axis)
def decode_y(x):
        y = np.zeros([5000, 10])
        for i in range(x.shape[0]):
            y[i][x[i]-1] = 1
        return y
def pack_params(Theta1, Theta2):
    return np.concatenate((Theta1.ravel(), Theta2.ravel()))
def unpack_params(res, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = res[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = res[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))
    return [Theta1, Theta2]
def rand_initialize_weights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init