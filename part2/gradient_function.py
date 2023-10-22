import scipy.io as sio
from math import log
from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid 
from functions import add_zero_feature, pack_params
from sigmoid_gradient import sigmoid_gradient
def gradient_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    Theta1 = nn_params[ :(hidden_layer_size * (input_layer_size + 1))].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape((num_labels, (hidden_layer_size + 1)))
    
    m = X.shape[0]
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = add_zero_feature(a2)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    DELTA_3 = a3 - Y
    DELTA_2 = np.multiply(np.dot(DELTA_3,Theta2)[:, 1:] , sigmoid_gradient(z2))
    
    Theta1_grad = np.dot(DELTA_2.T, a1)/m
    Theta2_grad = np.dot(DELTA_3.T, a2)/m

    Theta1_grad[:, 1:] += (lambda_coef/m)*Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_coef/m)*Theta2[:, 1:]
 
    
    res = pack_params(Theta1_grad, Theta2_grad)
    #print(len(res))
    return res
    
