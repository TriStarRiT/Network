import scipy.io as sio
import numpy as np
from sigmoid import sigmoid 
from functions import add_zero_feature
def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coef):
    Theta1 = nn_params[ :(hidden_layer_size * (input_layer_size + 1))].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape((num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    a1=X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = add_zero_feature(a2)
    z3 = np.dot(a2, Theta2.T)
    h = sigmoid(z3)
    err = np.sum(-y*np.log(h) - (1-y)*np.log(1-h))/m
    Theta1 = np.square(Theta1)
    Theta2 = np.square(Theta2)
    reg = sum([sum(i) for i in Theta1]) + sum([sum(i) for i in Theta2])
    reg = reg * (lambda_coef/(2*m))
    result = reg + err
    return result