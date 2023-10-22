import scipy.io as sio
import numpy as np
from scipy.optimize import minimize
def test(Theta1, Theta2, X, Y):

    #Theta1 = nn_params[0, :hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    #Theta2 = nn_params[0, hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))

    m = Y.shape[1]
    Y = Y.A

    A_1 = X
    Z_2 = Theta1*A_1.T
    A_2 = sigmoid(Z_2)
    A_2 = add_zero_feature(A_2, axis=0)
    Z_3 = Theta2*A_2
    A_3 = sigmoid(Z_3)
    H = A_3.A

    J = np.sum(-Y*np.log(H) - (1-Y)*np.log(1-H))/m

    reg_J = 0.0
    reg_J += np.sum(np.power(Theta1, 2)[:, 1:])
    reg_J += np.sum(np.power(Theta2, 2)[:, 1:])

    J += reg_J*(float(lambda_coef)/(2*m))

    print(J) 