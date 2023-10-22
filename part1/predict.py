import scipy.io as sio
import numpy as np
from sigmoid import sigmoid 

def predict(X, Theta1, Theta2):
    m,n = X.shape
    ones = np.ones([m, 1])
    def a_calc(ones, X, Theta):
        a = np.c_[ones, X]
        result = np.dot(a, Theta.T)
        result = sigmoid(result)
        return result

    result = a_calc(ones, X, Theta1)
    result = a_calc(ones, result, Theta2)
    max_dig = np.argmax(result, axis=1)
    max_dig = max_dig+1
    return max_dig