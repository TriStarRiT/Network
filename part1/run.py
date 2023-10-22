import scipy.io as sio
from displayData import displayData
from sigmoid import sigmoid 
from predict import predict 
import numpy as np
import matplotlib.pyplot as plt

def out(filename, key):
    matrix =  filename.get(key)
    return matrix

#2
test_set = sio.loadmat("D:/GIT_folder/Numbers/part1/test_set.mat")
weights = sio.loadmat("D:/GIT_folder/Numbers/part1/weights.mat")
x = out(test_set, "X")
y = out(test_set, "y")
Theta1 = out(weights, "Theta1")
Theta2 = out(weights, "Theta2")
m,n = x.shape
#3
mas = np.random.permutation(m)[:100]
mat = []
for i in range (len(mas)):
    mat.append(x[mas[i]])
ma = np.array(mat)
displayData(ma)
#5
pred = predict(x, Theta1, Theta2)
#6
y = np.ravel(y)
r = pred == y
r = np.mean(np.double(r))
print(r*100,"%")
#8
err = np.where(pred != y.ravel())[0]
print(err)
mat = []
for i in range (100):
    mat.append(x[err[i]])
ma = np.array(mat)
displayData(ma)
#7
rp = np.random.permutation(m)
plt.figure()
for i in range(5):
    X2 = x[rp[i],:]
    X2 = np.matrix(x[rp[i]])
    pred = predict(X2.getA(), Theta1, Theta2)
    pred = np.squeeze(pred)
    pred_str = 'Neural Network Prediction: %d (digit %d)' % (pred, y[rp[i]])
    displayData(X2, pred_str)
    plt.close()