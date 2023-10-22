import scipy.io as sio
from math import log
from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
from functions import add_zero_feature, decode_y, pack_params,\
    rand_initialize_weights, unpack_params
from cost_function import cost_function
from sigmoid_gradient import sigmoid_gradient
from gradient_function import gradient_function
from sigmoid import sigmoid 

if __name__ == '__main__':

    def decode_y(x):
        y = np.zeros([5000, 10])
        for i in range(x.shape[0]):
            y[i][x[i]-1] = 1
        return y
    def out(filename, key):
        matrix =  filename.get(key)
        return matrix
    # Задание 1.
    # Загрузить обучающую выборку из файла training_set.mat в переменные X и y
    # Загрузить весовые коэффициенты из файла weights.mat в переменные Theta1 и Theta2
    # Использовать для этого функцию scipy.io.loadmat

    training_set = sio.loadmat("D:/GIT_folder/Numbers/part2/training_set.mat")
    weights = sio.loadmat("D:/GIT_folder/Numbers/part2/weights.mat")
    X = out(training_set, "X")
    y = out(training_set, "y")
    Theta1 = out(weights, "Theta1")
    Theta2 = out(weights, "Theta2")
    # Задание 2.
    # Программно определить параметры нейронной сети
    # input_layer_size = ...  # количество входов сети (20*20=400)
    # hidden_layer_size = ... # нейронов в скрытом слое (25)
    # num_labels = ...        # число распознаваемых классов (10)
    # m = ...                 # количество примеров (5000)
    m,n = X.shape
    input_layer_size = n
    m,n = Theta1.shape
    hidden_layer_size = m
    num_labels = len(np.unique(y))
    
    # добавление единичного столбца - нейрон смещения
    X = add_zero_feature(X)
    # декодирование вектора Y
    y = decode_y(y)

    # объединение матриц Theta в один большой массив
    nn_params = pack_params(Theta1, Theta2)

    # проверка функции стоимости для разных lambda
    lambda_coef = 0
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
            nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coef)))

    lambda_coef = 1
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
            nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coef)))
    
    # проверка производной sigmoid
    gradient = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Производная функции sigmoid в точках -1, -0.5, 0, 0.5, 1:')
    print(gradient)

    # случайная инициализация параметров
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = pack_params(initial_Theta1, initial_Theta2)
    print(len(initial_nn_params))
    # обучение нейронной сети
    res = minimize(cost_function, initial_nn_params, method='L-BFGS-B',
        jac=gradient_function, options={'maxiter': 100000},
        args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coef)).x

    # разбор вычисленных параметров на матрицы Theta1 и Theta2
    Theta1, Theta2 = unpack_params(
        res, input_layer_size, hidden_layer_size, num_labels)

    # выичисление отклика сети на примеры из обучающей выборки
    y = out(training_set, "y")
    h1 = sigmoid(np.dot(X, Theta1.T))
    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))
    y_pred = np.argmax(h2, axis=1) + 1
    print('Точность нейронной сети на обучающей выборке: {}'.format(
        np.mean(y_pred == y.ravel(), ) * 100))
        
