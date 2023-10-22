import math  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sympy as sy
from random import randrange
from sympy import *
import scipy.integrate as integrate
import scipy.special as special

def norm_rand(mat, ot):
    s=2
    # Генерация двух равномерно распределённых незваисимых случайных величин
    while s>1 or s==0:
        rand1 = round(random.uniform(-1, 1),2)
        rand2 = round(random.uniform(-1, 1),2)
        s = rand1**2+rand2**2  
    # Преобразование независимых случайных величин, равномерно наспределённые на отрезке [-1, 1],
        # в независимые величины, удовлетворяющие стандартному нормальному распределению методом Бокса — Мюллера
    nrand1 = rand1*math.sqrt((-2*math.log(s))/s)
    nrand2 = rand2*math.sqrt((-2*math.log(s))/s) 
    nrand1 = mat+ot*nrand1
    nrand2 = mat+ot*nrand2
    return [nrand1, nrand2] 

R = [] 
N = []
x = []
y = []
mat = 0
ot = 1
a = -5
b = 8
q = 1.4
a1 =-5
num = 100
n_mas_rand = []
yrand = []
mat_t1 = 0
mat_t2 = 20
F = []

mat = np.linspace(mat_t1, mat_t2, num = 10000)
for i in range(10000):
    nrand1, nrand2 = norm_rand(mat[i], ot)
    n_mas_rand.append(nrand1)
    n_mas_rand.append(nrand2)
mat = 0
c = a
for i in range(num):
    R.append(a+((math.fabs(a)+math.fabs(b))/num))
    a = R[i]
x = R
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ax.hist(n_mas_rand,100)
ax.grid()
plt.show()

def polynomial(x):
    return math.e**(-((x)**2)/2)
plambda = ((2*ot)**-1)*(mat_t2 -mat_t1)
t = (10-mat)/(2*plambda*ot)
w = (1+(plambda**2)/3)**(1/2)
for i in range(num):
    o, p = integrate.quad(polynomial, 0,x[i])
    Fi = 1/(math.sqrt(2*math.pi))*(o-p)
    z = (x[i]-mat_t1+plambda*ot)/(ot*w)
    fi = (w/(2*plambda))*(Fi*(z*w+plambda)-Fi*(z*w-plambda))
    F.append((1/2)*(1+(1/plambda)*((z*w+plambda)*Fi*(z*w+plambda)-(z*w-plambda)*Fi*(z*w-plambda)+fi*(z*w+plambda)-fi*(z*w-plambda))))
plt.plot(x,F, 'r-')
plt.show()