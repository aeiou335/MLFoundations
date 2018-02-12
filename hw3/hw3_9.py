# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 17:18:08 2018

@author: kennylin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#learning_rate = 0.01
iterations = 2000
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def correctness(X, y, w):
    pred = sigmoid(np.dot(X, w))
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = -1
    return np.mean(pred != y)

def stochastic_gradient(x, y, w, lr, test_x, test_y):
    correct = []
    correct_out = []
    for it in range(iterations):
        i = it % 1000
        theta = sigmoid(np.dot(x[i:i+1,:], w) * y[i] * (-1))
        grad = np.dot((x[i:i+1,:] * y[i] * (-1)).T, theta)
        w = w - lr * grad
        correct.append(correctness(x, y, w))
        correct_out.append(correctness(test_x, test_y, w))
        '''
        if (it+1) % 1000 == 0:
            print(correctness(x, y, w))
        '''
    return w, correct, correct_out
def gradient(x, y, w, lr, test_x, test_y):    
    correct = []
    correct_out = []
    for it in range(iterations):
        theta = sigmoid(np.dot(x, w) * y * (-1))    
        grad = np.dot((x * y * (-1)).T, theta)/x.shape[0]
        w = w - lr * grad
        correct.append(correctness(x, y, w))
        correct_out.append(correctness(test_x, test_y, w))
        '''
        if (it+1) % 1000 == 0:
            print(correctness(x, y, w))
        '''
    return w, correct, correct_out
def read_csv(data):
    df = pd.read_csv(data, sep = '\s+', header = None)
    return df.as_matrix()

train_data = read_csv("hw3_train.dat.txt")
test_data = read_csv("hw3_test.dat.txt")

x, y = train_data[:,0:train_data.shape[1]-1], train_data[:,train_data.shape[1]-1:train_data.shape[1]]
test_x, test_y = test_data[:,0:test_data.shape[1]-1], test_data[:,test_data.shape[1]-1:test_data.shape[1]]

#Add bias
b =  np.ones((x.shape[0],1))
x = np.concatenate((x, b), axis = 1)
b_test = np.ones((test_x.shape[0],1))
test_x = np.concatenate((test_x, b_test), axis = 1)

#Initial weight
init_w = np.zeros((x.shape[1],1))

gd_w_01, g_19_1, g_19_1_out = gradient(x, y, init_w, 0.01, test_x, test_y)
print('E_in =', correctness(x, y, gd_w_01))
print('E_out =', correctness(test_x, test_y, gd_w_01))
sgd_w_01, sg_19_1, sg_19_1_out = stochastic_gradient(x, y, init_w, 0.01, test_x, test_y)
print('E_in =', correctness(x, y, sgd_w_01))
print('E_out =', correctness(test_x, test_y, sgd_w_01))
gd_w_001, g_19_2, g_19_2_out = gradient(x, y, init_w, 0.001, test_x, test_y)
print('E_in =', correctness(x, y, gd_w_001))
print('E_out =', correctness(test_x, test_y, gd_w_001))
sgd_w_001, sg_19_2, sg_19_2_out = stochastic_gradient(x, y, init_w, 0.001, test_x, test_y)
print('E_in =', correctness(x, y, sgd_w_001))
print('E_out =', correctness(test_x, test_y, sgd_w_001))

x = np.linspace(0,2000, 2000)
plt.plot(x, g_19_1, 'r')
plt.plot(x, sg_19_1, 'b--')
plt.legend(['Gradient Descent', 'Stochastic Gradient Descent'])
plt.title('lr = 0.01, E_in')
plt.xlabel('Iterations')
plt.ylabel('E_in')
plt.show()
plt.savefig('HW3_8_1.png')

x = np.linspace(0,2000, 2000)
plt.plot(x, g_19_2, 'r')
plt.plot(x, sg_19_2, 'b--')
plt.legend(['Gradient Descent', 'Stochastic Gradient Descent'])
plt.title('lr = 0.001, E_in')
plt.xlabel('Iterations')
plt.ylabel('E_in')
plt.show()
plt.savefig('HW3_8_2.png')

x = np.linspace(0,2000, 2000)
plt.plot(x, g_19_1_out, 'r')
plt.plot(x, sg_19_1_out, 'b--')
plt.legend(['Gradient Descent', 'Stochastic Gradient Descent'])
plt.title('lr = 0.01, E_out')
plt.xlabel('Iterations')
plt.ylabel('E_out')
plt.show()
plt.savefig('HW3_9_1.png')

x = np.linspace(0,2000, 2000)
plt.plot(x, g_19_2_out, 'r')
plt.plot(x, sg_19_2_out, 'b--')
plt.legend(['Gradient Descent', 'Stochastic Gradient Descent'])
plt.title('lr = 0.001, E_out')
plt.xlabel('Iterations')
plt.ylabel('E_out')
plt.show()
plt.savefig('HW3_9_2.png')