# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:55:41 2018

@author: kennylin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def correctness(X, y, w):
    pred = np.sign(np.dot(X, w))
    return np.mean(pred != y)
df = pd.read_csv("hw4_train.dat.txt", sep = ' ', header = None)
df_test = pd.read_csv("hw4_test.dat.txt", sep = ' ', header = None)

train_data = df.as_matrix()
test_data = df_test.as_matrix()

x_train = train_data[:,0:2]
y_train = train_data[:,2:3]
b = np.ones((x_train.shape[0],1))
x_train = np.concatenate((x_train, b), axis = 1)

x_test = test_data[:,0:2]
y_test = test_data[:,2:3]
b = np.ones((x_test.shape[0],1))
x_test = np.concatenate((x_test, b), axis = 1)

order = np.linspace(2,-10,13)
ein = []
eout = []
for i in order:

    l = 10**i
    
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train) + l * np.identity(x_train.shape[1])), x_train.T), y_train)
    
    E_in = correctness(x_train, y_train, w)
    E_out = correctness(x_test, y_test, w)
    ein.append(E_in)
    eout.append(E_out)
    print("Ein = ", E_in, " E_out = ", E_out, " Order = ", i)

plt.plot(order, ein, 'r')
plt.plot(order, eout, 'b--')
plt.legend(['E_in', 'E_out'])
plt.title('HW4_Q7')
plt.xlabel('log_10(lambda)')
plt.ylabel('Error')
plt.savefig('HW4_7.png')
plt.show()
