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

x = train_data[:,0:2]
y = train_data[:,2:3]
b = np.ones((x.shape[0],1))
x = np.concatenate((x, b), axis = 1)
x_train = x[0:120,:]
x_valid = x[120:,:]
y_train = y[0:120,:]
y_valid = y[120:,:]

x_test = test_data[:,0:2]
y_test = test_data[:,2:3]
b = np.ones((x_test.shape[0],1))
x_test = np.concatenate((x_test, b), axis = 1)

order = np.linspace(2,-10,13)
ein = []
eout = []
evalid = []
for i in order:

    l = 10**i
    
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train) + l * np.identity(x_train.shape[1])), x_train.T), y_train)
    
    E_in = correctness(x_train, y_train, w)
    E_out = correctness(x_test, y_test, w)
    E_valid = correctness(x_valid, y_valid, w)
    ein.append(E_in)
    eout.append(E_out)
    evalid.append(E_valid)
    print("Ein = ", E_in, " E_out = ", E_out,"E_Valid = ", E_valid, " Order = ", i)

plt.plot(order, ein, 'r')
plt.plot(order, evalid, 'b--')
plt.legend(['E_in', 'E_valid'])
plt.title('HW4_Q8')
plt.xlabel('log_10(lambda)')
plt.ylabel('Error')
plt.savefig('HW4_8.png')
plt.show()
