# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 02:15:05 2017

@author: kennylin
"""

import numpy as np
from random import shuffle
total_updates = 0
total_error = 0
for j in range(2000):
    Y = []
    X = []
    with open('hw1_18.txt', 'r') as f:
        for row in f:
            l = row.split()
            if len(l) == 5:
                #Y.append(float(l[4]))
                for i in range(5):
                    X.append(float(l[i]))
   
    X_t = np.array(X)
    X_t = X_t.reshape(-1,5)
    np.random.shuffle(X_t)
    X_train = X_t[:,0:4]
    Y_train = X_t[:,4]
    Y_train = Y_train.reshape(-1,1)
    b = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((X_train, b), axis = 1)
    w = np.zeros((X_train.shape[1],))
    w_hat = np.zeros((X_train.shape[1],))
    
    err = []
    mistake = True
    update = 0
    w = np.zeros((X_train.shape[1],))
    for t in range(50):
        for i in range(X_train.shape[0]):
            #print(np.dot(X_train[i], w.T) * Y_train[i])
            if np.dot(w, X_train[i]) * Y_train[i] <= 0:
                err.append(i)
        shuffle(err)
        rand_mistake = err[0]
        w = w + X_train[rand_mistake] * Y_train[rand_mistake]
        w_error = 0
        w_hat_error = 0
        for k in range(X_train.shape[0]):
            if np.dot(w, X_train[k]) * Y_train[k] <= 0:
                w_error += 1
        for l in range(X_train.shape[0]):
            if np.dot(w_hat, X_train[l]) * Y_train[l] <= 0:
                w_hat_error += 1
        if w_error < w_hat_error:
            w_hat = w
    X_test = []
    Y_test = []
    with open('hw1_18_test.txt', 'r') as f:
        for row in f:
            l = row.split()
            if len(l) == 5:
                Y_test.append(float(l[4]))
                for i in range(4):
                    X_test.append(float(l[i]))         
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1,4)
    Y_test = np.array(Y_test)
    Y_test = Y_test.reshape(-1,1)
    b = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((X_test, b), axis = 1)
    error = 0
    for i in range(X_test.shape[0]):
        if np.dot(w, X_test[i]) * Y_test[i] <= 0:
            error += 1
    total_error += error / X_test.shape[0]
    print("Error is {} in {} iterations".format(error/X_test.shape[0], j))
print("Average error: {}".format(total_error/2000))