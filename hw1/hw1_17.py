# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 02:09:53 2017

@author: kennylin
"""

import numpy as np
total_updates = 0
learning_rate = 0.5
for j in range(2000):
    Y = []
    X = []
    with open('hw1.txt', 'r') as f:
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
    
    
    
    mistake = True
    update = 0
    w = np.zeros((X_train.shape[1],))
    while mistake:
        mistake = False
        for i in range(X_train.shape[0]):
            #print(np.dot(X_train[i], w.T) * Y_train[i])
            if np.dot(w, X_train[i]) * Y_train[i] <= 0:
                update += 1
                #print("Update on {} elements".format(i))
                mistake = True
                w = w + learning_rate * X_train[i] * Y_train[i]
    total_updates += update
    print("Total updates = {} in {} times".format(update, j))           
         
print("Averge updates = {}".format(total_updates/2000))