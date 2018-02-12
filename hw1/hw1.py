# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:35:47 2017

@author: kennylin
"""
import numpy as np
Y = []
X = []
with open('hw1.txt', 'r') as f:
    for row in f:
        l = row.split()
        if len(l) == 5:
            Y.append(float(l[4]))
            for i in range(4):
                X.append(float(l[i]))

X_train = np.array(X)
X_train = X_train.reshape(-1,4)
Y_train = np.array(Y)
Y_train = Y_train.reshape(-1,1)
b = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((X_train, b), axis = 1)
w = np.zeros((X_train.shape[1],))
mistake = True
update = 0

while mistake:
    mistake = False
    for i in range(X_train.shape[0]):
        #print(np.dot(X_train[i], w.T) * Y_train[i])
        if np.dot(w, X_train[i]) * Y_train[i] <= 0:
            update += 1
            print("Update on {} elements".format(i))
            mistake = True
            w = w + X_train[i] * Y_train[i]
    if not mistake:
        break
            
print("Total updates = {}".format(update))
    
    