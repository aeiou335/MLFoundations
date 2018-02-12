# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:27:09 2018

@author: kennylin
"""
import matplotlib.pyplot as plt
import numpy as np
data_num = 1000
#Create Data
def random_data():
    x_1 = np.random.uniform(-1,1,data_num).reshape(-1,1)
    x_2 = np.random.uniform(-1,1,data_num).reshape(-1,1)
    y = np.sign(np.power(x_1,2) + np.power(x_2,2) - 0.6)
    b = np.ones((data_num,1))

    x = np.concatenate(([b,x_1,x_2,x_1*x_2,x_1**2,x_2**2]), axis = 1)

    for i in range(data_num):
        prob = np.random.random_sample()
        if prob < 0.1:
            y[i] = -1 * y[i]
    return x, y.reshape(-1,1)
total_eout = []
total = 0
for i in range(1000):
    x, y = random_data()
    x_test, y_test = random_data()
    w = np.dot(np.linalg.pinv(x), y)
    
    pred = np.sign(np.dot(x_test, w))
    E_out = np.mean(pred != y_test)
    total_eout.append(E_out)
    total += E_out
    
plt.hist(total_eout,40)
plt.xlabel('E_out')
plt.ylabel('Frequency')
plt.title('HW3_Q7')
plt.show()
plt.savefig('HW3_Q7.png')

print('E_out=', total/1000)

