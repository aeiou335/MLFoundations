# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:58:27 2017

@author: kennylin
"""

import matplotlib.pyplot as plt
import numpy as np
# Create random data
def random_data():
    x = np.random.uniform(-1,1,20)
    y = np.sign(x)
    
    for i in range(20):
        prob = np.random.random_sample()
        if prob < 0.2:
            y[i] = -1 * y[i]
    return x, y
# Implement decision stump algorithm
def decision_stump(x, y, theta):
    theta = np.tile(theta, len(x))
    #case s = 1
    s = 1
    h = np.sign(x - theta + 1e-10)
    err = np.sum(y != h)
    #case s = -1
    h = -1 * np.sign(x - theta + 1e-10)
    err2 = np.sum(y != h)
    #compare which case provides better error
    if err2 < err:
         err = err2
         s = -1
    return s, err/20
# Iteration 1000 times
E_ins = []
E_outs = []
for i in range(1000):
    x, y = random_data()
    theta = np.median(x)
    s, E_in = decision_stump(x, y, theta)
    E_out = 0.5 + 0.3 * s * (abs(theta) - 1)
    E_ins.append(E_in)
    E_outs.append(E_out)
    if i % 100 == 0:
        print(i)
# Plot scatter plot
fig = plt.figure()
plt.scatter(E_ins, E_outs)
plt.xlabel('E_in')
plt.ylabel('E_out')
plt.title('HW2_Q8')
plt.show()

fig.savefig('HW2_Q8.png')
