# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:23:34 2018

@author: user
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tfd = tf.contrib.distributions
sigma = tf.constant([1,1], dtype=tf.float32)
XY_generator = tfd.MultivariateNormalDiag(tf.ones([2]), sigma)
XY_gen = np.random.exponential(scale=[1,1])

w = tf.placeholder(tf.float32, shape=[1], name='w')
init = tf.global_variables_initializer()

lambda_1 = 1.
lambda_0 = 1.
T_0 = 1
T = 1000000.
time_limit = 100.
alpha = 1.67
m = ((alpha-1)/alpha)

def a(r):
    return  1 - lambda_1/(r**((alpha-1)/9))

waiting_times = []


def make_birth(t):
    x = np.random.exponential(scale=1)
    y = np.random.pareto(alpha)+m
    return [t, x, y]

def wait(current_lambda):
    return np.random.exponential(scale=1./current_lambda)

def population_alive(history, current_time):
    result = 0
    n = len(history)
    for j in range(n):
        time_since_birth = current_time - history[j][0]
        lifespan = history[j][2]
        size = history[j][1]
        if time_since_birth < lifespan:
            result += size
    return  result
        
    

history = []
intensities =[]
intensities.append([0, lambda_0])
current_time = 0
intensity = lambda_0
active_intensity = 0.

while current_time<time_limit:
    
    time_step = wait(intensity)
    if (current_time + time_step) > time_limit:
        break
    current_time += time_step
    newborn = make_birth(current_time + time_step)
    intensity = a(T) * (population_alive(history, current_time) + newborn[1]) + lambda_0  
    intensities.append([current_time, intensity])
    history.append(newborn)
    print('Time: ', current_time)
    print('Intensity: ', intensity)
    print()
    
intensities = np.transpose(intensities)
plt.plot(intensities[0], intensities[1])
plt.show()


def r_lambda(t):
    event_number = -1
    try:
        event_number = np.min(np.where(intensities[0] > T*t )) - 1
    except ValueError:
        return 0
    return (1/(T)) * intensities[1][event_number]

upper_time_limit = time_limit/T
x = np.linspace(0, upper_time_limit, 200)
y = [r_lambda(z) for z in x]
plt.plot(x, y)
    
    
    
    
