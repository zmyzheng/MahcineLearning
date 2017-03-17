# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:54:36 2016

@author: Zmy-Apple
"""


import numpy as np
from scipy.io import loadmat
import math





def objective_value(bata, train_data, train_labels):
    train_number = train_data.shape[0]
    inner_product = np.dot(train_data, bata)
    objective_matrix = np.log(np.exp(inner_product) + np.array([[1]] * train_number)) - train_labels * inner_product
    return np.sum(objective_matrix) / train_number

def gradient(bata, train_data, train_labels):
    train_number = train_data.shape[0]
    inner_product = np.dot(train_data, bata)
    exp = np.exp(inner_product)
    return (np.sum((exp / (exp + np.array([[1]] * train_number)) - train_labels) * train_data, axis=0) / train_number).reshape(1,train_data.shape[1])

def line_search(grad, bata, train_data, train_labels):
    eta = 1
    while objective_value(bata-eta*grad.T, train_data, train_labels) > objective_value(bata, train_data, train_labels) - 0.5 * eta * np.sum(np.square(grad)):
        eta = 0.5 * eta
    return eta

def error_rate_calculate(bata, holdout_data, holdout_labels):
    holdout_labels = (holdout_labels - 0.5)* 2
    return np.count_nonzero((np.sign(np.dot(holdout_data, bata)) - holdout_labels)) /1.0/ holdout_labels.shape[0]
        
    
    
def gradient_descent(train_data, train_labels, holdout_data, holdout_labels):
    demension_modify = train_data.shape[1]    
    bata = np.array([[0]] * demension_modify)
    count = 0
    power_of_two = 32
    best_error_rate = 1
    while True:
        grad = gradient(bata, train_data, train_labels)
        eta = line_search(grad, bata, train_data, train_labels)
        bata = bata - eta * grad.T
        count = count + 1
        if count == power_of_two:
            power_of_two = 2 * power_of_two
            holdout_error_rate = error_rate_calculate(bata, holdout_data, holdout_labels)
            if holdout_error_rate > 0.99 * best_error_rate:
                break
            else:
                best_error_rate = holdout_error_rate
    print "hold out error rate: ", best_error_rate    
    print "objective value: ", objective_value(bata, train_data, train_labels)
    return count
 

if __name__== "__main__":   
    hw4data = loadmat('hw4data.mat')
    
    data = hw4data['data']
    labels = hw4data['labels']
    
    data_number = data.shape[0]
    dimension = data.shape[1]
    
    ones = np.array([[1]] * data_number)
    
    data_modify = np.concatenate((data, ones), axis=1)
    
    train_data = data_modify[:int(math.floor(data_number*0.8)),:]
    holdout_data = data_modify[int(math.floor(data_number*0.8)+1):,:]
    train_labels = labels[:int(math.floor(data_number*0.8)),:]
    holdout_labels = labels[int(math.floor(data_number*0.8)+1):,:]
    count = gradient_descent(train_data, train_labels, holdout_data, holdout_labels)
    print "number of itertions: ", count      
    
