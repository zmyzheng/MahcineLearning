# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:54:36 2016



@author: Mingyang Zheng (mz2594)
"""


import numpy as np
from scipy.io import loadmat



def objective_value(bata, data_modify, labels):
    train_number = data_modify.shape[0]
    inner_product = np.dot(data_modify, bata)
    objective_matrix = np.log(np.exp(inner_product) + np.array([[1]] * train_number)) - labels * inner_product
    return np.sum(objective_matrix) / train_number

def gradient(bata, data_modify, labels):
    train_number = data_modify.shape[0]
    inner_product = np.dot(data_modify, bata)
    exp = np.exp(inner_product)
    return (np.sum((exp / (exp + np.array([[1]] * train_number)) - labels) * data_modify, axis=0) / train_number).reshape(1,data_modify.shape[1])

def line_search(grad, bata, data_modify, labels):
    eta = 1
    while objective_value(bata-eta*grad.T, data_modify, labels) > objective_value(bata, data_modify, labels) - 0.5 * eta * np.sum(np.square(grad)):
        eta = 0.5 * eta
    return eta
        
    
    
def gradient_descent(data_modify, labels):
    demension_modify = data_modify.shape[1]    
    bata = np.array([[0]] * demension_modify)
    count = 0
    while objective_value(bata, data_modify, labels) > 0.65064:
        grad = gradient(bata, data_modify, labels)
        eta = line_search(grad, bata, data_modify, labels)
        bata = bata - eta * grad.T
        count = count + 1
    print "final objective value: " , objective_value(bata, data_modify, labels)
    return count

if __name__== "__main__":    
    hw4data = loadmat('hw4data.mat')
    
    data = hw4data['data']
    labels = hw4data['labels']
    
    train_number = data.shape[0]
    dimension = data.shape[1]
    
    ones = np.array([[1]] * train_number)
    
    data_modify = np.concatenate((data, ones), axis=1)
    
    count = gradient_descent(data_modify, labels)
    print "number of itertions: ", count        

