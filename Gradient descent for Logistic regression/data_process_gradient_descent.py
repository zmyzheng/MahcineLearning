# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:54:36 2016

@author: Zmy-Apple
"""

import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


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

def draw_scatter(data, labels):
    index_list = []
    for i in  range(0,2):
        index_list.append([idx for idx, label in enumerate(labels) if labels[idx][0] == i])
    
    x,y,z = data[:,0],data[:,1],data[:,2]
    
    ax=plt.subplot(111,projection='3d')
    
    ax.scatter(x[index_list[0]], y[index_list[0]], z[index_list[0]],c='y') 
    ax.scatter(x[index_list[1]], y[index_list[1]], z[index_list[1]],c='g')
    ax.set_zlabel('Z') 
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

if __name__== "__main__":
    hw4data = loadmat('hw4data.mat')
    data = hw4data['data']
    labels = hw4data['labels']
    
    
    
    
    A = np.array([[0.05, 0, 0],[0, 1, 0], [0, 0, 0.05]])
    data2 = np.dot(data, A)
    
    train_number = data2.shape[0]
    dimension = data2.shape[1]
    
    ones = np.array([[1]] * train_number)
    
    data_modify = np.concatenate((data2, ones), axis=1)
    
    count = gradient_descent(data_modify, labels)
    print "number of itertions: ", count        

    print "the scatter plot for the original data is: "
    draw_scatter(data, labels)