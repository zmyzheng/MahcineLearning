# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:35:08 2016



@author: Mingyang Zheng (mz2594)
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def pre_processing(training_labels):
    '''
    collect the index of the training labels that traing_label[index] are the same into list
    '''
    index_list = []
    for i in range(10):
        index_list.append ( [idx for idx, label in enumerate(training_labels) if training_labels[idx][0]==i])
    return index_list
    
    
def collect_pic(training_data, index_list):
    '''
    collect the training data with same label
    '''
    collect = []    
    for i in range(10):
        arr2 = [training_data[x] for x in index_list[i]]
        collect.append(np.array( arr2))
    return collect   


def avg_pic(ocr):    
    '''
    calculate the mean of training data with the same label
    '''
    index_list = pre_processing(ocr['labels'])
    collect = collect_pic(ocr['data'].astype('float'), index_list)
    avg = []
    for i in range(10):
       avg.append( np.average(collect[i], axis=0))
    avg_array = np.array(avg)
    return avg_array
    

def distance_square_calculate(training_data, test_data):
    training_data_square = np.square(training_data) 
    training_data_square_sum = np.sum(training_data_square, axis=1)
    cross_term = np.dot(training_data, test_data.T)
    distance_square = training_data_square_sum - 2 * cross_term.T
    return distance_square
    
def NNC_1(training_data, training_label, test_data):
    '''
    1 Nearest neighbors
    
    note: when compare the Euclidean distance among different training data with the same test data,
    the square of test data is the same, so we didn't add this term into the distance_square
    '''
    distance_square = distance_square_calculate(training_data, test_data)
    min_value = np.argmin(distance_square, axis=1)
    return training_label[min_value]
  

def proptype_selection( collect, avg_array, n):
    '''
    choose the prototype, n is number of total prototypes 
    sort according to distance
    select samples with the same interval
    '''
    selected_data = [] 
    selected_data_format = []
    for i in range(10):
        distance_square = distance_square_calculate(collect[i], avg_array[i])
        idx_sorted = np.argsort(distance_square, axis=0)
        step = int(len(distance_square)/(n/10))
        selected_idx = idx_sorted[0:len(distance_square):step]
        selected_data.append(collect[i][selected_idx[0:(n/10)]])
    for i in range(10):
        for j in range(n/10):
            selected_data_format.append(list(selected_data[i][j]))
    return np.array(selected_data_format)



ocr = loadmat('ocr.mat')
n_list = [1000, 2000,4000,8000]
error_rate_list = []

for n in n_list:
    
    index_list = pre_processing(ocr['labels'])
    collect = collect_pic(ocr['data'].astype('float'), index_list)
    
    avg_array = avg_pic(ocr)
    avg_list = list(avg_array) 
    
    selected_data_format = proptype_selection( collect, avg_array, n)
    
    new_labels = np.array([[0], [1],[2], [3], [4], [5], [6], [7], [8], [9]]*(n/10))
    new_labels = new_labels.reshape(n/10, 10).T.reshape(n, 1)
    
    preds = NNC_1(selected_data_format.astype('float'), new_labels, ocr['testdata']) 
    dif = preds - ocr['testlabels']
    count = np.count_nonzero(dif)
    
    error_rate = count/10000.0 *100
    error_rate_list.append(error_rate)
    print error_rate

plt.errorbar(n_list, error_rate_list)
plt.axis([0,9000,5,13])
plt.xlabel("number of training data")
plt.ylabel("error rate(%)")
plt.title("learning curve")
plt.grid()
plt.show()

        
