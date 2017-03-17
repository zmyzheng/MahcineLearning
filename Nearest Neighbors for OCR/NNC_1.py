
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:43:40 2016



@author: Mingyang Zheng (mz2594)
"""
import random
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm



def NNC_1(training_data, training_label, test_data):
    '''
    1 Nearest neighbors
    
    note: when compare the Euclidean distance among different training data with the same test data,
    the square of test data is the same, so we didn't add this term into the distance_square
    
    '''
    training_data_square = np.square(training_data) 
    training_data_square_sum = np.sum(training_data_square, axis=1)
    cross_term = np.dot(training_data, test_data.T)
    distance_square = training_data_square_sum - 2 * cross_term.T
    min_value = np.argmin(distance_square, axis=1)
    return training_label[min_value]



ocr = loadmat('ocr.mat')

n_list = [1000, 2000,4000,8000]
avrg_list = []
stddeviation_list = []

for n in n_list:  
    s = []
    for i in range(10):
        sel = random.sample(xrange(60000), n)
        
        preds = NNC_1(ocr['data'][sel].astype('float'), ocr['labels'][sel], ocr['testdata'])    
        dif = preds - ocr['testlabels']
        count = np.count_nonzero(dif)
    
        error_rate = count/10000.0 *100
        
        s.append(error_rate)
        
    avrg_list.append(np.mean(s))
    print np.mean(s)
    stddeviation_list.append(np.std(s))
    
plt.errorbar(n_list, avrg_list, yerr=stddeviation_list)
plt.axis([0,9000,5,13])
plt.xlabel("number of training data")
plt.ylabel("error rate(%)")
plt.title("learning curve")
plt.grid()
plt.show()
    

    

        


