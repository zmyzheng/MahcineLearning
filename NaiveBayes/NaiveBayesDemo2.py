# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:26:12 2016



@author: Mingyang Zheng (mz2594)
"""


import numpy as np
from scipy.io import loadmat


def getIndexList(labels):
    '''
    This method seperate the indexes of labels into 20 classes
    
    input: labels(sparce matrix)
    output: index_list(list): every element in index_list (namely, index_list[i]) is also a list
            that indicates the indexes with whom the elements in labels are i
    '''
    index_list = []
    for i in  range(1,21):
        index_list.append([idx for idx, label in enumerate(labels) if labels[idx][0] == i])
    return index_list
    

def getSubIndexList(labels):
    '''
    This method is similar to getIndexList(labels), but it is used to 
    deal with the modified labels
    '''
    index_list = []
    for i in  range(0,2):
        index_list.append([idx for idx, label in enumerate(labels) if labels[idx][0] == i])
    return index_list


def numberOfEachLabel(index_list):
    '''
    This method counts the numeber of elements in each index_list[i]
    In this Homework, it is used to calculate how many samples there are 
    in each group 
    '''
    length_list = []
    for i in range(2):
        length_list.append(len(index_list[i]))
    return length_list


def getMLEConditionalDistribution(index_list,data):
    '''
    This method is used to get conditional distribution by using MLE,
    in this homework, we actually use laplace smoothing, so the main function
    does not call this method
    '''
    u = []
    for i in range(2):
        submatrix = data[index_list[i]]
        u.append(np.array(submatrix.mean(axis=0))[0])
    return np.array(u)

def getLaplaceConditionalDistribution(index_list, data, length_list):
    '''
    This method is used to get the conditional distribution by using 
    Laplace smoothing
    The output of this method is a 2 dimensional array 
    Each element of this array is conditional distribution u(y,j)
    '''
    u = []
    for i in range(2):
        submatrix = data[index_list[i]]
        u.append((np.array(submatrix.sum(axis=0))[0] + 1) / (length_list[i]+2))
    return np.array(u)
        

def naiveBayesClassifier(priors, u, testdata):
    '''
    This method realise Naive Bayes Classifier
    priors are the prior probability of y, namely the pi(y) in professor's PPT
    priors are calculated in the main function
    u is the conditional distribution 
    
    The output max_value+1 is because the groups are from 1 to 20, while
    the indexes are from 0 to 19
    '''
    logu = np.log(u)
    log1_u = np.log(1-u)
    
    summ = testdata.dot(logu.T-log1_u.T)  + np.sum(log1_u.T, axis=0) + np.log(priors)
    max_value = np.argmax(summ, axis=1)    
    return max_value




if __name__== "__main__":
    news = loadmat('news.mat')
    dimension = news['data'].shape[1]
    number_of_data = news['data'].shape[0]
    labels = news['labels']
    data = news['data']
    testdata = news['testdata']
    testlabels = news['testlabels']
    
    index_list = getIndexList(labels)
    test_index_list = getIndexList(testlabels)
    
    # the code below is to modify the data and labels for religious topics
    # and political topics
    sub_index_list = np.array(index_list[0] + index_list[15] + index_list[19] + index_list[16] + index_list[17] + index_list[18] ) 
    sub_test_index_list = np.array(test_index_list[0] + test_index_list[15] + test_index_list[19] + test_index_list[16] + test_index_list[17] + test_index_list[18] ) 
    data = data[sub_index_list]
    testdata = testdata[sub_test_index_list]
    number_of_data = data.shape[0]
    labels = labels[sub_index_list]
    
    labels[0:len(index_list[0] + index_list[15] + index_list[19]),0] = 0
    labels[len(index_list[0] + index_list[15] + index_list[19]):,0] = 1
    
    testlabels = testlabels[sub_test_index_list]
    testlabels[:len(test_index_list[0] + test_index_list[15] + test_index_list[19]),0] = 0
    testlabels[len(test_index_list[0] + test_index_list[15] + test_index_list[19]):,0] = 1
    
    index_list = getSubIndexList(labels)
    
    length_list = numberOfEachLabel(index_list)
    
    
    priors = np.array(length_list) /1.0 / number_of_data
    #this calculate the prior probablity of y
              
              
    u = getLaplaceConditionalDistribution(index_list, data, length_list)


test_max_value = naiveBayesClassifier(priors, u, testdata)

test_dif = test_max_value - testlabels.T

test_count = np.count_nonzero(test_dif)

test_error_rate = test_count / 1.0 / len(test_max_value) * 100

train_max_value = naiveBayesClassifier(priors, u, data)

train_dif = train_max_value - labels.T

train_count = np.count_nonzero(train_dif)

train_error_rate = train_count / 1.0 / len(train_max_value) * 100
    
print "test error rate: ", test_error_rate,"%"
print "train error rate: ", train_error_rate,"%"



