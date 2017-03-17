# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 23:10:36 2016



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
    
    log_priors = np.log(priors)
    logu = np.log(u)
    log1_u = np.log(1-u)
        
    alpha0 = log_priors[1] + np.sum(log1_u.T, axis=0)[1] - log_priors[0] - np.sum(log1_u.T, axis=0)[0]
    
    alphaj = (logu - log1_u)[1] - (logu - log1_u)[0]
    
    print "alpha0 is:", alpha0
    print "alphaj are:", alphaj    
    
    sorted_index = np.argsort(alphaj)
    max20index = sorted_index[-20:][ : :-1]
    min20index = sorted_index[:20]
    
    fp = open('news.vocab', 'r')
    
    vocab = np.array(fp.readlines())
    
    
    min20words = vocab[min20index]
    max20words = vocab[max20index]
    
    min20words = [ m.strip('\n') for m in min20words]
    max20words = [ m.strip('\n') for m in max20words]
    
    print "words with 20 smallest alpha values are:", min20words
    print "words with 20 largest alpha values are:", max20words