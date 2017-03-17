# -*- coding: utf-8 -*-

"""
Created on Sun Oct  9 17:00:01 2016

This code shows unigram presentation and Averaged Perceptron
@author: Mingyang Zheng (mz2594)
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold

def averagedPerceptron(data, labels):
    '''
    This function realize averaged Perceptron
    '''
    word_number = data.shape[1]
    train_number = data.shape[0]
    ones = np.array([[1]*train_number]).T
    data = hstack([data, ones], format='csr')
    word_number = data.shape[1]
    
    w = csr_matrix((1, word_number))
    
    data_labels = hstack([data, labels], format='csr')
    data_labels = shuffle(data_labels, random_state=0)
    data = data_labels[:,:word_number]
    labels = data_labels[:, word_number]
    
    for t in range(train_number):
        if (data.getrow(t).dot(w.T) * labels[t])[0,0] <= 0:
            w = w + labels[t] * data.getrow(t)
        else:
            w = w
    
    w_sum = w #calculate the sum of the last train_number+1 w
    
    data_labels = hstack([data, labels], format='csr')
    data_labels = shuffle(data_labels, random_state=0)
    data = data_labels[:,:word_number]
    labels = data_labels[:, word_number]
    
    for t in range(train_number):
        if (data.getrow(t).dot(w.T) * labels[t])[0,0] <= 0:
            w = w + labels[t] * data.getrow(t)
        else:
            w = w
        w_sum = w_sum + w
     
    return w_sum/1.0/(train_number+1)



def linearClassfier(w,testdata):
    '''
    This function realize linear classifier
    '''
    return testdata.dot(w.T)
    
#-----------------------------------------------------

if __name__== "__main__":
    reviews_tr = pd.read_csv("reviews_tr.csv")
    text_tr = reviews_tr["text"][:200000]
    label_tr = reviews_tr["label"][:200000]
    label_tr_matrix = (np.array([label_tr]).T - 0.5) * 2
    
    # unigram presentation
    unigram_vectorizer = CountVectorizer(min_df=1)
    tr_unigram_sparse_matrix = unigram_vectorizer.fit_transform(text_tr)
    
    # 5-folds cross validation
    K = 5
    kf = KFold(text_tr.shape[0],n_folds=K)
    tr_idx = []
    te_idx = []
    for train_index, test_index in kf:
        tr_idx.append(train_index)
        te_idx.append(test_index)
        
        
    error_number1 = []
    for i in range(K):
        X_train, X_test = tr_unigram_sparse_matrix[tr_idx[i]], tr_unigram_sparse_matrix[te_idx[i]]
        Y_train, Y_test = label_tr_matrix[tr_idx[i]], label_tr_matrix[te_idx[i]]
        w = averagedPerceptron(X_train, Y_train)
        
        ones = np.array([[1]*X_test.shape[0]]).T
        X_test = hstack([X_test, ones], format='csr')
        linear_result = linearClassfier(w, X_test)
        compare_result = len((linear_result.sign() - Y_test).nonzero()[0])
        error_number1.append(compare_result)
    error_number_linear_unigram = np.mean(np.array(error_number1))
    error_rate_linear_unigram = error_number_linear_unigram /1.0 /Y_test.shape[0]
    print "error_rate_linear_unigram", error_rate_linear_unigram
    fp = open("error_rate_linear_unigram.txt", 'w')
    fp.write('error_rate_linear_unigram')
    fp.write('\n')
    fp.write(str(error_rate_linear_unigram))
    fp.close()
    
    
    
