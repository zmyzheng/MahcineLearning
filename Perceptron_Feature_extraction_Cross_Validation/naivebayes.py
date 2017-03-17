# -*- coding: utf-8 -*-

"""
Created on Sun Oct  9 17:00:01 2016
This code shows naive bayes
@author:Mingyang Zheng(mz2594)
"""

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold


#-----------------------------------------------------

if __name__== "__main__":
    reviews_tr = pd.read_csv("reviews_tr.csv")
    text_tr = reviews_tr["text"][:200000]
    label_tr = reviews_tr["label"][:200000]
    label_tr_matrix = (np.array([label_tr]).T - 0.5) * 2
    
  
    unigram_binary_vectorizer = CountVectorizer(min_df=1, binary=True)
    tr_unigram_binary_sparse_matrix = unigram_binary_vectorizer.fit_transform(text_tr)
    
    
    # 5-folds cross validation
    K = 5
    kf = KFold(text_tr.shape[0],n_folds=K)
    tr_idx = []
    te_idx = []
    for train_index, test_index in kf:
        tr_idx.append(train_index)
        te_idx.append(test_index)

    error_number5 = []
    for i in range(K):
        X_train, X_test = tr_unigram_binary_sparse_matrix[tr_idx[i]], tr_unigram_binary_sparse_matrix[te_idx[i]]
        Y_train, Y_test = label_tr[tr_idx[i]], label_tr[te_idx[i]]
      
        naive_bayes_classifier = BernoulliNB()
        naive_bayes_classifier.fit(X_train, Y_train)
        naive_bayes_result = naive_bayes_classifier.predict(X_test)
        compare_result = np.nonzero(naive_bayes_result - Y_test)
        error_number5.append(len(compare_result[0]))
    error_number_naive_bayes = np.mean(np.array(error_number5))
    error_rate_naive_bayes = error_number_naive_bayes /1.0 /Y_test.shape[0]
    
    print "error_rate_naive_bayes", error_rate_naive_bayes
    fp = open("error_rate_naive_bayes.txt", 'w')
    fp.write('error_rate_naive_bayes')
    fp.write('\n')
    fp.write(str(error_rate_naive_bayes))
    fp.close()
    
    
    
    
    
    
    
    
    
