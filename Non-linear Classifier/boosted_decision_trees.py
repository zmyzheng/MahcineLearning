#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 01:15:00 2016

@author: Zmy-Apple
"""



import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import GradientBoostingClassifier





if __name__== "__main__":
    reviews_tr = pd.read_csv("reviews_tr.csv")
    text_tr = reviews_tr["text"][:200000]
    label_tr = reviews_tr["label"][:200000]
    
    reviews_te = pd.read_csv("reviews_te.csv")
    text_te = reviews_te["text"][:2000]
    label_te = reviews_te["label"][:2000]
    
    # unigram presentation
    unigram_vectorizer = CountVectorizer(min_df=1)
    tr_unigram_sparse_matrix = unigram_vectorizer.fit_transform(text_tr)
    te_unigram_sparse_matrix = unigram_vectorizer.transform(text_te)

    scaler = StandardScaler(with_mean=False)  
    scaler.fit(tr_unigram_sparse_matrix) 
    X_train = scaler.transform(tr_unigram_sparse_matrix)
    X_test = scaler.transform(te_unigram_sparse_matrix)  
    Y_train = label_tr
    Y_test = label_te
    
    
    
    clf = GradientBoostingClassifier()
   
    clf.fit(X_train, Y_train)
    
    
    
    Ytest_predict = clf.predict(X_test.toarray())
    
    Ytrain_predict = clf.predict(X_train.toarray())
    
    print np.count_nonzero(Y_test - Ytest_predict)
    print np.count_nonzero(Y_train - Ytrain_predict)
