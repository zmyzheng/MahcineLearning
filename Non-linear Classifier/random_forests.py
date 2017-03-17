#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:24:08 2016

@author: Zmy-Apple
"""


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV




if __name__== "__main__":
    reviews_tr = pd.read_csv("reviews_tr.csv")
    text_tr = reviews_tr["text"][:200000]
    label_tr = reviews_tr["label"][:200000]
    
    reviews_te = pd.read_csv("reviews_te.csv")
    text_te = reviews_te["text"][:200000]
    label_te = reviews_te["label"][:200000]
    
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
    
    
    parameters = {'n_estimators':[10, 50,100], 'max_features':['auto', 'log2'], 'min_samples_leaf':[1, 2, 4]}
    clf = RandomForestClassifier()
    clf2 = GridSearchCV(clf, parameters)
    clf2.fit(X_train, Y_train)
    print clf2.best_params_
    
    
    Ytest_predict = clf2.predict(X_test)
    
    Ytrain_predict = clf2.predict(X_train)
    
    print np.count_nonzero(Y_test - Ytest_predict)
    print np.count_nonzero(Y_train - Ytrain_predict)
