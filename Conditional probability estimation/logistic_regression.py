#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:49:33 2016

@author: Zmy-Apple
"""


import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression

def objective_value(bata, data_modify, labels):
    train_number = data_modify.shape[0]
    inner_product = np.dot(data_modify, bata)
    objective_matrix = np.log(np.exp(inner_product) + np.array([[1]] * train_number)) - labels * inner_product
    return np.sum(objective_matrix) / train_number


if __name__== "__main__":
    dataset = loadmat('hw5data.mat')
    data = dataset["data"]
    labels = dataset["labels"]
    testdata = dataset["testdata"]
    testlabels = dataset["testlabels"]
    
    
    clf = LogisticRegression()
    clf.fit(data, np.ravel(labels))
    
    
    train_number = data.shape[0]
    dimension = data.shape[1]
    ones = np.array([[1]] * train_number)
    data_modify = np.concatenate((ones, data), axis=1)
    bata = np.concatenate((clf.intercept_.reshape(1, 1), clf.coef_.reshape(dimension, 1)), axis=0)
    print "objective_value is : ", objective_value(bata, data_modify, labels)
    
    
    prob_pred = clf.predict_proba(testdata)[:,1][:1024]
    testlabels = testlabels.reshape(128, 1024)
    prob_true = np.average(testlabels, axis=0) 
    mae = np.average(np.absolute(prob_pred - prob_true))
    print "MAE is : ", mae
    
    