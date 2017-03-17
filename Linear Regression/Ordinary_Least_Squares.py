#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:58:17 2016

@author: Zmy-Apple
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars

if __name__== "__main__":
    housing = loadmat('housing.mat')
    data = housing["data"]
    labels = housing["labels"]
    testdata = housing["testdata"]
    testlabels = housing["testlabels"]
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(data, np.ravel(labels))
    
    Y_pred_original = reg.predict(testdata)
    avg_squ_loss_original = np.average(np.square(np.ravel(testlabels) - Y_pred_original))
    
    
    
    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3, fit_intercept=True)
    omp.fit(data, np.ravel(labels))
    coef_omp = omp.coef_
    intercept_omp = omp.intercept_
    Y_pred_omp = omp.predict(testdata)
    avg_squ_loss_omp = np.average(np.square(np.ravel(testlabels) - Y_pred_omp))
    
    
    las = Lasso(alpha=2.1)
    las.fit(data, np.ravel(labels))
    coef_las = las.coef_
    intercept_las = las.intercept_
    Y_pred_las = las.predict(testdata)
    avg_squ_loss_las = np.average(np.square(np.ravel(testlabels) - Y_pred_las))
    
    lar = Lasso(alpha=2.1)
    lar.fit(data, np.ravel(labels))
    coef_lar = lar.coef_
    intercept_lar = lar.intercept_
    Y_pred_lar = lar.predict(testdata)
    avg_squ_loss_lar = np.average(np.square(np.ravel(testlabels) - Y_pred_lar))
    