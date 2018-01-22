#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:52:28 2018

@author: nick
"""

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures

X = pd.read_csv("train.csv")
y = X["count"]

dates = X["datetime"]
hours = [date.split()[1].split(":")[0] for date in dates]
weekday = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%w') for date in dates]

X["hour"] = hours
X["weekday"] = weekday

X_frame = X.drop(["datetime","casual", "registered", "count", "atemp", "holiday", "weather", "weekday", "season"], axis=1)

X = np.asarray(X_frame)
ylog = np.log1p(np.asarray(y))

regr = linear_model.LinearRegression()
#poly = PolynomialFeatures(degree=3)

kf = KFold(n_splits=10, shuffle= True)
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], ylog[train], ylog[test]  
    regr.fit(X_train, y_train) 
    y_pred = regr.predict(X_test)
    
    #print "feature importance",regr.feature_importances_
    print "mean squared error", mean_squared_error(y_test, y_pred)
    print "R²", r2_score(y_test, y_pred)

