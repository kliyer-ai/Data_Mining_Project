#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:52:28 2018

@author: nick
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

X = pd.read_csv("train.csv")
y = X["count"]

dates = X["datetime"]
hours = [date.split()[1].split(":")[0] for date in dates]
weekday = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%w') for date in dates]
#month = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%m') for date in dates]

X["hour"] = hours
X["weekday"] = weekday
#X["month"] = month

X_frame = X.drop(["datetime","casual", "registered", "count", "atemp", "holiday", "windspeed", "weekday"], axis=1)

X = np.asarray(X_frame)
y = np.asarray(y)
ylog = np.log1p(y)

regr = linear_model.LinearRegression()
#poly = PolynomialFeatures(degree=3)

kf = KFold(n_splits=10, shuffle= True)
errors = []
r2s= []
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]  
    regr.fit(X_train, y_train) 
    y_pred = regr.predict(X_test)
    
    #print "feature importance",regr.feature_importances_
    errors.append(mean_squared_error((y_test), (y_pred)))
    r2s.append(r2_score((y_test), (y_pred)))

print "average SSE", np.mean(errors)
print "average R²", np.mean(r2s)


