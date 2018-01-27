#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:16:54 2018

@author: nick
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn import tree

X = pd.read_csv("train.csv")
y = X["count"]

dates = X["datetime"]
hours = [int(date.split()[1].split(":")[0]) for date in dates]
weekday = [int(datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%w')) for date in dates]
#month = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%m') for date in dates]

X["hour"] = hours
X["weekday"] = weekday
#X["month"] = month


X_frame = X.drop(["datetime","casual", "registered", "count", "atemp", "holiday", "windspeed", "weekday"], axis=1)

X = np.asarray(X_frame)
y = np.asarray(y)
ylog = np.log(y)

regr = tree.DecisionTreeRegressor(max_depth=10)
kf = KFold(n_splits=10, shuffle= True)

errors = []
r2s= []

for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], ylog[train], ylog[test]    
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    
    #print "feature importance",regr.feature_importances_
    #print "score",regr.score(X_test, y_test)
    errors.append(mean_squared_error(np.exp(y_test), np.exp(y_pred)))
    r2s.append(r2_score(np.exp(y_test), np.exp(y_pred)))

print "average SSE", np.mean(errors)
print "average R²", np.mean(r2s)