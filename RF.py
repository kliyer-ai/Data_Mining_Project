#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:46:50 2018

@author: nick
"""

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime
import seaborn as sn

def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))



X = pd.read_csv("train.csv")
y = X["count"]


dates = X["datetime"]
hours = [date.split()[1].split(":")[0] for date in dates]
weekday = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%w') for date in dates]
month = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%m') for date in dates]

X["hour"] = hours
X["weekday"] = weekday
X["month"] = month

print X.dtypes

categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]
for var in categoryVariableList:
    X[var] = X[var].astype("category")

print X.dtypes

plt.figure()
sn.distplot(np.log(X["count"]))
plt.ylabel("density")
plt.show()

plt.figure()
sn.distplot(X["count"])
plt.ylabel("density")
plt.show()

ax = sn.boxplot(np.log(X["count"]), orient="v")
#ax.set_ylim(-100,)  







"""
X_frame = X.drop(["datetime","casual", "registered", "count", "atemp", "holiday", "weather", "weekday"], axis=1)

X = np.asarray(X_frame)
y = np.asarray(y)

regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2)
kf = KFold(n_splits=10, shuffle= True)
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]    
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    
    print "feature importance",regr.feature_importances_
    print "score",regr.score(X_test, y_test)
    print "mean squared error", mean_squared_error(y_test, y_pred)
    print "R²", r2_score(y_test, y_pred)
    
"""