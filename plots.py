#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:14:41 2018

@author: nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sn
import calendar


X = pd.read_csv("train.csv")
y = X["count"]


dates = X["datetime"]
hours = [int(date.split()[1].split(":")[0]) for date in dates]
weekday = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%A') for date in dates]
month = [datetime.strptime(date.split()[0], "%Y-%m-%d").strftime('%m') for date in dates]

X["hour"] = hours
X["weekday"] = weekday
X["month"] = month

#X["season"] = X.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
#sn.boxplot(data = X, x="season", y="count", orient="v")

#fig = plt.figure()
#fig.set_size_inches(10,5)
#X["month"] = X.month.apply(lambda m : calendar.month_name[int(m)])
#sn.boxplot(data = X, x="month", y="count", orient="v")

sn.boxplot(data = X, x="weather", y="count", orient="v")

wd = np.asarray(X["workingday"])
count = np.asarray(X["count"])
h = np.asarray(X["hour"])

#hourAggregated = pd.DataFrame(X.groupby(["hour","workingday"],sort=True)["count"].mean()).reset_index()
#sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["workingday"], data=hourAggregated, join=True)

#hourAggregated = pd.DataFrame(X.groupby(["hour","holiday"],sort=True)["count"].mean()).reset_index()
#sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["holiday"], data=hourAggregated, join=True)

#hueOrder = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday", "Sunday"]
#hourAggregated = pd.DataFrame(X.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
#sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True)

#hourAggregated = pd.DataFrame(X.groupby(["hour","holiday"],sort=True)["count"].mean()).reset_index()
#sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["holiday"], data=hourAggregated, join=True)
 

   
"""    
corrMatt = X[["temp", "atemp","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
"""


"""
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]
for var in categoryVariableList:
    X[var] = X[var].astype("category")


plt.figure()
sn.distplot(np.log(X["count"]))
plt.ylabel("density")
plt.show()

plt.figure()
sn.distplot(X["count"])
plt.ylabel("density")
plt.show()

plt.figure()
sn.boxplot(np.log(X["count"]), orient="v")
plt.show()
"""




