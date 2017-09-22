import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from est_entro import est_entro_JVHW, est_entro_MLE
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from math import log, pi, exp
import lnn 

def lnn_samplor(length):
    tmpList = np.array(ITM_samplor(length)[0]).reshape(length,1)
    return tmpList

def my_function(x):
    if x<0 or x>1 :
        return 0
    elif x<1/4 or x>=3/4 :
        return 0
    elif x>=1/4 and x<1/2 :
        return 16*(x-1/4)
    elif x>=1/2 and x<3/4 :
        return -16*(x-3/4)

#Using acceptive-rejection method
def sampler():
    while True:
        X = random.uniform(0.0, 1.0)
        Y = random.uniform(0.0, 100.0)
        if Y < my_function(X):
            return X
        
def ARM_samplor(num, m=1):
    numList = []
    for _ in range(m):
        tmpList = []
        for i in range(num):
            tmp = sampler()
            tmpList.append(tmp)
        numList.append(tmpList)
        
    return numList

def itm(n=1000,c=0.5, a=0.25, h=4):
    # invPosArray_ufunc1 = np.frompyfunc(invPosArray, 4, 1)
    n = int(n)
    yb = np.random.rand(1, n)
    b = 2/h - a
    if ((b + c) >1):
        print("parameters out the defined domain")
        return
    # yb = np.array(yb)
    x = ((yb <= 0.0)*(c-a)+
    (yb > 0.0)*(yb < a*h/2)*((c-a) + (2*a*yb/h)**0.5)+
    (yb >= a*h/2)*(yb < 1.0)*((c+b) -(2*b*(1-yb)/h)**0.5)+
    (yb >= 1.0)*(c+b))
    return x

def ITM_samplor(num, m=1):
    numList = []
    for _ in range(m):
        numList.append(itm(num).tolist()[0])
        
    return numList


fig,ax = plt.subplots()
m = 100
length = [m]
true = 0.5*(1-4*np.log(2))/np.log(2)
errorSet = []

for _ in range(m):
    for l in length:
        data = lnn_samplor(l)
        H = lnn.entropy(data)
        err = np.sqrt(np.mean(np.square(H-true)))
        errorSet.append(err)
        

mc_times = 50
record_n = np.array([100])
record_h = [1/3]
num = len(record_h)

JVHW_err = np.zeros(num)
total_time = []
err = 0
errset1 = []
for order in range(m):
    for n in record_n:
        #tmpSamplor = ARM_samplor(n, m = mc_times) # ARM sampling is much slower than ITM
        tmpSamplor = np.array(ITM_samplor(n, m = mc_times)).reshape(n,1)
        for i, h in enumerate(record_h):
            S = int(1/h)
            edges = np.linspace(0,1,S+1)
            samp = np.digitize(tmpSamplor, edges)
            record_JVHW = est_entro_JVHW(samp) - np.log(S)
            JVHW_err[i] = np.sqrt(np.mean(np.square(record_JVHW - true)))
            errset1.append(np.mean(np.square(record_JVHW - true)))
            if (h == 1/3):
                err = JVHW_err[i]

errset1 = np.array(errset1).reshape(100,1)

fig,ax = plt.subplots()
ax.plot(range(m), errorSet, 's-', linewidth=1.5)
ax.plot(range(m), errset1, 's-', linewidth=1.5)
ax.legend(['lnn','SHVW'])
ax.set_title('n = 100, m = 100')
ax.set_ylabel('RMSE')
