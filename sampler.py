# -*- coding: utf-8 -*-
# @Date: 2017-09-15
# @Author: Chaoran Chen
# @Link: chenchaoran1234@gmail.com

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from est_entro import est_entro_JVHW, est_entro_MLE
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

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

mc_times = 50  # Total number of Monte-Carlo trials for each alphabet size
record_n = np.array([100000, 10000, 1000, 100]) # sample size 
#step length
record_h = np.array([1/1000000000, 1/100000000, 1/10000000, 1/1000000, 1/100000, 1/100000, 1/10000, 1/1000, 1/100, 1/50, 1/15, 1/5, 1/4, 1/3, 1/2, 1])
num = len(record_h)

true = 0.5*(1-4*np.log(2))/np.log(2)
JVHW_err = np.zeros(num)
fig,ax = plt.subplots()

for n in record_n:
    tmpSamplor = ARM_samplor(n, m = mc_times) # ARM sampling is much slower than ITM
    #tmpSamplor = ITM_samplor(n, m = mc_times)
    for i, h in enumerate(record_h):
        S = int(1/h)
        edges = np.linspace(0,1,S+1)
        samp = np.digitize(tmpSamplor, edges)
        record_JVHW = est_entro_JVHW(samp) - np.log(S)
        JVHW_err[i] = np.sqrt(np.mean(np.square(record_JVHW - true)))
        
    print('Calculation of n='+str(n)+' has finished. Time use:',elapsed)
    ax.plot(record_h, JVHW_err, 's-', linewidth=1.5)
    
plt.legend(['$n=100,000$','$n=10,000$','$n=1,000$','$n=100$'], loc='upper right')
plt.xscale('log')
ax.set_xlabel('h')
ax.set_ylabel('RMSE')
ax.set_ylim(-1,10)
ax.set_xticks([1/1000000000, 1/100000000, 1/10000000, 1/1000000, 1/100000, 1/100000, 1/10000, 1/1000, 1/100, 1/10, 1])
ax.set_title('Disfferential Entropy Estimation')
plt.savefig('entropy_estimation.png', dpi=300)
plt.show()
