'''
This is the code for creating a sampler from the certain function
edit by Jupytor notebook python3
'''

import random
import numpy as np  
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def my_function(x):
    if x<0 or x>1 :
        return 0
    elif x<1/4 or x>=3/4 :
        return 0
    elif x>=1/4 and x<1/2 :
        return 16*(x-1/4)
    elif x>=1/2 and x<3/4 :
        return -16*(x-3/4)

#Using an algorithm called acceptive-rejection method
def sampler():
    while True:
        X = random.uniform(0.0, 1.0)
        Y = random.uniform(0.0, 1000.0)
        if Y < my_function(X):
            return X
        
numList = []
for i in range(10000):
    tmp = sampler()
    numList.append(tmp)
    
x=np.arange(0,1,0.01)  
y=[]  
for i in x:  
    y_1=my_function(i)
    y.append(y_1)  
    
plt.plot(x,y)  
plt.xlabel("x")  
plt.ylabel("y")
plt.title("differential entropy")
sns.distplot(numList)
plt.legend(['f (x)','sample'])
plt.show() 
