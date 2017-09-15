#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-03 16:18:12
# @Author  : Qimin Wang (wangqimin@tongji.edu.cn)
# @Link    : wangqimin@tongji.edu.cn
# @Version : 1.0

import numpy as np
import sympy
from sympy import Symbol, expand, integrate, log, symbols
from fractions import Fraction
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
# f, g, h = symbols('f g h', cls=Function)

def density(x, c=0.5, a=0.25, h=4):
    '''defaulted domian for x:[0 , 1]'''
    b = 2/h - a
    if ((b + c) >1):
        print("parameters out the defined domain")
        return
    if x < (c - a): r = 0.0
    elif x < c: r = (x - c + a) * h / a
    elif x < (c + b): r = -(x - c - b) * h / b
    else: r = 0.0
    return r

def plotDensity():
    density_ufunc1 = np.frompyfunc(density, 4, 1)
    x = np.linspace(0, 1, 1000)
    y2 = density_ufunc1(x, 0.5, 0.25, 4)
    y2 = y2.astype(np.float)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill(x, y2, 'b', alpha=0.3)
    ax.plot(x, y2, 'b', label='Density Curve', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(('0', '1/4', '1/2', '3/4', '1'))
    ax.set_yticks(np.linspace(0, 4, 9))
    plt.show()
    # fig.savefig("Density Curve.jpg",dpi=300)
    return

# plotDensity()

def possibility_int(xb, c=0.5, a=0.25, h=4):
    b = 2/h - a
    x = Symbol('x')
    f1 = (x - c + a) * h / a
    f2 = -(x - c - b) * h / b
    if ((c+b) >1):
        print("parameters out the defined domain") 
        return f
    if xb < (c - a):
        r = 0.0
    elif xb < c:
        r = integrate(f1,(x,c - a,xb))
    elif xb < (c + b):
        r = integrate(f2,(x,c,xb)) + integrate(f1,(x,c - a,c))
    else:
        r = integrate(f1,(x,c - a,c)) + integrate(f2,(x,c,c+b))
    return r

def plotPossibility():
    possibility_int_ufunc1 = np.frompyfunc(possibility_int, 4, 1)
    x = np.linspace(0, 1, 1000)
    y3 = possibility_int_ufunc1(x, 0.5, 0.25, 4)
    y3 = y3.astype(np.float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y3, 'r', label='Possibility Curve', alpha=0.7)
    ax.legend(loc='lower right')
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(('0', '1/4', '1/2', '3/4', '1'))
    
    plt.show()
    # fig.savefig("Possibility Curve.jpg",dpi=300)
    return

# plotPossibility()

def shannonEntropy_int(xb, c=0.5, a=0.25, h=4, tor=1e-7):
    b = 2/h - a
    x = Symbol('x')
    f0 = integrate(-x*log(x))
    t1 = (x - c + a) * h / a
    t2 = -(x - c - b) * h / b
    if ((c+b) >1):
        print("parameters out the defined domain") 
        return f
    if xb <= (c - a + tor): 
        r = 0.0
    elif xb < c:
        r = (f0.subs(x,t1.subs(x,xb)) - f0.subs(x,t1.subs(x,c - a + tor) )) /(h/a)
    elif xb <= (c + b - tor): 
        r = (f0.subs(x,t2.subs(x,xb)) - f0.subs(x,t2.subs(x,c)))/(-h/b) + \
        (f0.subs(x,t1.subs(x,c)) - f0.subs(x,t1.subs(x,c - a+ tor))) /(h/a)
    else: 
        r = (f0.subs(x,t2.subs(x,c + b - tor)) - f0.subs(x,t2.subs(x,c)))/(-h/b) + \
        (f0.subs(x,t1.subs(x,c)) - f0.subs(x,t1.subs(x,c - a + tor))) /(h/a)
    return r

def plotShannon():
    shannonEntropy_int_ufunc1 = np.frompyfunc(shannonEntropy_int, 5, 1)
    x = np.linspace(0,1,100)
    y4 = shannonEntropy_int_ufunc1(x, 0.5, 0.25, 4, 1e-9)
    y4 = y4.astype(np.float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y4, 'r', label='Shannon Entropy Curve', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(('0', '1/4', '1/2', '3/4', '1'))
    plt.show()
    # fig.savefig("Shannon Entropy Curve.jpg",dpi=300)
    return

# using Inverse Transform Method

# # Inverse function: c=0.5, a=0.25, h=4
def inversePossibility(yb,c=0.5, a=0.25, h=4):
    b = 2/h - a
    if ((b + c) >1):
        print("parameters out the defined domain")
        return
    y = Symbol('y')
    f1 = (c-a) + (2*a*y/h)**0.5
    f2 = (c+b) -(2*b*(1-y)/h)**0.5
    if yb <= 0.0:
        x = c-a
    elif yb < a*h/2:
        x = f1.subs(y,yb)
    elif yb < 1.0:
        x = f2.subs(y,yb)
    else:
        x = c+b
    return x

def plotInvPossibility():
    inversePossibility_ufunc1 = np.frompyfunc(inversePossibility, 4, 1)
    x = np.linspace(0,1, 1000)
    y5 = inversePossibility_ufunc1(x, 0.5, 0.25, 4)
    y5 = y5.astype(np.float)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y5, 'r', label='Inverse Possibility Curve', alpha=0.7)
    ax.legend(loc='lower right')
    ax.set_yticks(np.linspace(0.25,0.75,5))
    ax.set_yticklabels(('1/4','3/8','1/2','5/8','3/4'))
    plt.show()
    # fig.savefig("Inverse Possibility Curve.jpg", dpi=300)
    return

def invPosArray(yb,c=0.5, a=0.25, h=4):
    b = 2/h - a
    if ((b + c) >1):
        print("parameters out the defined domain")
        return
    yb = np.array(yb)
    d = ((yb <= 0.0)*(c-a)+
    (yb > 0.0)*(yb < a*h/2)*((c-a) + (2*a*yb/h)**0.5)+
    (yb >= a*h/2)*(yb < 1.0)*((c+b) -(2*b*(1-yb)/h)**0.5)+
    (yb >= 1.0)*(c+b))

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

    fig, axes = plt.subplots()
    sns.distplot(x, color="m", ax=axes)
    plt.setp(axes, yticks=[])
    plt.title("Iterations = " +str(n), loc='center')
    
    fig.savefig("C:/Users/DELL/Desktop/ITM" +str(n)+ ".png", dpi=300)
    plt.show()
    return x

x = itm(1e5)
f1 = open('data.pkl','wb')
pickle.dump(x, f1, True)
f1.close()
# f2 = file('temp.pkl', 'rb')
# y = pickle.load(f2)
# f2.close()  