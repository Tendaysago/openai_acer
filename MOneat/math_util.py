"""Commonly used functions not available in the Python2 standard library."""
from __future__ import division

from math import sqrt, exp


def mean(values, d=None):
    values = list(values)
    return sum(map(float, values)) / len(values)

def momean(values,d):
    valuesmean=[]
    for _ in range(d):
        valuesmean.append(0.0)
    for l in range(len(values)):
        valuesmean = [ x + y for (x, y) in zip(valuesmean,values[l]) ]
    for idx in range(d):
        valuesmean[idx]/=len(values)
    return valuesmean


def llmean(values):
    ret=[]
    tmp = []
    for i in range(len(values[0])):
        tmp=0
        for j in range(len(values)):
            print(values[j][i])
            tmp += values[j][i]
        ret.append(tmp/len(values))
    return ret



def median(values):
    values = list(values)
    values.sort()
    return values[len(values) // 2]

def median2(values):
    values = list(values)
    n = len(values)
    if n <= 2:
        return mean(values)
    values.sort()
    if (n % 2) == 1:
        return values[n//2]
    i = n//2
    return (values[i - 1] + values[i])/2.0

def variance(values):
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)

def movariance(values,d):
    m = momean(values,d)
    #print(values)
    for l in range(len(values)):
        valuesstdev = [ (x - y) ** 2 for (x, y) in zip(m,values[l]) ]
    #print(valuesstdev)
    for idx in range(d):
        valuesstdev[idx]=sqrt(valuesstdev[idx])
    #for idx in range(d):
    #    valuesmean[idx]/=len(values)
    return valuesstdev

def stdev(values,d=None):
    return sqrt(variance(values))

def mostdev(values,d):
    return movariance(values,d)


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..)."""
    e_values = list(map(exp, values))
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2, 'momean': momean}
