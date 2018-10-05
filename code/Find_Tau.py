
# coding: utf-8

# In[2]:


# used to find the average Tau. This code is not optimized and take ~45mins for 80,000 signals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
from numba import jit
import scipy.optimize as opt
from lmfit.models import GaussianModel, ConstantModel

import tables
import os


# In[3]:


h5file = tables.open_file("../data/test_input.h5", driver="H5FD_CORE")
print(h5file)
data = h5file.root.RawData
EventData = h5file.root.EventData
print(len(EventData))


# In[4]:


# Find pulses that triggered multiple times and delete them
i_pileup = 0
j_pileup = 0
pileup_args = np.array([])
EventDataArray = np.array(EventData)

for x in range(len(EventData)):
    if int(EventData[x][3])>1:
        pileup_args = np.append(pileup_args,i_pileup)
        j_pileup = j_pileup+1  
    i_pileup = i_pileup+1

print(data)
print(len(pileup_args))

data = np.delete(data[:], pileup_args, 0)
    
print('deleted ', len(pileup_args),' signals due to pile up')
#print(len(data2))


# In[8]:


@jit(parallel = True)
def baseLineReduction(mysignal):
    avgNoise = np.mean(mysignal[0:1000])
    return mysignal-avgNoise

@jit(parallel = True)
def func(x, a, b):
    return a * np.exp(-b * x)


# In[6]:


rows = len(data[:,0])
print(rows)
cols = len(data[0,:])
mysignals = np.zeros((rows,cols))

@jit(parallel = True)
def makesignals(raw_data):
    for j in range(0,rows):
        mysignals[j] = data[j,:]
    return mysignals

startTime = datetime.now()

mysignals = makesignals(data)

print(datetime.now() - startTime)


# In[9]:


startTime = datetime.now()
events = len(mysignals)
cols = len(data[0,:])
trap_out = np.zeros((events,cols))
tau = np.zeros(events)
for j in range(0,events):
    signalOnly = baseLineReduction(mysignals[j])
    popt, pcov = curve_fit(func, range(0,len(signalOnly)-1115), signalOnly[1115:])
    tau[j] = 1.0/popt[1]
print('Average Tau across all signals',np.mean(tau))
print(datetime.now() - startTime)

