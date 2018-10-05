
# coding: utf-8

# In[28]:


#This code optimizes the gap time (m) for the trapizoidal filter. ~20min for 80000 signals. Use real data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
from numba import jit
import scipy.optimize as opt

import tables
import os


# In[29]:


h5file = tables.open_file("../data/cs137_co60.h5", driver="H5FD_CORE")
print(h5file)
data = h5file.root.RawData
EventData = h5file.root.EventData


# In[30]:


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


# In[31]:


@jit(parallel = True)
def baseLineReduction(mysignal):
    avgNoise = np.mean(mysignal[0:1000])
    return mysignal-avgNoise


# In[32]:


@jit(parallel = True)
def TrapFilter(mysignal,tau,k,m):
    signalOnly = baseLineReduction(mysignal) 
    M = tau
    Acc1=0.0
    Acc2=0.0
    i=0
    zeros = np.zeros(2*k+m)
    extSignalOnly=np.append(zeros,signalOnly)
    final = np.zeros(len(signalOnly))
    for i in range(len(signalOnly)):
        parta = extSignalOnly[i+(2*k+m)] - extSignalOnly[i-k+(2*k+m)]
        partb = extSignalOnly[i-2*k-m+(2*k+m)] - extSignalOnly[i-k-m+(2*k+m)]
        partc = parta + partb
        Acc1 = Acc1+partc
        partd = partc*M+Acc1
        Acc2 = Acc2 + partd
        final[i] = Acc2
        i=i+1
    
    normFinal = np.array(final)
    return normFinal


# In[33]:


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


# In[34]:


@jit(parallel = True)
def func(x, a, b):
    return a * np.exp(-b * x)


# In[35]:


@jit(parallel = True)
def eventsProcess(mysignals, events, cols,tau,k,m):
#    tau = np.zeros(events)
    startTime = datetime.now()
    trap_out = np.zeros((events,cols))
    for j in range(0,events):
        trap_out[j] = TrapFilter(mysignals[j,:],tau,k,m)
    print(datetime.now() - startTime)
    return trap_out


# In[36]:


@jit(parallel = True)
def gauss(x, a, u, sig): # p[0]==mean, p[1]==stdev
#    return 1.0/(sig*np.sqrt(2.0*np.pi))*np.exp(-(x-u)**2.0/(2.0*sig**2.0))
    return a*np.exp(-(x-u)**2.0/(2.0*sig**2.0))

@jit(parallel = True)
def FWHM(counts,lower_bound,upper_bound):
    X = range(lower_bound, upper_bound)
    Y = counts[lower_bound:upper_bound]
    # Fit a guassian
#    p0 = [5590,20] # Inital guess is a normal distribution
#    errfunc = lambda p, x, y: gauss(x, p) - y # Distance to the target function
#    p1, success = opt.leastsq(errfunc, p0[:], args=(X, counts[lower_bound:upper_bound]))
#    print(success)
    mean = sum(X * Y) / sum(Y)
    sigma = np.sqrt(sum(Y * (X - mean)**2) / sum(Y))
    print(sigma)
    pi = [max(Y), mean,sigma]
    popt, pcov = curve_fit(gauss, X, Y, p0=pi)
    print(popt)
    fit_a, fit_mu, fit_stdev = popt
#    plt.plot(X,gauss(X,*popt),'r')
#    plt.bar(X,Y, width=1.0)
#    plt.xlim(5540,5650)
    fwhm = 2*np.sqrt(2*np.log(2))*np.abs(fit_stdev)
    cent = fit_mu
    return fwhm, cent


# In[37]:


@jit(parallel = True)
def TraptoCounts(trap_out,k,M):
    counts = []
    for j in range(0,len(trap_out)):
        counts = np.append(counts,np.amax(trap_out[j]))
    counts2 = np.trim_zeros(counts)
    counts2 = counts2/((M+1)*k)
    counts3 = counts2[(counts2>0) & (counts2<1e3)]
    yhist, bins_hist = np.histogram(counts3,bins=8196)
    return yhist


# In[39]:


print(len(mysignals))
events = len(mysignals)
tau = 5810.79

@jit(parallel = True)
def find_m(mysignals,events,cols,tau,k):
    out_f = []
    out_c = []
    for m in range(0,100,2):
        processedEvents = eventsProcess(mysignals,events,cols,tau,k,m)
        counts = TraptoCounts(processedEvents,k,tau)
        out_fwhm, out_cent = FWHM(counts,5000,6000)
        out_f.append(out_fwhm)
        out_c.append(out_cent)
        print('m=',m)
    return out_f, out_c



startTime = datetime.now()
#find k first
FWHM_m, cent_m = find_m(mysignals,events,cols,tau,740)

print(datetime.now() - startTime)


# In[43]:


fwhm1 = np.array(FWHM_m)
h = np.array(cent_m)

fwhm_f = fwhm1/cent_m

plt.figure()
plt.plot(range(0,100,2),fwhm_f)
plt.xlabel('m value (10ns)')
plt.ylabel('FWHM / H')
plt.title('Energy Resolution vs Gap Time')

plt.savefig('../images/RvsM.png')

