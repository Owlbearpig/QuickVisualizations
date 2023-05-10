# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:15:16 2019

@author: JanO
"""

from THz.importing import import_teralyzer, import_tds_gui
import numpy as np
import matplotlib.pyplot as plt
from THz.preprocessing import offset, fft

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data, names, _ = import_teralyzer()

n = []
f = []
alpha = []
plt.figure(1)
for dat, name in zip(data, names):
    ftemp = dat.iloc[:,0].values*1e-12
    ntemp = dat.iloc[:,6].values #1, 6 smoothed
    atemp = dat.iloc[:,8].values #3, 8 smoothed
    n.append(ntemp)
    f.append(ftemp)
    alpha.append(atemp)
    plt.subplot(1,2,1)
    plt.plot(ftemp,ntemp)
    plt.subplot(1,2,2)
    plt.plot(ftemp,atemp)

plt.show()
f = np.asarray(f)
n = np.asarray(n)
alpha = np.asarray(alpha)
idx = f[0,:] < 1.2

plt.figure()
plt.plot(f[0,idx], np.mean(alpha[:,idx],0))
plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption coefficient (1/cm)')
plt.xlim([0.3, 2])
plt.grid(True)
plt.tight_layout()
plt.savefig('Hesperetin_Pressling.png', dpi = 300)
plt.show()

plt.figure()
for al in alpha:
    plt.plot(f[0,idx], al[idx])

plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption coefficient (1/cm)')
plt.xlim([0.3, 1.2])
plt.grid(True)
plt.tight_layout()
plt.savefig('Paper_Hesperetin40mg_Pressling.png', dpi = 300)
plt.show()

t, a, _,_,_ = import_tds_gui()

t, a = offset(t,a,3)
f, amp = fft(t,a)
idx = (f[0,:]>0.3) & (f[0,:]<2)
f = f[0, idx]
amp = np.abs(amp[:,idx])

ref = amp[0:5,:]
sam = amp[5:,:]
T = sam/ref
T2 = -20*np.log10(T)



plt.figure()
plt.plot(f,np.mean(T2,0))
mean = np.mean(T2,0)
std = np.std(T2,0)
plt.fill_between(f, mean - std, mean + std,
                 color='gray', alpha=0.2)
plt.xlabel('Frequency (THz)')
plt.ylabel('Extinction (dB)')
plt.xlim([0.3, 2])
plt.grid(True)
plt.tight_layout()
plt.savefig('Papier1_OhneBeladung_EinzelSchicht_AGKeck_mitSTD.png', dpi = 300)
plt.show()