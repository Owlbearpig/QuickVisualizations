# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:00:46 2021

@author: MachC
"""

from THz.importing import import_teralyzer
import numpy as np
import matplotlib.pyplot as plt


out,names,_=import_teralyzer()

name=[]
for nam in names:
    name.append(nam.split('_')[0])  
name=np.asarray(name)


freqlist=[]
Nlist=[]
Alist=[]
deltaNlist=[]
deltaAlist=[]

for dframe in out:
    freqlist.append(dframe.values[:,0].astype('float64'))
    Nlist.append(dframe.values[:,1].astype('float64'))
    Alist.append(dframe.values[:,3].astype('float64'))
    deltaNlist.append(dframe.values[:,12].astype('float64'))
    deltaAlist.append(dframe.values[:,13].astype('float64'))

freq=np.asarray(freqlist)
Narr=np.asarray(Nlist)
Aarr=np.asarray(Alist)
deltaNarr=np.asarray(deltaNlist)
deltaAarr=np.asarray(deltaAlist)


plt.figure(figsize=(12,5))
plt.subplot(121)
#ax = plt.gca()
for i in range(len(name)):
    #color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(freq[i]*1e-12,Narr[i],label=name[i])#,color=color)
    #plt.plot(freq[i]*1e-12,Narr[i]+deltaNarr[i],ls='--',lw=0.4,color=color)
    #plt.plot(freq[i]*1e-12,Narr[i]-deltaNarr[i],ls='--',lw=0.4,color=color)
    plt.fill_between(freq[i]*1e-12,Narr[i]+deltaNarr[i],Narr[i]-deltaNarr[i],alpha=0.1)
    plt.xlabel('Frequenz [THz]')
    plt.ylabel('Brechungsindex n')
    #plt.ylim((1.15,1.375))
plt.legend()

plt.subplot(122)
for i in range(len(name)):
    #plt.errorbar(freq[i]*1e-12,Aarr[i],deltaAarr[i],label=name[i])
    plt.plot(freq[i]*1e-12,Aarr[i],label=name[i])
    plt.fill_between(freq[i]*1e-12,Aarr[i]+deltaAarr[i],Aarr[i]-deltaAarr[i],alpha=0.1)
    plt.xlabel('Frequenz [THz]')
    plt.ylabel('Absorptionskoeffizient [cm$^{-1}$]')
plt.legend()
plt.savefig('Transmitdat.png',dpi=120,bbox_inches='tight')
plt.show()






