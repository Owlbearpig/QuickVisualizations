# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:00:46 2021

@author: MachC
"""
from mpl_settings import *
from THz.importing import import_teralyzer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy as copy
import scipy.signal as sig

out,names,_=import_teralyzer()

#woodcl1=np.loadtxt('Woodcluster/Woodcluster1.txt')
#woodcl2=np.loadtxt('Woodcluster/Woodcluster2.txt')

name=[]
for nam in names:
    nam=nam.replace('Chaetomium','C')
    #name.append(' '.join(nam.split('_')[0:-1]))
    name.append(' '.join(nam.split('_')))
name=np.asarray(name)


freqlist=[]
Nlist=[]
Alist=[]
deltaNlist=[]
deltaAlist=[]

def findFirstEmptyRowPos(dframe):
    minim=np.argmin(dframe.values[:,0])
    if(' ' in str(dframe.values[:,0][minim])):
        return minim
    else:
        return len(dframe.values[:,0])

for dframe in out:
    emptyRowPos=findFirstEmptyRowPos(dframe)
    freqlist.append(dframe.values[:,0][0:emptyRowPos].astype('float64'))
    Nlist.append(dframe.values[:,1][0:emptyRowPos].astype('float64'))
    Alist.append(dframe.values[:,3][0:emptyRowPos].astype('float64'))
    deltaNlist.append(dframe.values[:,12][0:emptyRowPos].astype('float64'))
    deltaAlist.append(dframe.values[:,13][0:emptyRowPos].astype('float64'))

freq=np.asarray(freqlist)
Narr=np.asarray(Nlist)
Aarr=np.asarray(Alist)
deltaNarr=np.asarray(deltaNlist)
deltaAarr=np.asarray(deltaAlist)

def SVMAF(narr,deltanarr,repeats=5):
    def corralgorithm(narr, deltanarr):
        narrsav=sig.savgol_filter(narr,3,0,axis=0)
        narrcorr=copy.deepcopy(narr)
        
        idx=np.abs(narrsav-narr)<=np.abs(deltanarr)
        narrcorr[idx]=narrsav[idx]
        
        return narrcorr
        
    narrcorr=copy.deepcopy(narr)
    for i in range(repeats+1):
        narrcorr=corralgorithm(narrcorr,deltanarr)

    return narrcorr

Narrsav=[]
Aarrsav=[]
for i in range(len(name)):
    Narrsav.append(SVMAF(Narr[i],deltaNarr[i]))
    Aarrsav.append(SVMAF(Aarr[i],deltaAarr[i]))
Narrsav=np.asarray(Narrsav)
Aarrsav=np.asarray(Aarrsav)

nstartvallist=[]
for i in range(len(name)):
    nstartvallist.append(Narr[i][0])
nstartvals=np.asanyarray(nstartvallist)
nsmin=np.min(nstartvals)
nsmax=np.max(nstartvals)
#colmult=256/(nsmax-nsmin)
#ncolvalues=((nstartvals-nsmin)*colmult).astype('int')
colmult=64/(nsmax-nsmin)
ncolvalues=((nstartvals-nsmin)*colmult).astype('int')+16#14
colmap=cm.get_cmap('nipy_spectral',92)


fig=plt.figure(figsize=(15,5))
plt.subplot(121)
#ax = plt.gca()
for i in range(len(name)):
    #color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(freq[i]*1e-12,Narrsav[i],label=name[i],color=colmap(ncolvalues[i]))
    #plt.plot(freq[i]*1e-12,Narr[i]+deltaNarr[i],ls='--',lw=0.4,color=color)
    #plt.plot(freq[i]*1e-12,Narr[i]-deltaNarr[i],ls='--',lw=0.4,color=color)
    plt.fill_between(freq[i]*1e-12,Narr[i]+deltaNarr[i],Narr[i]-deltaNarr[i],alpha=0.08,color=colmap(ncolvalues[i]))
    
#plt.plot(woodcl1[:,0],woodcl1[:,1],label='Lindenholz Cluster 1',color='black')
#plt.fill_between(woodcl1[:,0],woodcl1[:,2]+woodcl1[:,3],woodcl1[:,2]-woodcl1[:,3],alpha=0.1,color='black')
#plt.plot(woodcl2[:,0],woodcl2[:,1],label='Lindenholz Cluster 2',color='black')
#plt.fill_between(woodcl2[:,0],woodcl2[:,2]+woodcl2[:,3],woodcl2[:,2]-woodcl2[:,3],alpha=0.1,color='black')

plt.xlabel('Frequenz [THz]')
plt.ylabel('Brechungsindex n')
#plt.ylim((1.15,1.375))
#plt.legend()

plt.subplot(122)
for i in range(len(name)):
    #plt.errorbar(freq[i]*1e-12,Aarr[i],deltaAarr[i],label=name[i])
    plt.plot(freq[i]*1e-12,Aarrsav[i],label=name[i],color=colmap(ncolvalues[i]))
    plt.fill_between(freq[i]*1e-12,Aarr[i]+deltaAarr[i],Aarr[i]-deltaAarr[i],alpha=0.08,color=colmap(ncolvalues[i]))
    
#plt.plot(woodcl1[:,0],woodcl1[:,4],label='Lindenholz Cluster 1',color='black')
#plt.fill_between(woodcl1[:,0],woodcl1[:,5]+woodcl1[:,6],woodcl1[:,5]-woodcl1[:,6],alpha=0.1,color='black')
#plt.plot(woodcl2[:,0],woodcl2[:,4],label='Lindenholz Cluster 2',color='black')
#plt.fill_between(woodcl2[:,0],woodcl2[:,5]+woodcl2[:,6],woodcl2[:,5]-woodcl2[:,6],alpha=0.1,color='black')

#plt.ylim((0,44))
plt.ylim(bottom=0)
plt.xlabel('Frequenz [THz]')
plt.ylabel('Absorptionskoeffizient [cm$^{-1}$]')

handles,labels=plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc=1)#, fontsize='x-small', ncol=2)#,ncol=5)
fig.tight_layout()
fig.subplots_adjust(right=0.85) 

plt.savefig('Transmitdat.png',dpi=120,bbox_inches='tight')
plt.show()




"""
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig=plt.figure(figsize=(15,5))
plt.subplot(121)
#ax = plt.gca()
for i in range(len(name)):
    #color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(freq[i]*1e-12,Narrsav[i],label=name[i],color=colors[i%len(colors)])
    plt.fill_between(freq[i]*1e-12,Narr[i]+deltaNarr[i],Narr[i]-deltaNarr[i],alpha=0.08,color=colors[i%len(colors)])
    
plt.plot(woodcl1[:,0],woodcl1[:,1],label='Lindenholz Cluster 1',color='black')
plt.fill_between(woodcl1[:,0],woodcl1[:,2]+woodcl1[:,3],woodcl1[:,2]-woodcl1[:,3],alpha=0.1,color='black')
plt.plot(woodcl2[:,0],woodcl2[:,1],label='Lindenholz Cluster 2',color='black')
plt.fill_between(woodcl2[:,0],woodcl2[:,2]+woodcl2[:,3],woodcl2[:,2]-woodcl2[:,3],alpha=0.1,color='black')

plt.xlabel('Frequenz [THz]')
plt.ylabel('Brechungsindex n')
#plt.ylim((1.15,1.375))
#plt.legend()

plt.subplot(122)
for i in range(len(name)):
    #plt.errorbar(freq[i]*1e-12,Aarr[i],deltaAarr[i],label=name[i])
    plt.plot(freq[i]*1e-12,Aarrsav[i],label=name[i],color=colors[i%len(colors)])
    plt.fill_between(freq[i]*1e-12,Aarr[i]+deltaAarr[i],Aarr[i]-deltaAarr[i],alpha=0.08,color=colors[i%len(colors)])
    
plt.plot(woodcl1[:,0],woodcl1[:,4],label='Lindenholz Cluster 1',color='black')
plt.fill_between(woodcl1[:,0],woodcl1[:,5]+woodcl1[:,6],woodcl1[:,5]-woodcl1[:,6],alpha=0.1,color='black')
plt.plot(woodcl2[:,0],woodcl2[:,4],label='Lindenholz Cluster 2',color='black')
plt.fill_between(woodcl2[:,0],woodcl2[:,5]+woodcl2[:,6],woodcl2[:,5]-woodcl2[:,6],alpha=0.1,color='black')

plt.xlabel('Frequenz [THz]')
plt.ylabel('Absorptionskoeffizient [cm$^{-1}$]')

handles,labels=plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc=1)#,ncol=5)
fig.tight_layout()
fig.subplots_adjust(right=0.8) 

plt.savefig('Transmitdat.png',dpi=120,bbox_inches='tight')
plt.show()

"""



colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig=plt.figure(figsize=(15,5))
plt.subplot(121)
colnum=0
for i in range(len(name)):
    color=colors[colnum]
    colnum = (colnum + 1) % len(colors)
    """
    if(i<len(name)-1 and not name[i].split(' ')[0][0:5]==name[i+1].split(' ')[0][0:5]):
        colnum=(colnum+1)%len(colors)
    if(name[i].split(' ')[1]=='C2W'):
        lstyle=':'
    elif(name[i].split(' ')[1]=='C4W'):
        lstyle='--'
    else:
        lstyle='-'
    """
    lstyle='-'
    plt.plot(freq[i]*1e-12,Narrsav[i],label=name[i],color=color,ls=lstyle)
    plt.fill_between(freq[i]*1e-12,Narr[i]+deltaNarr[i],Narr[i]-deltaNarr[i],alpha=0.08,color=color)

"""
plt.plot(woodcl1[:,0],woodcl1[:,1],label='Lindenholz Cluster 1',color='black')
plt.fill_between(woodcl1[:,0],woodcl1[:,2]+woodcl1[:,3],woodcl1[:,2]-woodcl1[:,3],alpha=0.1,color='black')
plt.plot(woodcl2[:,0],woodcl2[:,1],label='Lindenholz Cluster 2',color='black')
plt.fill_between(woodcl2[:,0],woodcl2[:,2]+woodcl2[:,3],woodcl2[:,2]-woodcl2[:,3],alpha=0.1,color='black')
"""

plt.xlabel('Frequenz [THz]')
plt.ylabel('Brechungsindex n')
#plt.ylim((1.15,1.375))
#plt.legend()

plt.subplot(122)
colnum=0
for i in range(len(name)):
    color=colors[colnum]
    colnum = (colnum + 1) % len(colors)
    """
    if(i<len(name)-1 and not name[i].split(' ')[0][0:5]==name[i+1].split(' ')[0][0:5]):
        colnum=(colnum+1)%len(colors)
    if(name[i].split(' ')[1]=='C2W'):
        lstyle=':'
    elif(name[i].split(' ')[1]=='C4W'):
        lstyle='--'
    else:
        lstyle='-'
    """
    lstyle='-'
    plt.plot(freq[i]*1e-12,Aarrsav[i],label=name[i],color=color,ls=lstyle)
    plt.fill_between(freq[i]*1e-12,Aarr[i]+deltaAarr[i],Aarr[i]-deltaAarr[i],alpha=0.08,color=color)

""" 
plt.plot(woodcl1[:,0],woodcl1[:,4],label='Lindenholz Cluster 1',color='black')
plt.fill_between(woodcl1[:,0],woodcl1[:,5]+woodcl1[:,6],woodcl1[:,5]-woodcl1[:,6],alpha=0.1,color='black')
plt.plot(woodcl2[:,0],woodcl2[:,4],label='Lindenholz Cluster 2',color='black')
plt.fill_between(woodcl2[:,0],woodcl2[:,5]+woodcl2[:,6],woodcl2[:,5]-woodcl2[:,6],alpha=0.1,color='black')
"""

plt.xlabel('Frequenz [THz]')
plt.ylabel('Absorptionskoeffizient [cm$^{-1}$]')

handles,labels=plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc=1)#,ncol=5)
fig.tight_layout()
fig.subplots_adjust(right=0.8) 

plt.savefig('Transmitdat.png',dpi=120,bbox_inches='tight')
plt.show()




