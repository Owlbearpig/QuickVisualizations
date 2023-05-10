# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:38:08 2019

@author: JanO
"""

import numpy as np
from scipy.linalg import toeplitz, norm

def decon_matrix(ref):
    #======================CHANGES=========================================
    c=np.hstack((ref,0))
    c = toeplitz(c,c[-1::-1])
    H = c[0:-1,0:-1]
    #H = toeplitz(ref)
    #=======================================================================
    tau = 1 / norm(H.T @ H, 2)
    return H, tau

def sparse_decon(sample, H, tau, lambda_=0.1, eps=1e-7):
    
    def soft_threshold(v):
        ret = np.zeros(v.shape)
        
        id_smaller = v <= -lambda_ * tau
        id_larger  = v >=  lambda_ * tau
        
        ret[id_smaller] = v[id_smaller] + lambda_ * tau
        ret[id_larger]  = v[id_larger] - lambda_ * tau
        
        return ret
    
    f = np.zeros(sample.shape)
    ssq = norm(f, 1)
    
    relerr = 1
    
    while relerr > eps:
        f = soft_threshold(f - tau * H.T @ (H @ f - sample))
        ssq_new = norm(f, 1)
        relerr = abs(1 - ssq/ssq_new)
        ssq = ssq_new

    return f