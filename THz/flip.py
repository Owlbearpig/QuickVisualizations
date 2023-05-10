# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:53:33 2019

@author: JanO
"""

from importing import import_tds_gui, selectloc
from preprocessing import rotate 
import numpy as np
import re, os


def flip():
    time, amp, names, paths, _ = import_tds_gui(files=False, prompt = 'Choose folder with files')
    time = rotate(time)
    loc = selectloc()
    for name,t,a in zip(names, time,amp):
        fname = name
        fname = re.split('.txt',fname)            
        fname = os.path.join(loc, fname[0] + '_rotated.txt')    
        np.savetxt(fname, np.vstack((t, a)).T)