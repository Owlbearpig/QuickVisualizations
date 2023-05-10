from tkinter import Tk
from tkinter import filedialog as fd
import os.path
import numpy as np
import glob
import re
import pandas as pd
from datetime import datetime as dt 
from time import mktime

def selectloc():    
    root = Tk(); root.iconify()    
    loc=fd.askdirectory(title="Choose directory to save modified files",\
                            parent=root,\
                            )    
    root.destroy()    
    return loc

def selectFile(title = "Choose filename to save", filetypes = [("csv file (.csv)",".csv"),("numpy file (.npz)",".npz")]):    
    root = Tk(); root.iconify()    
    loc=fd.asksaveasfilename(title= title,\
                            parent=root,\
                            filetypes= filetypes)    
    root.destroy()    
    return loc

def import_teralyzer(initialdirectory = '', prompt = 'choose files'):
    root = Tk(); root.iconify()
    file_names=fd.askopenfilename(title=prompt,\
                initialdir = initialdirectory,\
                filetypes=(("csv file","*.csv"),("numpy file","*.npz*")),\
                multiple='true')
    root.destroy()
    out = []
    names = []
    for name in file_names:
        out.append(pd.read_csv(name, delimiter = ','))
        path, tail = os.path.split(name)
        names.append(re.split(r'.csv', tail)[0])
    
    return out, names, path

def extract_rob_data(names):
    info = np.zeros((np.size(names),8), float) #N, x, y, z, 4x quarternions, 
    for i, nam in enumerate(names):
        temp = re.split(r'-', nam)
        #m = re.search(r"\[(\w+)\]", test) # finds only the first occurance between [], where (\w+) let's any character, _ or digit to be written between []. However no signs: "." or "," though
        #(\w+) =  ([A-Za-z0-9_]+)
        temp = re.findall(r"\[(.*?)\]",nam) # findall finds all instances in a strin that are between [] 
        info[i,0] = float(temp[0])
        tmp2 = re.split(r",",temp[1])
        info[i,1:4] = float(tmp2[0]), float(tmp2[1]), float(tmp2[2])
        tmp2 = re.split(r",",temp[2])
        info[i,4:] = float(tmp2[0]), float(tmp2[1]), float(tmp2[2]), float(tmp2[3])
    return info


def extract_timestamp(names, form = 'Taipan'):
    #made for taipan format
    if form == 'Taipan':
        timestamp = np.zeros((np.size(names),1), float)
        for i, name in enumerate(names):
            temp = name[0:10] + ' ' + name[11:26]            
            temp = dt.strptime(temp, '%Y-%m-%d %H-%M-%S.%f').timetuple()
            timestamp[i] = mktime(temp)
    if form == 'KlimaLogger':
        if type(names) == str:        
            temp = names[0:10] + ' ' + names[11:26]            
            temp = dt.strptime(temp, '%Y-%m-%d %H:%M:%S').timetuple()
            timestamp = mktime(temp)
        else:
            timestamp = np.zeros((np.size(names),1), float)
            for i, name in enumerate(names):
                temp = name[0:10] + ' ' + name[11:26]            
                temp = dt.strptime(temp, '%Y-%m-%d %H:%M:%S').timetuple()
                timestamp[i] = mktime(temp)
    return timestamp

def import_klimalogger(name):
    fklima = open(name)
    timestamp = []
    T = []
    rh = []
    for i, line in enumerate(fklima):
        if i == 0:# skip header line
            continue          
        temp = line.split(';')
        timestamp.append(extract_timestamp(temp[0], form = 'KlimaLogger'))
        T.append(float(temp[1].split('"')[1]))
        rh.append(float(temp[2].split('"')[1]))               
    fklima.close()
    timestamp = np.asanyarray(timestamp)
    T = np.asanyarray(T)
    rh = np.asanyarray(rh)
    return timestamp, T, rh

def import_tds_gui(files = True, initialdirectory = '', prompt = 'choose files/folder'):
    list_t = []
    list_a = []
    list_name = []
    list_ctime = []
    paths = ''
    file_names = 1
    imported = 0
    new_format = True
    if files:
        root = Tk(); root.iconify()
        file_names=fd.askopenfilename(title=prompt,\
						initialdir = initialdirectory,\
						filetypes=\
						(("txt file","*.txt"),("numpy file","*.npz*")),\
						multiple='true')
        root.destroy()
        if len(file_names) == 1:
            _, tail = os.path.split(file_names[0])
            if tail == 'imported.npz':
                file = np.load(file_names[0])
                t = file['arr_0']
                a = file['arr_1']
                list_name = file['arr_2']
                paths = file['arr_3']
                list_ctime = file['arr_4']
                imported = 1
            else:
                dtype = re.split(r'.', file_names[0])[-1]
                if dtype == 'npz':
                    print('Warning: Only npz files named imported will be imported')
        else:
            dtype = re.split(r'.', file_names[0])[-1]
            if dtype == 'npz':
                print('Error: Only a single .npz file can be imported')
    else: # select all files in the folder
        root = Tk(); root.iconify()
        directory = fd.askdirectory(title=prompt, initialdir = initialdirectory)
        root.destroy()
        if directory != '':
            directory2 = os.path.join(directory, "imported.npz")
        else:
            directory2 = directory
        if os.path.isfile(directory2):
            file = np.load(directory2)
            t = file['arr_0']
            a = file['arr_1']
            list_name = file['arr_2']
            #paths = file['arr_3']
            paths = directory
            list_ctime = file['arr_4']
            imported = 1
        else:
            directory2 = os.path.join(directory, "*.txt")
            file_names = glob.glob(directory2)
    if (file_names != 1) and (imported == 0):
        length = []
        for names in file_names:
            if new_format:
                try:
                    data=np.loadtxt(names)
                except ValueError:
                    print('Switched to the old format import!')
                    new_format = False
                    data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            else:
                data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            if np.abs(data[-1,0]) < 1: #time in seconds - convert to ps
                list_t.append(data[:,0]*1E12)
            else:
                list_t.append(data[:,0])
            list_a.append(data[:,1])
            length.append(len(data[:,1])) # checking the length
            paths, tail = os.path.split(names)
            list_name.append(tail)
            list_ctime.append(os.path.getmtime(names))
        
        #CHECKING WHETHER ALL FILES HAVE THE SAME NUMBER OF ROWS (CUTTING IF NOT)
        length = np.asarray(length)
        if len(np.unique(length)) != 1:
            print('Warning: files had different amount of rows, from ' + str(min(length)) + ' to ' + str(max(length)) + '!' )
            length = min(length)
            for k in range(len(list_t)):
                list_t[k] = list_t[k][0:length]
                list_a[k] = list_a[k][0:length]
                
                
        t = np.asarray(list_t)
        a = np.asarray(list_a)
        if not files:
            if os.path.exists(os.path.join(paths, "imported.npz")):
                file = selectFile(title = "Choose filename to save", filetypes = [("npz file (.npz)",".npz")])
                savename = file
            else:
                savename = os.path.join(paths, "imported.npz")
            if not savename == '':
                np.savez(savename, t, a, list_name, paths,list_ctime) 
            list_ctime = np.asarray(list_ctime)
    if imported == 1:
        print('.npz file was imported')
    return t, a, np.asarray(list_name), paths, list_ctime
	
	
def import_tds(whole_path): #safety for not overwritting .npz not added
    list_a = []
    list_name = []
    paths = ''
    file_names = 1
    imported = 0
    list_t = []
    list_ctime = []
    sp = re.split(r'.txt', whole_path)
    new_format = True
    if len(sp) == 2: # input was a string with a path directly to a single file
        if new_format:
            try:
                data=np.loadtxt(whole_path)
            except ValueError:
                print('Switched to load format import!')
                new_format = False
                data = pd.read_table(whole_path,header = None, delimiter='\t', decimal=b',').values
        else:
            data = pd.read_table(whole_path,header = None, delimiter='\t', decimal=b',').values
        if np.abs(data[0-1,0]) < 1: #time in seconds - convert to ps
            t = data[:,0] * 1E12
        else:
            t = data[:,0]
        a = data[:,1]
        paths, list_name = os.path.split(whole_path)
        imported = 1 # make sure it does not save .npz and stuff by single file import

    sp = re.split(r'.npz', whole_path)
    if len(sp) == 2: # input was a string with a path directly to a single file
        directory2 = whole_path
    else:
        directory2 = os.path.join(whole_path, "imported.npz")

    if os.path.isfile(directory2):
        file = np.load(directory2)
        t = file['arr_0']
        a = file['arr_1']
        list_name = file['arr_2']
        #paths = file['arr_3']
        paths = whole_path
        list_ctime = file['arr_4']
        imported = 1
    else:
        directory2 = os.path.join(whole_path, "*.txt")
        file_names = glob.glob(directory2)
    if (file_names != 1) and (imported == 0):
        for names in file_names:
            if new_format:
                try:
                    data=np.loadtxt(names)
                except ValueError:
                    print('Switched to load format import!')
                    new_format = False
                    data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            else:
                data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            #data=np.loadtxt(names)
            if np.abs(data[-1,0]) < 1: #time in seconds - convert to ps
                list_t.append(data[:,0]*1E12)
            else:
                list_t.append(data[:,0])
            list_a.append(data[:,1])
            paths, tail = os.path.split(names)
            list_name.append(tail)
            list_ctime.append(os.path.getmtime(names))
        t = np.asarray(list_t)
        a = np.asarray(list_a)
        list_ctime = np.asarray(list_ctime)
        savename = os.path.join(paths, "imported.npz")
        np.savez(savename, t, a, list_name, paths, list_ctime) 
        #Test - there was a problem with the shape of the imported data. Solved
        file = np.load(savename)
        t = file['arr_0']
        a = file['arr_1']
        list_name = file['arr_2']
        #paths = file['arr_3']
        paths = whole_path
        list_ctime = file['arr_4']
        #imported = 1
    if imported == 1:
        print('.npz file or a single .txt file was imported')
    return t, a, list_name, paths, list_ctime


def import_csv_gui(files = True, initialdirectory = '', prompt = 'choose files/folder'):
    list_t = []
    list_a = []
    list_name = []
    list_ctime = []
    paths = ''
    file_names = 1
    imported = 0
    new_format = True
    if files:
        root = Tk(); root.iconify()
        file_names=fd.askopenfilename(title=prompt,\
						initialdir = initialdirectory,\
						filetypes=\
						(("txt file","*.txt"),("numpy file","*.npz*")),\
						multiple='true')
        root.destroy()
        if len(file_names) == 1:
            _, tail = os.path.split(file_names[0])
            if tail == 'imported.npz':
                file = np.load(file_names[0])
                t = file['arr_0']
                a = file['arr_1']
                list_name = file['arr_2']
                paths = file['arr_3']
                list_ctime = file['arr_4']
                imported = 1
            else:
                dtype = re.split(r'.', file_names[0])[-1]
                if dtype == 'npz':
                    print('Warning: Only npz files named imported will be imported')
        else:
            dtype = re.split(r'.', file_names[0])[-1]
            if dtype == 'npz':
                print('Error: Only a single .npz file can be imported')
    else: # select all files in the folder
        root = Tk(); root.iconify()
        directory = fd.askdirectory(title=prompt, initialdir = initialdirectory)
        root.destroy()
        if directory != '':
            directory2 = os.path.join(directory, "imported.npz")
        else:
            directory2 = directory
        if os.path.isfile(directory2):
            file = np.load(directory2)
            t = file['arr_0']
            a = file['arr_1']
            list_name = file['arr_2']
            #paths = file['arr_3']
            paths = directory
            list_ctime = file['arr_4']
            imported = 1
        else:
            directory2 = os.path.join(directory, "*.txt")
            file_names = glob.glob(directory2)
    if (file_names != 1) and (imported == 0):
        length = []
        for names in file_names:
            if new_format:
                try:
                    data=np.loadtxt(names)
                except ValueError:
                    print('Switched to the old format import!')
                    new_format = False
                    data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            else:
                data = pd.read_table(names,header = None, delimiter='\t', decimal=b',').values
            if np.abs(data[-1,0]) < 1: #time in seconds - convert to ps
                list_t.append(data[:,0]*1E12)
            else:
                list_t.append(data[:,0])
            list_a.append(data[:,1])
            length.append(len(data[:,1])) # checking the length
            paths, tail = os.path.split(names)
            list_name.append(tail)
            list_ctime.append(os.path.getmtime(names))
        
        #CHECKING WHETHER ALL FILES HAVE THE SAME NUMBER OF ROWS (CUTTING IF NOT)
        length = np.asarray(length)
        if len(np.unique(length)) != 1:
            print('Warning: files had different amount of rows, from ' + str(min(length)) + ' to ' + str(max(length)) + '!' )
            length = min(length)
            for k in range(len(list_t)):
                list_t[k] = list_t[k][0:length]
                list_a[k] = list_a[k][0:length]
                
                
        t = np.asarray(list_t)
        a = np.asarray(list_a)
        savename = os.path.join(paths, "imported.npz")
        np.savez(savename, t, a, list_name, paths,list_ctime) 
        list_ctime = np.asarray(list_ctime)
    if imported == 1:
        print('.npz file was imported')
    return t, a, np.asarray(list_name), paths, list_ctime