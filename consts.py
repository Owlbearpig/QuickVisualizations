from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from numpy import pi, inf, array, nan_to_num

post_process_config = {"sub_offset": True, "en_windowing": False, "normalize": False}

sample_names = []

# thicknesses = [502, 625, 1205, 4130, 9106]
thicknesses = [486, 585, 1163, 4098, 9106]  # teralyzer
# thicknesses = [502, 585, 1205, 4130, 9106]

samples = {"1": thicknesses[0], "2": thicknesses[1], "3": thicknesses[2], "4": thicknesses[3], "5": thicknesses[4]}

plot_range = slice(25, 200)
plot_range1 = slice(0, 1000)
plot_range_sub = slice(25, 250)

cur_os = os.name

c_thz = c0 * 10 ** -6  # um / ps

um = 10 ** -6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in cur_os:
    data_dir_ext = Path(r"/home/alex/Data/MSLA")
else:
    data_dir_ext = Path(r"E:\MeasurementData\MSLA")
    try:
        data_dir_ext = Path(r"E:\MeasurementData\MSLA")
        os.scandir(r"E:\MeasurementData\MSLA")
    except FileNotFoundError:
        data_dir_ext = Path(r"OOPS 2")
