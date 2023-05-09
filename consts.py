from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from numpy import pi, inf, array, nan_to_num

post_process_config = {"sub_offset": True, "en_windowing": False, "normalize": False}

# Directly indexed; don't change order
sample_names = []
sample_labels = [1, 2, 3, 4, 5]

thicknesses = [502, 625, 1205, 4130, 9106]

plot_range = slice(25, 200)
plot_range1 = slice(0, 1000)
plot_range_sub = slice(25, 350)

cur_os = os.name

c_thz = c0 * 10 ** -6  # um / ps

um = 10 ** -6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in cur_os:
    data_dir_ext = Path(r"/home/alex/Data/Misc/Klimaanlagenbeeinflussung/Leermessungen")
else:
    data_dir_ext = Path(r"E:\measurementdata\Misc\Klimaanlagenbeeinflussung\Leermessungen")
    try:
        data_dir_ext = Path(r"E:\MeasurementData\MSLA")
        os.scandir(r"E:\MeasurementData\MSLA")
    except FileNotFoundError:
        data_dir_ext = Path(r"OOPS 2")
