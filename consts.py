from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from numpy import pi, inf, array

# Directly indexed; don't change order
sample_names = ["10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3", "5x5cm_sqrd", "2023-03-20", "2023-03-21"]
sample_labels = ["(200 nm Ag)", "(500 nm Al:ZnO)", "(200 nm Ag + 500 nm Al:ZnO)", "(200 nm ITO)"]


plot_range = slice(25, 200)
plot_range1 = slice(0, 1000)
plot_range_sub = slice(25, 350)

cur_os = os.name

c_thz = c0 * 10 ** -9  # mm / s

um = 10 ** -6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in cur_os:
    data_dir = Path(r"/home/alex/Data/Misc/Klimaanlagenbeeinflussung/Leermessungen")
else:
    data_dir = Path(r"E:\measurementdata\Misc\Klimaanlagenbeeinflussung\Leermessungen")
    try:
        os.scandir(data_dir)
    except FileNotFoundError:
        data_dir = Path(r"OOPS 2")
