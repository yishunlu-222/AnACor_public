import os
import json

import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils.utils_rt import *
import gc
import sys
import multiprocessing
from multiprocessing import Pool
import ctypes as ct
import re
try:
    from scipy.interpolate import interp2d,interpn, RectSphereBivariateSpline,SmoothSphereBivariateSpline
    import psutil
    from memory_profiler import profile
except:
    pass



def kp_rotation(axis, theta):
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param axis:
    :param theta:
    :return:
    """

    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    first_row = np.array([c + (x**2)*(1-c), x*y*(1-c) - z*s, y*s + x*z*(1-c)])
    seconde_row = np.array(
        [z*s + x*y*(1-c),  c + (y**2)*(1-c), -x*s + y*z*(1-c)])
    third_row = np.array([-y*s + x*z*(1-c), x*s + y*z*(1-c), c + (z**2)*(1-c)])
    matrix = np.stack((first_row, seconde_row, third_row), axis=0)
    return matrix

def python_2_c_3d(label_list):
    # this is a one 1d conversion
    # z, y, x = label_list.shape
    # label_list_ctype = (ct.c_int8 * z * y * x)()
    # for i in range(z):
    #     for j in range(y):
    #         for k in range(x):
    #             label_list_ctype[i][j][k] = ct.c_int8(label_list[i][j][k])
    labelPtr = ct.POINTER(ct.c_int8)
    labelPtrPtr = ct.POINTER(labelPtr)
    labelPtrPtrPtr = ct.POINTER(labelPtrPtr)
    labelPtrCube = labelPtrPtr * label_list.shape[0]
    labelPtrMatrix = labelPtr * label_list.shape[1]
    matrix_tuple = ()
    for matrix in label_list:
        array_tuple = ()
        for row in matrix:
            array_tuple = array_tuple + (row.ctypes.data_as(labelPtr),)
        matrix_ptr = ct.cast(labelPtrMatrix(
            *(array_tuple)), labelPtrPtr)
        matrix_tuple = matrix_tuple + (matrix_ptr,)
    label_list_ptr = ct.cast(labelPtrCube(
        *(matrix_tuple)), labelPtrPtrPtr)
    return label_list_ptr

def stacking(path,keyword):
    def sort_key(s):
        if s:
            try:
                c = re.findall('(\d+)', s)[-1]
            except:
                c = -1
            return int(c)

    refl_filaname_list=[]
    for file in os.listdir(path):
        # number = re.findall('(\d+)', file)
        # pdb.set_trace()
        if 'json' not in file:
            continue
        if keyword in file:
                refl_filaname_list.append(file)
    refl_filaname_list.sort(key=sort_key)
    print(refl_filaname_list)
    if len(refl_filaname_list) == 0:
        return None

    

    for j,i in enumerate(refl_filaname_list):
        filename=os.path.join(path,i)

        with open(filename,'r') as f1:
            data = json.load(f1)

        if j ==0:
            corr = data
        else:
            corr+=data

        f1.close()

    return corr