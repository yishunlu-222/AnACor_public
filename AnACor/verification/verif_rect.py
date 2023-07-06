import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
# from utils import *
import ctypes as ct
import multiprocessing as mp
from analytical import ana_f_exit_sides , ana_f_exit_top
# ===========================================
#        Parse the argument
# ===========================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_parser():
    def str2bool ( v ) :
        if isinstance( v , bool ) :
            return v
        if v.lower( ) in ('yes' , 'true' , 't' , 'y' , '1') :
            return True
        elif v.lower( ) in ('no' , 'false' , 'f' , 'n' , '0') :
            return False
        else :
            raise argparse.ArgumentTypeError( 'Boolean value expected.' )

    parser = argparse.ArgumentParser( description = "multiprocessing for batches" )

    parser.add_argument(
        "--low" ,
        type = int ,
        default = 0 ,
        help = "the starting point of the batch" ,
    )
    parser.add_argument(
        "--up" ,
        type = int ,
        default = -1 ,
        help = "the ending point of the batch" ,
    )
    parser.add_argument(
        "--store-paths" ,
        type = int ,
        default = 0 ,
        help = "orientation offset" ,
    )


    global args
    args = parser.parse_args()
    return args

def kp_rotation(axis,theta, raytracing=True):
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param axis:
    :param theta:
    :return:
    """

    x,y,z = axis
    c =np.cos(theta)
    s = np.sin(theta)
    first_row = np.array([ c + (x**2)*(1-c), x*y*(1-c) - z*s, y*s + x*z*(1-c)  ])
    seconde_row = np.array([z*s + x*y*(1-c),  c + (y**2)*(1-c) , -x*s + y*z*(1-c) ])
    third_row = np.array([ -y*s + x*z*(1-c), x*s + y*z*(1-c), c + (z**2)*(1-c)  ])
    matrix = np.stack(( first_row, seconde_row, third_row), axis = 0)
    return matrix

def python_2_c_3d ( label_list ) :
    # this is a one 1d conversion
    # z, y, x = label_list.shape
    # label_list_ctype = (ct.c_int8 * z * y * x)()
    # for i in range(z):
    #     for j in range(y):
    #         for k in range(x):
    #             label_list_ctype[i][j][k] = ct.c_int8(label_list[i][j][k])
    labelPtr = ct.POINTER( ct.c_int8 )
    labelPtrPtr = ct.POINTER( labelPtr )
    labelPtrPtrPtr = ct.POINTER( labelPtrPtr )
    labelPtrCube = labelPtrPtr * label_list.shape[0]
    labelPtrMatrix = labelPtr * label_list.shape[1]
    matrix_tuple = ()
    for matrix in label_list :
        array_tuple = ()
        for row in matrix :
            array_tuple = array_tuple + (row.ctypes.data_as( labelPtr ) ,)
        matrix_ptr = ct.cast( labelPtrMatrix( *(array_tuple) ) , labelPtrPtr )
        matrix_tuple = matrix_tuple + (matrix_ptr ,)
    label_list_ptr = ct.cast( labelPtrCube( *(matrix_tuple) ) , labelPtrPtrPtr )
    return label_list_ptr

def main(mu,width,height,resolution):
    args=set_parser()
    print("\n==========\n")
    print("start AAC")
    print("\n==========\n")
    angle_list=np.linspace(start = 0,stop=90,num = 20,endpoint = True)

    errors=[]
    coefficients = np.array([0,0,mu,0]).astype(np.float64)
    ana_list=[]
    gpu_list=[]
    errors_result=[]
    # resolution = 100  # must be the factor of 10
    for angle in angle_list:
        t_theta = angle / 180 * np.pi
        try :
            T_l_2 = ana_f_exit_top( mu , t_theta , width , height )
        except :
            T_l_2 = ana_f_exit_sides( mu , t_theta , width , height )
        ana_list.append(T_l_2)
    
    pixel_size= min( width , height ) / resolution
    voxel_size = np.array([pixel_size,pixel_size,pixel_size])
    label_list = np.ones( (1 , int( height * resolution  ) , int( width * resolution  )) ).astype(np.int8)   #0.3875031836219992
    shape=np.array(label_list.shape)
    zz , yy , xx = np.where( label_list == 1 )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )

    xray = np.array( [0 , 0 , -1] )
    theta , phi =  angle / 180 * np.pi, 0/ 180 * np.pi
    theta_1 , phi_1 = 180 / 180 * np.pi, 0/ 180 * np.pi



    dials_lib = ct.CDLL( os.path.join( os.path.dirname( os.path.abspath( __file__ )), './verif_ray_tracing.so' ))
        # dials_lib = ct.CDLL( './ray_tracing.so' )s
        # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

    dials_lib.ray_tracing_gpu_verification.restype = ct.POINTER(ct.c_double)
    dials_lib.ray_tracing_gpu_verification.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer( dtype = np.float64 ) ,
        ct.c_int ,  # angle_list_length
        np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # coordinate_list
        ct.c_int ,  # coordinate_list_length
        ct.POINTER( ct.c_int8 ) ,  # label_list
        np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # voxel_size
        np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # coefficients
        np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # shape
    ]
        # crystal_coordinate_shape = np.array(crystal_coordinate.shape)


    result_list =  dials_lib.ray_tracing_gpu_verification(angle_list,len(angle_list),crystal_coordinate,len(crystal_coordinate),label_list.ctypes.data_as(ct.POINTER(ct.c_int8)) ,voxel_size,coefficients,shape)

    for i in range(len(angle_list)):
        gpu_list.append(result_list[i])
    t2 = time.time()
    dials_lib.free(result_list)
    pdb.set_trace()
    gpu_arr=np.array(gpu_list)
    ana_arr=np.array(ana_list)
    err=(gpu_arr - ana_arr)/ana_arr *100
    for i,angle in enumerate(angle_list):
        errors_result.append([angle,err[i]])
    with open("rectangular sample 1 w1_h1.json", "w") as f1:  # Pickling
        json.dump(errors_result, f1, indent=2)



    

if __name__ == '__main__':
    mu=1
    width = 1
    height = 1
    resolution = 500
    main(mu,width,height,resolution)