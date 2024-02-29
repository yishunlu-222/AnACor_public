import os
import json
# import pickle
# from matplotlib import pyplot as plt
# from multiprocessing import Process
# import multiprocessing
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils import *
from utils_lite import *
from sys import getsizeof
import resource
import  gc
import ctypes as ct
# from unit_test_pl import *
from scipy.spatial import ConvexHull
# ===========================================
#        Parse the argument
# ===========================================

rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
def str2bool ( v ) :
    if isinstance( v , bool ) :
        return v
    if v.lower( ) in ('yes' , 'true' , 't' , 'y' , '1') :
        return True
    elif v.lower( ) in ('no' , 'false' , 'f' , 'n' , '0') :
        return False
    else :
        raise argparse.ArgumentTypeError( 'Boolean value expected.' )

parser = argparse.ArgumentParser(description="multiprocessing for batches")

parser.add_argument(
    "--low",
    type=int,
    default=0,
    help="the starting point of the batch",
)
parser.add_argument(
    "--up",
    type=int,
    default=-1,
    help="the ending point of the batch",
)
parser.add_argument(
    "--store-paths",
    type=int,
    default=0,
    help="orientation offset",
)

parser.add_argument(
    "--offset",
    type=float,
    default=0,
    help="orientation offset",
)

parser.add_argument(
    "--dataset",
    type=int,
    default=16846,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--modelpath",
    type=str,
    default="./",
    help="full model path",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default="./",
    help="full storing path",
)
parser.add_argument(
    "--refl-path",
    type=str,
    default="./",
    help="full reflection path",
)
parser.add_argument(
    "--expt-path",
    type=str,
    default="./",
    help="full experiment path",
)
parser.add_argument(
    "--li",
    type=float,
    default=0,
    help="abs of liquor",
)
parser.add_argument(
    "--lo",
    type=float,
           default=0,
    help="abs of loop",
)
parser.add_argument(
    "--cr",
    type=float,
        default=0,
    help="abs of crystal",
)
parser.add_argument(
    "--bu",
    type=float,
        default=0,
    help="abs of other component",
)
parser.add_argument(
    "--sampling-num",
    type=int,
    default=5000,
    help="pixel size of tomography",
)
parser.add_argument(
    "--full-iteration",
    type=int,
    default=0,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-x",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-y",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-z",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--by-c",
    type=str2bool,
    default=True,
    help="pixel size of tomography",
)
parser.add_argument(
    "--slicing",
    type=str,
    default='z',
    help="pixel size of tomography",
)
parser.add_argument(
    "--bisection",
    type=str2bool,
    default=False,
    help="pixel size of tomography",
)
parser.add_argument(
    "--partial-illumination",
    type=str2bool,
    default=True,
    help="whether to use partial illumination",
)
global args
args = parser.parse_args()

def ada_sampling(crystal_coordinate ,threshold=15000):
    num=len(crystal_coordinate)
    sampling=1
    result = num
    while result >threshold:
        sampling=sampling *2
        result = num/sampling

 
    return sampling
    



def kp_rotation(axis,theta):
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

def slice_sampling(label_list, dim='z', sampling=5000, auto=True):

    # Find the indices of the non-zero elements directly
    crystal_coordinate = np.argwhere(label_list == rate_list['cr'])
    
    if auto:
        # When sampling ~= N/2000, the results become stable
        sampling = len(crystal_coordinate) // 2000
        print(" The sampling number is {}".format(sampling))
    
    output_lengths = []
    if dim == 'z':
        index = 0

    elif dim == 'y':
        index = 1

    elif dim == 'x':
        index = 2
    zz_u = np.unique(crystal_coordinate[:, index])
    
    # Sort the crystal_coordinate array using the np.argsort() function
    sorted_indices = np.argsort(crystal_coordinate[:, index])
    crystal_coordinate = crystal_coordinate[sorted_indices]
    # total_size=len(crystal_coordinate)

    # Use np.bincount() to count the number of occurrences of each value in the array
    output_lengths = np.bincount(crystal_coordinate[:, index], minlength=len(zz_u))
    zz_u= np.insert(zz_u,0,np.zeros(len(output_lengths)-len(zz_u)))
    # Compute the sampling distribution
    if sampling / len(output_lengths) < 0.5:
        sorted_indices = np.argsort(output_lengths)[::-1] # descending order
        sampling_distribution = np.zeros(len(output_lengths))
        sampling_distribution[sorted_indices[:sampling]] = 1
    else:
        sampling_distribution = np.round(output_lengths / output_lengths.mean() * sampling / len(output_lengths)).astype(int)
    
    coord_list = []

    # Use boolean indexing to filter the output array based on the sampling distribution
    for i, sampling_num in enumerate(sampling_distribution):
        if sampling_num == 0:
            continue
        # output_layer = crystal_coordinate[crystal_coordinate[:, index] == zz_u[i]]
        # Use np.random.choice() to randomly sample elements from the output arrays
        before=output_lengths[:i].sum() 
        after=output_lengths[:i + 1].sum()
        output_layer = crystal_coordinate[before: after]
        numbers=[]
        for k in range(sampling_num):
            
            numbers.append(int(output_lengths[i]/(sampling_num+1) * (k+1)) )

        for num in numbers:
            coord_list.append(output_layer[num])
        # sampled_indices = np.random.choice(range(len(output_layer)), size=int(sampling_num), replace=False)
        # coord_list.extend(output_layer[sampled_indices])
        # pdb.set_trace()

    return np.array(coord_list)
if __name__ == "__main__":

    """label coordinate loading"""

    path_l = ''
    dataset = args.dataset
    label_list = np.load(args.modelpath).astype(np.int8)
    refl_filename = args.refl_path
    expt_filename = args.expt_path   # only contain axes
    save_dir = args.save_dir
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    # zz, yy, xx = np.where(label_list == rate_list['cr'])  
    # crystal_coordinate = np.stack((zz,yy,xx),axis=1)
    # del zz, yy, xx  #
    # gc.collect()
    # sampling = ada_sampling(crystal_coordinate )
    # sampling=2000
    # print("the chosen sampling is {}".format(sampling))

    # seg = int(np.round(len(crystal_coordinate) / sampling))
    # coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    # print(" {} voxels are calculated".format(len(coordinate_list)))

    coord_list = slice_sampling(label_list,dim=args.slicing,sampling=args.sampling_num)
    print(" {} voxels are calculated".format(len(coord_list)))
    """tomography setup """
    pixel_size = args.pixel_size * 1e-3  # it means how large for a pixel of tomobar in real life

    mu_li = args.li*1e3    # (unit in mm-1) 16010
    mu_lo = args.lo*1e3
    mu_cr = args.cr*1e3
    mu_bu=args.bu*1e3
    coe = {'mu_li': mu_li, 'mu_lo': mu_lo, "mu_cr": mu_cr}
    #
    t1 = time.time()
    shape = np.array(label_list.shape)


    with open(expt_filename) as f2:
        axes_data = json.load(f2)
    with open(refl_filename) as f1:
        data = json.load(f1)
    print('The total size of the dataset is {}'.format(len(data)))
    corr = []
    dict_corr = []
    voxel_size=np.array([args.pixel_size_z* 1e-3 ,
                         args.pixel_size_y* 1e-3 ,
                         args.pixel_size_x* 1e-3 ])
    low = args.low
    up = args.up

    if up == -1:
        select_data = data[low:]
    else:
        select_data = data[low:up]

    del data
    coefficients = np.array([mu_li, mu_lo, mu_cr, mu_bu])

    axes=axes_data[0]
 # should be chagned

    kappa_axis=np.array(axes["axes"][1])
    kappa = axes["angles"][1]/180*np.pi
    kappa_matrix = kp_rotation(kappa_axis, kappa)

    phi_axis=np.array(axes["axes"][0])
    phi = axes["angles"][0]/180*np.pi
    phi_matrix = kp_rotation(phi_axis, phi)
  #https://dials.github.io/documentation/conventions.html#equation-diffractometer

    omega_axis=np.array(axes["axes"][2])
    F = np.dot(kappa_matrix , phi_matrix )   # phi is the most intrinsic rotation, then ka

    co=0
    if args.by_c:
                
        class Thetaphi(ct.Structure):
            _fields_ = [("theta", ct.c_double),
                        ("phi", ct.c_double)]


        class Vector3D(ct.Structure):
            _fields_ = [("x", ct.c_int),
                        ("y", ct.c_int),
                        ("z", ct.c_int)]


        class Path2(ct.Structure):
            _fields_ = [("ray", ct.POINTER(Vector3D)),
                        ("posi", ct.POINTER(ct.c_int)),
                        ("classes", ct.POINTER(ct.c_char))]


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
                matrix_ptr = ct.cast(labelPtrMatrix(*(array_tuple)), labelPtrPtr)
                matrix_tuple = matrix_tuple + (matrix_ptr,)
            label_list_ptr = ct.cast(labelPtrCube(*(matrix_tuple)), labelPtrPtrPtr)
            return label_list_ptr


        def python_2_c_2d(arr_2d):
            labelPtr = ct.POINTER(ct.c_int)
            labelPtrPtr = ct.POINTER(labelPtr)
            labelPtrMatrix = labelPtr * label_list.shape[0]
            array_tuple = ()
            # Assign the numpy array to the pointer
            for row in arr_2d:
                array_tuple = array_tuple + (row.ctypes.data_as(labelPtr),)
            arr_2d_ptr = ct.cast(labelPtrMatrix(*(array_tuple)), labelPtrPtr)
            return arr_2d_ptr


        dials_lib = ct.CDLL('./ray_tracing.so')
        # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

        # Define the argument and return types of the function
        # dials_lib.dials_2_thetaphi_22.restype = Thetaphi
        # dials_lib.dials_2_thetaphi_22.argtypes = [
        #     ct.POINTER(ct.c_double), ct.c_int]
        # dials_lib.which_face_2.argtypes = [
        #     np.ctypeslib.ndpointer(dtype=np.int64),
        #     np.ctypeslib.ndpointer(dtype=np.int64),
        #     ct.c_double,
        #     ct.c_double]
        # dials_lib.which_face_2.restype = ct.c_char_p

        # voxel_size_c = (ct.c_double * len(voxel_size))(*voxel_size)

        dials_lib.ray_tracing.restype = ct.c_double
        dials_lib.ray_tracing.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int64),  # crystal_coordinate
            np.ctypeslib.ndpointer(dtype=np.int64),  # crystal_coordinate_shape
            np.ctypeslib.ndpointer(dtype=np.int64),      # coordinate_list
            ct.c_int,                    # coordinate_list_length
            np.ctypeslib.ndpointer(dtype=np.float64),   # rotated_s1
            np.ctypeslib.ndpointer(dtype=np.float64),   # xray
            np.ctypeslib.ndpointer(dtype=np.float64),   # voxel_size
            np.ctypeslib.ndpointer(dtype=np.float64),   # coefficients
            ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),     # label_list
            np.ctypeslib.ndpointer(dtype=np.int64),      # shape
            ct.c_int,                      # full_iteration
            ct.c_int                       # store_paths
        ]
        dials_lib.ray_tracing_sampling.restype = ct.c_double
        dials_lib.ray_tracing_sampling.argtypes = [# crystal_coordinate_shape
            np.ctypeslib.ndpointer(dtype=np.int64),      # coordinate_list
            ct.c_int,                    # coordinate_list_length
            np.ctypeslib.ndpointer(dtype=np.float64),   # rotated_s1
            np.ctypeslib.ndpointer(dtype=np.float64),   # xray
            np.ctypeslib.ndpointer(dtype=np.float64),   # voxel_size
            np.ctypeslib.ndpointer(dtype=np.float64),   # coefficients
            ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),     # label_list
            np.ctypeslib.ndpointer(dtype=np.int64),      # shape
            ct.c_int,                      # full_iteration
            ct.c_int                       # store_paths
        ]
        label_list_c = python_2_c_3d(label_list)
        # crystal_coordinate_shape = np.array(crystal_coordinate.shape)   
    xray_region=[700,500,500,700]

    
    # hull = safe_half_planes_intersection(coord_list[0], bubble_voxels, k=5)
    
    for i, row in enumerate(select_data):
        # if i!=3:
        #     continue
        counter=0
        counter_m=0
        intensity = float(row['intensity.sum.value'])
        scattering_vector = literal_eval(row['s1'])  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']

        rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]
        rotation_frame_angle+=args.offset/180 *np.pi
        rotation_matrix_frame_omega = kp_rotation(omega_axis, rotation_frame_angle)

        total_rotation_matrix = np.dot(rotation_matrix_frame_omega,F)
        total_rotation_matrix = np.transpose(total_rotation_matrix)
        
        xray = -np.array(axes_data[1]["direction"]) 
        xray=np.dot(total_rotation_matrix ,xray)
        rotated_s1 = np.dot(total_rotation_matrix, scattering_vector)

        theta,phi=dials_2_thetaphi_11(rotated_s1)
        theta_1,phi_1=dials_2_thetaphi_11(xray,L1=True)


        if args.by_c:
            result = dials_lib.ray_tracing_sampling(
                                coord_list,len(coord_list) ,
                                rotated_s1, xray, voxel_size,
                            coefficients, label_list_c, shape,
                            args.full_iteration, args.store_paths)
            # result = dials_lib.ray_tracing(crystal_coordinate, crystal_coordinate_shape,
            #                     coordinate_list,len(coordinate_list) ,
            #                     rotated_s1, xray, voxel_size,
            #                 coefficients, label_list_c, shape,
            #                 args.full_iteration, args.store_paths)
        else:
            ray_direction = dials_2_numpy_11(rotated_s1)
            xray_direction = dials_2_numpy_11(xray)
            # absorp = np.empty(len(coordinate_list))
            # for k , index in enumerate( coordinate_list ) :
            #     coord = crystal_coordinate[index]
            absorp = np.empty(len(coord_list))
            for k, coord in enumerate(coord_list):
                if args.partial_illumination is True:
                    in_xray=partial_illumination_selection(xray_region,total_rotation_matrix,coord)
                    if in_xray is False:
                        continue
                face_1 = cube_face(coord, xray_direction, shape, L1=True)
                face_2 = cube_face(coord, ray_direction, shape)
                # if args.bisection is False:
                path_1 = cal_coord_2(theta_1,phi_1,coord,face_1,shape,label_list) # 37
                numbers_1 = np.array(cal_num(path_1,voxel_size))
                path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list)  # 1
                numbers_2 = np.array(cal_num(path_2,voxel_size))  # 3.5s
                pdb.set_trace()

                if args.store_paths == 1 :
                    if k == 0 :
                        path_length_arr_single = np.expand_dims(np.array((numbers_1+numbers_2)), axis = 0)
                    else :

                        path_length_arr_single = np.concatenate(
                            (path_length_arr_single, np.expand_dims(np.array((numbers_1+numbers_2)), axis = 0)), axis = 0)
                absorption = cal_rate((numbers_1+numbers_2), coefficients)

                absorp[k] = absorption
        
            if args.store_paths == 1:
                if i ==0:
                    path_length_arr= np.expand_dims(path_length_arr_single,axis=0 )
                else:
                    path_length_arr= np.concatenate( ( path_length_arr,np.expand_dims(path_length_arr_single,axis=0 ))   ,axis =0 )
            result = absorp.mean()

        # print('counter sum of standard {}'.format(counter))
        # # print('counter_2  {}'.format(counter_2))
        # print('counter sum of itb  {}'.format(counter_m))
        # # print('counter_it2  {}'.format(counter_it2))
        # print("difference in counter is {}".format(
        # (counter-counter_m)/(counter)*100))
        # # pdb.set_trace()
        # print(result)
        print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                  low + len(
                                                                                                      select_data),
                                                                                                  theta * 180 / np.pi,
                                                                                                  phi * 180 / np.pi,
                                                                                                  rotation_frame_angle * 180 / np.pi,
                                                                                                  result))
        pdb.set_trace()
        if i ==10:
            pdb.set_trace()
        corr.append(result)
        t2 = time.time()
        print('it spends {}'.format(t2 - t1))
        
        dict_corr.append({'index' : low + i, 'miller_index' : miller_index,
                          'intensity' : intensity, 'corr' : result, 
                          'theta' : theta * 180 / np.pi,
                          'phi' : phi * 180 / np.pi,
                          'theta_1' : theta_1 * 180 / np.pi,
                          'phi_1' : phi_1 * 180 / np.pi,})
        if i % 1000 == 1 :
            if args.store_paths == 1 :
                np.save(os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)), path_length_arr)
            with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz :  # Pickling
                json.dump(corr, fz, indent = 2)

            with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1 :  # Pickling
                json.dump(dict_corr, f1, indent = 2)
    if args.store_paths == 1 :
        np.save(os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)), path_length_arr)
    with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz :  # Pickling
        json.dump(corr, fz, indent = 2)

    with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1 :  # Pickling
        json.dump(dict_corr, f1, indent = 2)
    with open(os.path.join(save_dir, "{}_time_{}.json".format(dataset, up)), "w") as f1 :  # Pickling
        json.dump(t2-t1, f1, indent = 2)
    print('Finish!!!!')







