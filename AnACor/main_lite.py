import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
try:
    from utils.utils_rt import *
    from utils.utils_ib import *
except:
    from AnACor.utils.utils_rt import *
    from AnACor.utils.utils_ib import *
import ctypes as ct
import multiprocessing as mp    
# try:
#     from AnACor.RayTracing import RayTracingBasic,kp_rotation
# except:
#     from RayTracing import RayTracingBasic,kp_rotation
# from dials.util.filter_reflections import *
# from dials.algorithms.scaling.scaler_factory import *
# from dials.array_family import flex
# from dxtbx.serialize import load
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
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

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
        type=str,
        default=16846,
        help="1 is true, 0 is false",
    )
    parser.add_argument(
        "--model-storepath",
        type=str,
        required=True,
        help="full model path",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default="./",
        help="full storing path",
    )
    parser.add_argument(
        "--refl-path",
        type=str,
        required=True,
        help="full reflection path",
    )
    parser.add_argument(
        "--expt-path",
        type=str,
        required=True,
        help="full experiment path",
    )
    parser.add_argument(
        "--liac",
        type=float,
        required=True,
        help="abs of liquor",
    )
    parser.add_argument(
        "--loac",
        type=float,
        required=True,
        help="abs of loop",
    )
    parser.add_argument(
        "--crac",
        type=float,
        required=True,
        help="abs of crystal",
    )
    parser.add_argument(
        "--buac",
        type=float,
        required=True,
        help="abs of other component",
    )
    parser.add_argument(
        "--sampling-num",
        type=int,
        default=5000,
        help="sampling for picking crystal point to calculate",
    )
    parser.add_argument(
        "--auto-sampling",
        type=str2bool,
        default=True,
        help="automatically determine sampling number",
    )
    parser.add_argument(
        "--full-iteration",
        type=int,
        default=0,
        help="whether to do full iteration(break when encounter an air point)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.3,
        help="overall pixel size of tomography",
    )
    parser.add_argument(
        "--pixel-size-x",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in x dimension in  mm",
    )
    parser.add_argument(
        "--pixel-size-y",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in y dimension in  mm",
    )
    parser.add_argument(
        "--pixel-size-z",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in z dimension in  mm",
    )
    parser.add_argument(
        "--by-c",
        type=str2bool,
        default=False,
        help="calculate by c instead of python",
    )
    parser.add_argument(
        "--single-c",
        type=str2bool,
        default=False,
        help="calculate by c instead of python",
    )
    parser.add_argument(
        "--slicing",
        type=str,
        default='z',
        help="slicing sampling direction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--test-mode",
        type=str2bool,
        default=False,
        help="test mode",
    )
    parser.add_argument(
        "--bisection" ,
        type = str2bool,
        default = False ,
        help = "activate bisection method" ,
    )
    parser.add_argument(
        "--sampling-method" ,
        type = str,
        default = False ,
        help = "whether to apply sampling evenly" ,
    )
    global args
    args = parser.parse_args()
    return args

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


def worker_function(t1, low, up, dataset, selected_data, label_list,
                    voxel_size, coefficients, F, coord_list,
                    omega_axis, axes_data, save_dir, by_c,
                    offset, full_iteration, store_paths, printing):
    corr = []
    dict_corr = []
    arr_scattering = []
    arr_omega = []
    xray = -np.array(axes_data[1]["direction"])
    shape = np.array(label_list.shape)
    dials_lib = ct.CDLL(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), './ray_tracing.so'))
    # dials_lib = ct.CDLL( './ray_tracing.so' )s
    # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

    dials_lib.ray_tracing_overall.restype = ct.POINTER(ct.c_double)
    dials_lib.ray_tracing_overall.argtypes = [  # crystal_coordinate_shape
        ct.c_int,  # low
        ct.c_int,  # up
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # scattering_vector_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_axis
        np.ctypeslib.ndpointer(dtype=np.float64),  # kp rotation matrix: F
        ct.c_int,  # len_result
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int  # store_paths
    ]

    dials_lib.ray_tracing_sampling.restype = ct.c_double
    dials_lib.ray_tracing_sampling.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int  # store_paths
    ]
    dials_lib.ib_test.restype = ct.c_double
    dials_lib.ib_test.argtypes = [# crystal_coordinate_shape
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
    if by_c:


        # crystal_coordinate_shape = np.array(crystal_coordinate.shape)
        for i, row in enumerate(selected_data):

            intensity = float(row['intensity.sum.value'])
            # all are in x, y , z in the origin dials file
            miller_index = row['miller_index']

            scattering_vector = literal_eval(row['s1'])
            rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]
            rotation_frame_angle += offset / 180 * np.pi
            arr_scattering.append(scattering_vector)
            arr_omega.append(rotation_frame_angle)

        arr_scattering = np.array(arr_scattering)
        arr_omega = np.array(arr_omega)
        # print('low is {} in processor {} the type is {}'.format( low,os.getpid(),type(low) ))
        # print('up is {} in processor {} the type is {}'.format( low+len(selected_data),os.getpid(),type(low+len(selected_data)) ))

        result_list = dials_lib.ray_tracing_overall(low, low+len(selected_data),
                                                    coord_list, len(
                                                        coord_list),
                                                    arr_scattering, arr_omega, xray, omega_axis,
                                                    F, len(selected_data),
                                                    voxel_size,
                                                    coefficients, label_list_c, shape,
                                                    full_iteration, store_paths)
        for i in range(len(selected_data)):
            corr.append(result_list[i])
        t2 = time.time()
        dials_lib.free(result_list)
    else:
        for i, row in enumerate(selected_data):

            intensity = float(row['intensity.sum.value'])
            # all are in x, y , z in the origin dials file
            scattering_vector = literal_eval(row['s1'])
            miller_index = row['miller_index']

            rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]
            rotation_frame_angle += offset / 180 * np.pi
            rotation_matrix_frame_omega = kp_rotation(
                omega_axis, rotation_frame_angle)

            kp_rotation_matrix = np.dot(rotation_matrix_frame_omega, F)
            total_rotation_matrix = np.transpose(kp_rotation_matrix)
            # total_rotation_matrix is orthogonal matrix so transpose is faster than inverse
            # total_rotation_matrix =np.linalg.inv(kp_rotation_matrix)  
            xray = -np.array(axes_data[1]["direction"])

            xray = np.dot(total_rotation_matrix, xray)
            rotated_s1 = np.dot(total_rotation_matrix, scattering_vector)

            theta, phi = dials_2_thetaphi(rotated_s1)
            theta_1, phi_1 = dials_2_thetaphi(xray, L1=True)

            # if by_c :
            if args.bisection:
                result = dials_lib.ib_test(
                                coord_list,len(coord_list) ,
                                rotated_s1, xray, voxel_size,
                            coefficients, label_list_c, shape,
                            args.full_iteration, args.store_paths)
            elif args.single_c:
                                result = dials_lib.ray_tracing_sampling(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, store_paths)
            
            else:
                
                    ray_direction = dials_2_numpy( rotated_s1 )
                    xray_direction = dials_2_numpy( xray )

                    absorp = np.empty( len( coord_list ) )
                    for k , coord in enumerate( coord_list ) :
                        # face_1 = which_face_2(coord, shape, theta_1, phi_1)
                        # face_2 = which_face_2(coord, shape, theta, phi)
                        face_1 = cube_face( coord , xray_direction , shape , L1 = True )
                        face_2 = cube_face( coord , ray_direction , shape )
                        path_1 = cal_coord( theta_1 , phi_1 , coord , face_1 , shape , label_list )  # 37
                        path_2 = cal_coord( theta , phi , coord , face_2 , shape , label_list )  # 16

                        numbers_1 = cal_path_plus( path_1 , voxel_size )  # 3.5s
                        numbers_2 = cal_path_plus( path_2 , voxel_size )  # 3.5s
                        if store_paths == 1 :
                            if k == 0 :
                                path_length_arr_single = np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )
                            else :

                                path_length_arr_single = np.concatenate(
                                    (
                                    path_length_arr_single , np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )) ,
                                    axis = 0 )
                        absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )

                        absorp[k] = absorption
                    result = absorp.mean( )

            t2 = time.time()
            if printing:
                print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                          low + len(
                                                                                                              selected_data),
                                                                                                          theta * 180 / np.pi,
                                                                                                          phi * 180 / np.pi,
                                                                                                          rotation_frame_angle * 180 / np.pi,
                                                                                                          result))
            # pdb.set_trace()

            print('process {} it spends {}'.format(os.getpid(), t2 -
                                                   t1))

            corr.append(result)
            # print( 'it spends {}'.format( t2 - t1 ) )
            dict_corr.append({'index': low + i, 'miller_index': miller_index,
                              'intensity': intensity, 'corr': result,
                              'theta': theta * 180 / np.pi,
                              'phi': phi * 180 / np.pi,
                              'theta_1': theta_1 * 180 / np.pi,
                              'phi_1': phi_1 * 180 / np.pi, })
            if i % 1000 == 1:
                

                with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
                    json.dump(corr, fz, indent=2)
                with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)),
                          "w") as f1:  # Pickling
                    json.dump(dict_corr, f1, indent=2)


    with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
        json.dump(corr, fz, indent=2)

    with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
        json.dump(dict_corr, f1, indent=2)
    with open(os.path.join(save_dir, "{}_time_{}.json".format(dataset, up)), "w") as f1:  # Pickling
        json.dump(t2 - t1, f1, indent=2)
    print('{} ({} ) process is Finish!!!!'.format(os.getpid(), up))



def main():
    args = set_parser()
    print("\n==========\n")
    print("start AAC")
    print("\n==========\n")
    dataset = args.dataset
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
    save_dir = os.path.join(args.store_dir, '{}_save_data'.format(dataset))
    result_path = os.path.join(save_dir, 'ResultData', 'absorption_factors')
    refl_dir = os.path.join(save_dir, 'ResultData', 'reflections')

    try:
        os.makedirs(save_dir)
        os.makedirs(result_path)
        os.makedirs(refl_dir)
    except:
        pass

    if args.model_storepath == 'None':
        models_list = []
        for file in os.listdir(save_dir):
            if dataset in file and ".npy" in file:
                models_list.append(file)

        if len(models_list) == 1:
            model_path = os.path.join(save_dir, models_list[0])
        elif len(models_list) == 0:
            raise RuntimeError(
                "\n There are no 3D models of sample {} in this directory \n  Please create one by command python setup.py \n".format(dataset))
        else:
            raise RuntimeError(
                "\n There are many 3D models of sample {} in this directory \n  Please delete the unwanted models \n".format(dataset))
    else:
        model_path = args.model_storepath

    args.model_storepath = model_path
    args.save_dir = result_path

    # algorithm = RayTracingBasic(args)
    # algorithm.mp_run(printing=True,test=args.test_mode)
    # algorithm.run()
    label_list = np.load(args.model_storepath).astype(np.int8)
    refl_filename = args.refl_path
    expt_filename = args.expt_path   # only contain axes
    # coord_list_even = slice_sampling(label_list, dim=args.slicing, sampling_size=args.sampling_num,
    #                             rate_list=rate_list, auto=args.auto_sampling,method='even')
    # coord_list_random = slice_sampling(label_list, dim=args.slicing, sampling_size=args.sampling_num,
    #                             rate_list=rate_list, auto=args.auto_sampling,method='random')
    # coord_list_slice = slice_sampling(label_list, dim=args.slicing, sampling_size=args.sampling_num,
    #                             rate_list=rate_list, auto=args.auto_sampling,method='slice')
    coord_list = generate_sampling(label_list, dim=args.slicing, sampling_size=args.sampling_num,
                                rate_list=rate_list, auto=args.auto_sampling,method=args.sampling_method)

    print(" {} voxels are calculated".format(len(coord_list)))
  
    """tomography setup """
    # pixel_size = args.pixel_size * 1e-3  # it means how large for a pixel of tomobar in real life

    mu_li = args.liac*1e3    # (unit in mm-1) 16010
    mu_lo = args.loac*1e3
    mu_cr = args.crac*1e3
    mu_bu = args.buac*1e3
    #
    t1 = time.time()
    offset = args.offset
    full_iteration = args.full_iteration
    store_paths = args.store_paths

    with open(expt_filename) as f2:
        axes_data = json.load(f2)
    with open(refl_filename) as f1:
        data = json.load(f1)
    print('The total size of the dataset is {}'.format(len(data)))

    voxel_size = np.array([args.pixel_size_z * 1e-3,
                           args.pixel_size_y * 1e-3,
                           args.pixel_size_x * 1e-3])
    low = args.low
    up = args.up

    if up == -1:
        select_data = data[low:]
    else:
        select_data = data[low:up]

    del data
    coefficients = np.array([mu_li, mu_lo, mu_cr, mu_bu])

    num_workers = args.num_workers
    len_data = len(select_data)
    each_core = int(len_data//num_workers)

    axes = axes_data[0]
 # should be chagned

    kappa_axis = np.array(axes["axes"][1])
    kappa = axes["angles"][1]/180*np.pi
    kappa_matrix = kp_rotation(kappa_axis, kappa)

    phi_axis = np.array(axes["axes"][0])
    phi = axes["angles"][0]/180*np.pi
    phi_matrix = kp_rotation(phi_axis, phi)
  # https://dials.github.io/documentation/conventions.html#equation-diffractometer

    omega_axis = np.array(axes["axes"][2])
    # phi is the most intrinsic rotation, then kappa
    F = np.dot(kappa_matrix, phi_matrix)

    printing = True
    # Create a list of 48 data copies
    data_copies = [label_list.copy() for _ in range(num_workers)]

    # Create a queue to store the results from each worker process
    # pdb.set_trace()
    # Create a list of worker processes
    processes = []
    if num_workers > 1:
        for i in range(num_workers):
            # Create a new process and pass it the data copy and result queue
            if i != num_workers-1:
                process = mp.Process(target=worker_function,
                                     args=(t1, i*each_core, (i+1)*each_core, dataset,
                                           select_data[i*each_core:(i+1)
                                                       * each_core], data_copies[i],
                                           voxel_size, coefficients, F, coord_list,
                                           omega_axis, axes_data, args.save_dir, args.by_c,
                                           offset, full_iteration, store_paths, printing))
                # worker_function()
            else:
                process = mp.Process(target=worker_function,
                                     args=(t1, i*each_core, '-1', dataset,
                                           select_data[i *
                                                       each_core:], data_copies[i],
                                           voxel_size, coefficients, F, coord_list,
                                           omega_axis, axes_data, args.save_dir, args.by_c,
                                           offset, full_iteration, store_paths, printing))

            processes.append(process)
        # pdb.set_trace()
        # Start all worker processes
        for process in processes:
            process.start()

        # Wait for all worker processes to finish
        for process in processes:
            process.join()
    else:
        worker_function(t1, 0, '-1', dataset, select_data, label_list,
                             voxel_size, coefficients, F, coord_list,
                             omega_axis, axes_data, save_dir, args.by_c,
                             offset, full_iteration, store_paths, printing)


if __name__ == '__main__':
    main()
