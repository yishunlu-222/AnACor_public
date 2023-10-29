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
    from utils.utils_gridding import mp_create_gridding,mp_interpolation_gridding
    from utils.utils_os import stacking,python_2_c_3d,kp_rotation
except:
    from AnACor.utils.utils_rt import *
    from AnACor.utils.utils_ib import *
    from AnACor.utils.utils_gridding import mp_create_gridding,mp_interpolation_gridding
    from AnACor.utils.utils_os import stacking,python_2_c_3d,kp_rotation
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
        default="None",
        help="full reflection path",
    )
    parser.add_argument(
        "--expt-path",
        type=str,
        default="None",
        help="full experiment path um-1",
    )
    parser.add_argument(
        "--absorption-map",
        type=str2bool,
        default=False,
        help="producing absorption map",
    )
    parser.add_argument(
        "--map-theta",
        type=int,
        default=360,
        help="producing absorption map theta number",
    )
    parser.add_argument(
        "--map-phi",
        type=int,
        default=180,
        help="producing absorption map phi number",
    )
 
    parser.add_argument(
        "--gridding-theta",
        type=int,
        default=360,
        help="producing absorption map theta number",
    )
    parser.add_argument(
        "--gridding-phi",
        type=int,
        default=180,
        help="producing absorption map phi number",
    )
    parser.add_argument(
        "--liac",
        type=float,
        required=True,
        help="abs of liquor um-1",
    )
    parser.add_argument(
        "--loac",
        type=float,
        required=True,
        help="abs of loop um-1",
    )
    parser.add_argument(
        "--crac",
        type=float,
        required=True,
        help="abs of crystal um-1",
    )
    parser.add_argument(
        "--buac",
        type=float,
        required=True,
        help="abs of other component um-1",
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
        "--pixel-size-x",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in x dimension in  um",
    )
    parser.add_argument(
        "--pixel-size-y",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in y dimension in  um",
    )
    parser.add_argument(
        "--pixel-size-z",
        type=float,
        default=0.3,
        help="overall pixel size of tomography in z dimension in  um",
    )
    parser.add_argument(
        "--openmp",
        type=str2bool,
        default=False,
        help="calculate by c instead of python",
    )
    parser.add_argument(
        "--gpu",
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
        "--bisection",
        type=str2bool,
        default=False,
        help="activate bisection method",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default='even',
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--gpumethod",
        type=int,
        default=1,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--gridding",
        type=str2bool,
        default=False,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default='linear',
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--bisection-py",
        type=str2bool,
        default=False,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--DEBUG",
        type=str2bool,
        default=False,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--gridding-method",
        type=int,
        default=2,
        help="whether to apply sampling evenly",
    )
    parser.add_argument(
        "--printing",
        type=str2bool,
        default=True,
        help="whether to apply sampling evenly",
    )
    global args
    args = parser.parse_args()
    return args






def worker_function(t1, low,  dataset, selected_data, label_list,
                    voxel_size, coefficients, F, coord_list,
                    omega_axis, axes_data, save_dir, args,
                    offset, full_iteration, store_paths, printing, num_cls):
    corr = []
    dict_corr = []
    arr_scattering = []
    arr_omega = []
    IsExp = 1
    xray = -np.array(axes_data[1]["direction"])
    shape = np.array(label_list.shape)
    anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), './src/ray_tracing_cpu.so'))
    # anacor_lib_cpu = ct.CDLL( './ray_tracing.so' )s
    # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC
    up = low+len(selected_data)
    anacor_lib_cpu.ray_tracing_overall.restype = ct.POINTER(ct.c_double)
    anacor_lib_cpu.ray_tracing_overall.argtypes = [  # crystal_coordinate_shape
        ct.c_int64,  # low
        ct.c_int64,  # up
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int64,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # scattering_vector_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_axis
        np.ctypeslib.ndpointer(dtype=np.float64),  # kp rotation matrix: F
        ct.c_int64,  # len_result
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # store_paths
        ct.c_int,  # num_workers
        ct.c_int,                      # IsExp
    ]

    anacor_lib_cpu.ray_tracing_single.restype = ct.c_double
    anacor_lib_cpu.ray_tracing_single.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # store_paths
        ct.c_int,                      # IsExp
    ]
    anacor_lib_cpu.ib_test.restype = ct.c_double
    anacor_lib_cpu.ib_test.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),      # coordinate_list
        ct.c_int,                    # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),   # rotated_s1
        np.ctypeslib.ndpointer(dtype=np.float64),   # xray
        np.ctypeslib.ndpointer(dtype=np.float64),   # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),   # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),     # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),      # shape
        ct.c_int,                      # full_iteration
        ct.c_int,                     # store_paths
        ct.c_int,                       # num_cls
        ct.c_int,                      # IsExp
    ]
    label_list_c = python_2_c_3d(label_list)
    if args.gpu:
        anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './src/ray_tracing_gpu.so'))
        anacor_lib_gpu.ray_tracing_gpu_overall.restype = ct.POINTER(ct.c_float)
        anacor_lib_gpu.ray_tracing_gpu_overall.argtypes = [  # crystal_coordinate_shape
            ct.c_int,  # low
            ct.c_int,  # up
            np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
            ct.c_int64,  # coordinate_list_length
            np.ctypeslib.ndpointer(dtype=np.float32),  # scattering_vector_list
            np.ctypeslib.ndpointer(dtype=np.float32),  # omega_list
            np.ctypeslib.ndpointer(dtype=np.float32),  # xray
            np.ctypeslib.ndpointer(dtype=np.float32),  # omega_axis
            np.ctypeslib.ndpointer(dtype=np.float32),  # kp rotation matrix: F
            ct.c_int64,  # len_result
            np.ctypeslib.ndpointer(dtype=np.float32),  # voxel_size
            np.ctypeslib.ndpointer(dtype=np.float32),  # coefficients
            ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
            ct.POINTER(ct.c_int8),  # label_list flattened
            np.ctypeslib.ndpointer(dtype=np.int32),  # shape
            ct.c_int,  # full_iteration
            ct.c_int,  # store_paths
            ct.c_int  # gpumethod
        ]

    if args.gpu or args.openmp:
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

        if args.gpu:
            t1 = time.time()
            print("\033[92m GPU  is used for ray tracing \033[0m")

            result_list = anacor_lib_gpu.ray_tracing_gpu_overall(low, low+len(selected_data),
                                                                 coord_list.astype(np.int64), np.int64(len(
                                                                     coord_list)),
                                                                 arr_scattering.astype(np.float32), arr_omega.astype(
                np.float32), xray.astype(np.float32), omega_axis.astype(np.float32),
                F.astype(np.float32), np.int64(len(selected_data)),
                voxel_size.astype(np.float32),
                coefficients.astype(np.float32), label_list_c, label_list.ctypes.data_as(
                    ct.POINTER(ct.c_int8)), shape.astype(np.int32),
                full_iteration, store_paths, args.gpumethod)

            t2 = time.time()
            print('GPU time is {}'.format(t2-t1))
        elif args.openmp is True:
            print(
                "\033[92m Openmp/C with {} cores is used for ray tracing \033[0m".format(args.num_workers))
            result_list = anacor_lib_cpu.ray_tracing_overall(low, low+len(selected_data),
                                                             coord_list, len(
                coord_list),
                arr_scattering, arr_omega, xray, omega_axis,
                F, len(selected_data),
                voxel_size,
                coefficients, label_list_c, shape,
                full_iteration, store_paths, args.num_workers, IsExp)
        else:
            raise RuntimeError(
                "\n Please use either GPU or Openmp/C options to calculate the absorption \n")

        for i in range(len(selected_data)):
            corr.append(result_list[i])
        t2 = time.time()
        anacor_lib_cpu.free(result_list)

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
                result = anacor_lib_cpu.ib_test(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    args.full_iteration, args.store_paths, num_cls, IsExp)
            elif args.single_c:
                if i == 0:

                    print("\033[92m C with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                result = anacor_lib_cpu.ray_tracing_single(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, store_paths, IsExp)

            else:

                if i == 0:

                    print("\033[92m Python with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                ray_direction = dials_2_myframe(rotated_s1)
                xray_direction = dials_2_myframe(xray)

                absorp = np.empty(len(coord_list))
                absorprt = np.empty(len(coord_list))
                for k, coord in enumerate(coord_list):
                    # face_1 = which_face_2(coord, shape, theta_1, phi_1)
                    # face_2 = which_face_2(coord, shape, theta, phi)
                    face_1 = cube_face(coord, xray_direction, shape, L1=True)
                    face_2 = cube_face(coord, ray_direction, shape)

                    if args.bisection_py:
                        it_1, counter_it1 = iterative_bisection(
                            theta_1, phi_1, coord, face_1, label_list, num_cls)
                        it_2, counter_it2 = iterative_bisection(
                            theta, phi, coord, face_2, label_list, num_cls)
                        numbers_1 = np.array(
                            cal_path2_bisection(it_1, voxel_size, face_1))
                        numbers_2 = np.array(
                            cal_path2_bisection(it_2, voxel_size, face_2))

                        if args.DEBUG:
                            path_1 = cal_coord(
                                theta_1, phi_1, coord, face_1, shape, label_list)  # 37
                            path_2 = cal_coord(
                                theta, phi, coord, face_2, shape, label_list)  # 16

                            numbers_1rt = cal_path_plus(
                                path_1, voxel_size)  # 3.5s
                            numbers_2rt = cal_path_plus(
                                path_2, voxel_size)  # 3.5s\

                            absorptionrt = cal_rate(
                                (numbers_1rt + numbers_2rt), coefficients)
                            absorprt[k] = absorptionrt
                            # pdb.set_trace()
                    else:
                        path_1 = cal_coord(
                            theta_1, phi_1, coord, face_1, shape, label_list)  # 37
                        path_2 = cal_coord(
                            theta, phi, coord, face_2, shape, label_list)  # 16

                        numbers_1 = cal_path_plus(path_1, voxel_size)  # 3.5s
                        numbers_2 = cal_path_plus(path_2, voxel_size)  # 3.5s

                    if store_paths == 1:
                        if k == 0:
                            path_length_arr_single = np.expand_dims(
                                np.array((numbers_1 + numbers_2)), axis=0)
                        else:

                            path_length_arr_single = np.concatenate(
                                (
                                    path_length_arr_single, np.expand_dims(np.array((numbers_1 + numbers_2)), axis=0)),
                                axis=0)
                    absorption = cal_rate(
                        (numbers_1 + numbers_2), coefficients)

                    absorp[k] = absorption
                result = absorp.mean()
            if args.DEBUG:
                result_c = anacor_lib_cpu.ray_tracing_single(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, store_paths, IsExp)
                diff = (result_c - result)/result_c
                print('diff is {}'.format(diff))
                try:
                    diff_2 = (result - absorprt.mean()) / absorprt.mean()

                    print('diff_2 is {}'.format(diff_2))
                except:
                    pass
                pdb.set_trace()
            t2 = time.time()
            if printing:
                print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                          low + len(
                                                                                                              selected_data),
                                                                                                          theta * 180 / np.pi,
                                                                                                          phi * 180 / np.pi,
                                                                                                          rotation_frame_angle * 180 / np.pi,
                                                                                                          result))

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


def worker_function_am(t1, low,  dataset, map_data, selected_data, label_list,
                       voxel_size, coefficients, F, coord_list,
                       omega_axis, axes_data, save_dir, args,
                       offset, full_iteration, store_paths, printing, num_cls):
    corr = []
    dict_corr = []
    arr_scattering = []
    arr_thetaphi = []
    arr_map = []
    arr_omega = []
    IsExp = 1
    xray = -np.array(axes_data[1]["direction"])
    shape = np.array(label_list.shape)

    anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), './src/ray_tracing_cpu.so'))
    # anacor_lib_cpu = ct.CDLL( './ray_tracing.so' )s
    # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC
    up = low+len(map_data)
    anacor_lib_cpu.ray_tracing_overall_am.restype = ct.POINTER(ct.c_double)
    anacor_lib_cpu.ray_tracing_overall_am.argtypes = [  # crystal_coordinate_shape
        ct.c_int64,  # low
        ct.c_int64,  # up
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int64,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # scattering_vector_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_list
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # omega_axis
        np.ctypeslib.ndpointer(dtype=np.float64),  # kp rotation matrix: F
        ct.c_int64,  # len_result
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # store_paths
        ct.c_int,  # num_workers
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ct.c_int,                      # IsExp
    ]

    anacor_lib_cpu.ray_tracing_single_am.restype = ct.c_double
    anacor_lib_cpu.ray_tracing_single_am.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        ct.c_double,  # theta
        ct.c_double,  # phi
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # index
        ct.c_int,                      # IsExp
    ]
    anacor_lib_cpu.ib_am.restype = ct.c_double
    anacor_lib_cpu.ib_am.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        ct.c_double,  # theta
        ct.c_double,  # phi
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # index
        ct.c_int,  # num_cls
        ct.c_int,                      # IsExp
    ]
    label_list_c = python_2_c_3d(label_list)
    if args.gpu:
        anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './src/ray_tracing_gpu.so'))
        anacor_lib_gpu.ray_tracing_gpu_overall.restype = ct.POINTER(ct.c_float)
        anacor_lib_gpu.ray_tracing_gpu_overall.argtypes = [  # crystal_coordinate_shape
            ct.c_int,  # low
            ct.c_int,  # up
            np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
            ct.c_int64,  # coordinate_list_length
            np.ctypeslib.ndpointer(dtype=np.float32),  # scattering_vector_list
            np.ctypeslib.ndpointer(dtype=np.float32),  # omega_list
            np.ctypeslib.ndpointer(dtype=np.float32),  # xray
            np.ctypeslib.ndpointer(dtype=np.float32),  # omega_axis
            np.ctypeslib.ndpointer(dtype=np.float32),  # kp rotation matrix: F
            ct.c_int64,  # len_result
            np.ctypeslib.ndpointer(dtype=np.float32),  # voxel_size
            np.ctypeslib.ndpointer(dtype=np.float32),  # coefficients
            ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
            ct.POINTER(ct.c_int8),  # label_list flattened
            np.ctypeslib.ndpointer(dtype=np.int32),  # shape
            ct.c_int,  # full_iteration
            ct.c_int,  # store_paths
            ct.c_int  # gpumethod
        ]

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
    for i, row in enumerate(map_data):
        theta, phi = row
        # scattering_vector = thetaphi_2_dials(theta, phi)
        arr_map.append(myframe_2_dials(thetaphi_2_myframe(theta, phi)))
        arr_thetaphi.append([theta, phi])
    arr_map = np.array(arr_map)

    arr_thetaphi = np.array(arr_thetaphi)
    if args.gpu or args.openmp:

        if args.gpu:
            t1 = time.time()
            print("\033[92m GPU  is used for ray tracing \033[0m")

            result_list = anacor_lib_gpu.ray_tracing_gpu_overall(low, low+len(selected_data),
                                                                 coord_list.astype(np.int64), np.int64(len(
                                                                     coord_list)),
                                                                 arr_scattering.astype(np.float32), arr_omega.astype(
                np.float32), xray.astype(np.float32), omega_axis.astype(np.float32),
                F.astype(np.float32), np.int64(len(selected_data)),
                voxel_size.astype(np.float32),
                coefficients.astype(np.float32), label_list_c, label_list.ctypes.data_as(
                    ct.POINTER(ct.c_int8)), shape.astype(np.int32),
                full_iteration, store_paths, args.gpumethod)

            t2 = time.time()
            print('GPU time is {}'.format(t2-t1))
        elif args.openmp is True:
            print(
                "\033[92m Openmp/C with {} cores is used for ray tracing \033[0m".format(args.num_workers))
            result_list = anacor_lib_cpu.ray_tracing_overall_am(low, low+len(arr_map),
                                                                coord_list, len(
                coord_list),
                arr_scattering, arr_omega, xray, omega_axis,
                F, len(arr_map),
                voxel_size,
                coefficients, label_list_c, shape,
                full_iteration, store_paths, args.num_workers, arr_thetaphi, arr_map, IsExp)
        else:
            raise RuntimeError(
                "\n Please use either GPU or Openmp/C options to calculate the absorption \n")

        for i in range(len(arr_map)):
            corr.append(result_list[i])
        t2 = time.time()
        anacor_lib_cpu.free(result_list)

    else:
        for i, row in enumerate(arr_map):

            rotated_s1 = row
            map_theta, map_phi = arr_thetaphi[i]

            theta, phi = dials_2_thetaphi((rotated_s1))
            # mse_diff(theta, phi, map_theta, map_phi,i)
            # continue

            # theta_1, phi_1 = dials_2_thetaphi(xray, L1=True)
            # scattering_vector = thetaphi_2_dials()
            # if by_c :
            theta_1, phi_1 = 0, 0
            numbers_1 = (0, 0, 0, 0)
            numbers_1rt = (0, 0, 0, 0)
            if args.bisection:
                if i == 0:

                    print("\033[92m C with {} cores is used for bisection \033[0m".format(
                        args.num_workers))
                result = anacor_lib_cpu.ib_am(
                    coord_list, len(coord_list),

                    rotated_s1, theta, phi, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, i, num_cls, IsExp)
            elif args.single_c:
                if i == 0:

                    print("\033[92m C with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                result = anacor_lib_cpu.ray_tracing_single_am(
                    coord_list, len(coord_list),
                    rotated_s1, theta, phi, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, i, IsExp)

            else:

                if i == 0:

                    print("\033[92m Python with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                ray_direction = dials_2_myframe(rotated_s1)
                xray_direction = dials_2_myframe(xray)

                absorp = np.empty(len(coord_list))
                absorprt = np.empty(len(coord_list))
                for k, coord in enumerate(coord_list):
                    # face_1 = which_face_2(coord, shape, theta_1, phi_1)
                    face_2 = cube_face(coord, ray_direction, shape)
                    if args.bisection_py:
                        it_2, counter_it2 = iterative_bisection(
                            theta, phi, coord, face_2, label_list, num_cls)

                        numbers_2 = cal_path2_bisection(
                            it_2, voxel_size, face_2)

                        if args.DEBUG:

                            path_2 = cal_coord(
                                theta, phi, coord, face_2, shape, label_list)  # 16
                            # pdb.set_trace()
                            numbers_2rt = cal_path_plus(
                                path_2, voxel_size)  # 3.5s\

                            absorptionrt = cal_rate(
                                (numbers_1rt + numbers_2rt), coefficients)
                            absorprt[k] = absorptionrt
                    else:

                        path_2 = cal_coord(
                            theta, phi, coord, face_2, shape, label_list)  # 16
                        numbers_2 = cal_path_plus(path_2, voxel_size)  # 3.5s
                    absorption = cal_rate(
                        (numbers_1 + numbers_2), coefficients)

                    absorp[k] = absorption
                result = absorp.mean()
            if args.DEBUG:
                result_c = anacor_lib_cpu.ray_tracing_single_am(
                    coord_list, len(coord_list),
                    rotated_s1, theta, phi, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, i, IsExp)
                diff = (result_c - result)/result_c
                print('diff is {}'.format(diff))
                try:
                    diff_2 = (result - absorprt.mean()) / absorprt.mean()

                    print('diff_2 is {}'.format(diff_2))
                except:
                    pass
                pdb.set_trace()
            t2 = time.time()
            if printing:
                print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                          low + len(
                                                                                                              map_data),
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


def gridding_3D_mp(params):
    label_list, coord_list, voxel_size, coefficients, arr_map_chunk, gridding_method, shape = params






def create_directory(directory_path):
    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass




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
    gridding_dir = os.path.join(save_dir, 'ResultData', 'gridding')
    directories = [save_dir, gridding_dir, result_path, refl_dir]

    for directory in directories:
        create_directory(directory)

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

    coord_list = generate_sampling(label_list, dim=args.slicing, sampling_size=args.sampling_num,
                                   cr=3, auto=args.auto_sampling, method=args.sampling_method, sampling_ratio=args.sampling_ratio)
    num_cls = np.unique(label_list).shape[0]-1
    # num_cls=4
    print("the number of classes is {}".format(num_cls))
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
    len_data = len(select_data)

    if args.absorption_map is True:

        theta = np.linspace(-180, 180, args.map_theta, endpoint=False)
        phi = np.linspace(-90, 90, args.map_phi, endpoint=False)

        theta_grid, phi_grid = np.meshgrid(theta, phi)
        data = np.stack((theta_grid.ravel(), phi_grid.ravel()), axis=-1)
        map_data = data / 180 * np.pi
        len_data = len(map_data)
        if up == -1:
            map_data = map_data[low:]
        else:
            map_data = map_data[low:up]

    # if args.gridding is True:


    coefficients = np.array([mu_li, mu_lo, mu_cr, mu_bu])

    num_processes = args.num_workers

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
    printing=args.printing
    if args.DEBUG:
        printing = True


    if args.gridding is True:
        
        abs_gridding=stacking(gridding_dir,'gridding_{}'.format(args.sampling_ratio))
        # abs_gridding=stacking(gridding_dir,'gridding')
        


 
        if abs_gridding is None:
            print('gridding map is not found')
            print('creating gridding map...')
            mp_create_gridding(t1, low, label_list,dataset,
                             voxel_size, coefficients,coord_list,
                             gridding_dir, args,
                             offset, full_iteration, store_paths, printing, num_cls, args.gridding_method,num_processes)
            print('gridding map is finished and created')

        abs_gridding=stacking(gridding_dir,'gridding_{}'.format(args.sampling_ratio))
        print('Loading gridding map')
        mp_interpolation_gridding(t1, low,  abs_gridding, select_data, label_list,
                          voxel_size, coefficients, F, coord_list,
                          omega_axis, axes_data, gridding_dir, args,
                          offset, full_iteration, store_paths, printing, num_cls,num_processes,args.interpolation_method)

    else:
        def create_process(worker_function, data_slice, data_copy,i):
            return mp.Process(target=worker_function, args=(t1, low + i * each_core, dataset, data_slice, data_copy,
                                                            voxel_size, coefficients, F, coord_list,
                                                            omega_axis, axes_data, args.save_dir, args,
                                                            offset, full_iteration, store_paths, printing, num_cls))

        def determine_worker_function_and_data():
            if args.absorption_map is True:
                return worker_function_am, map_data
            else:
                return worker_function, select_data

        processes = []
        num_processes = 1 if args.gpu or args.openmp else num_processes

        if num_processes > 1:
            each_core = int(len_data // num_processes)
            data_copies = [label_list.copy() for _ in range(num_processes)]
            worker_function, data = determine_worker_function_and_data()
            
            for i in range(num_processes):
                if i != num_processes - 1:
                    process = create_process(worker_function, data[i * each_core:(i + 1) * each_core], data_copies[i,i])
                else:
                    process = create_process(worker_function, data[i * each_core:], data_copies[i],i)

                processes.append(process)

            for process in processes:
                process.start()

            for process in processes:
                process.join()
        else:
            worker_function, data = determine_worker_function_and_data()
            worker_function(t1, low, dataset, data, label_list, voxel_size, coefficients, F, coord_list,
                            omega_axis, axes_data, args.save_dir, args, offset, full_iteration, store_paths, printing, num_cls)


    # Create a list of 48 data copies

    # Create a queue to store the results from each worker process
    # pdb.set_trace()
    # Create a list of worker processes
    # processes = []
    # if args.gpu is True:
    #     num_processes = 1
    # if args.openmp is True:
    #     num_processes = 1

    # if num_processes > 1:

    #     each_core = int(len_data//num_processes)
    #     data_copies = [label_list.copy() for _ in range(num_processes)]
    #     for i in range(num_processes):
    #         # Create a new process and pass it the data copy and result queue
    #         if i != num_processes-1:
    #             if args.absorption_map is True:
    #                 process = mp.Process(target=worker_function_am,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            map_data[i*each_core:(i+1)
    #                                                     * each_core], select_data,  data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))
    #             elif args.gridding is True:
    #                 process = mp.Process(target=worker_function_gridding,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            gridding_data[i*each_core:(i+1)
    #                                                          * each_core], data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))
    #             else:
    #                 process = mp.Process(target=worker_function,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            select_data[i*each_core:(i+1)
    #                                                        * each_core], data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))
    #             # worker_function()
    #         else:
    #             if args.absorption_map is True:
    #                 process = mp.Process(target=worker_function_am,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            map_data[i*each_core:], select_data,  data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))
    #             elif args.gridding is True:
    #                 process = mp.Process(target=worker_function_gridding,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            gridding_data[i *
    #                                                          each_core:],  data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))
    #             else:
    #                 process = mp.Process(target=worker_function,
    #                                      args=(t1, low+i*each_core, dataset,
    #                                            select_data[i *
    #                                                        each_core:], data_copies[i],
    #                                            voxel_size, coefficients, F, coord_list,
    #                                            omega_axis, axes_data, args.save_dir, args,
    #                                            offset, full_iteration, store_paths, printing, num_cls))

    #         processes.append(process)
    #     # pdb.set_trace()
    #     # Start all worker processes
    #     for process in processes:
    #         process.start()

    #     # Wait for all worker processes to finish
    #     for process in processes:
    #         process.join()
    # else:
    #     if args.absorption_map is True:
    #         worker_function_am(t1, low,  dataset, map_data, select_data, label_list,
    #                            voxel_size, coefficients, F, coord_list,
    #                            omega_axis, axes_data, args.save_dir, args,
    #                            offset, full_iteration, store_paths, printing, num_cls)
    #     elif args.gridding is True:
    #         worker_function_gridding(t1, low, dataset,
    #                                  gridding_data,  label_list,
    #                                  voxel_size, coefficients, F, coord_list,
    #                                  omega_axis, axes_data, args.save_dir, args,
    #                                  offset, full_iteration, store_paths, printing, num_cls, args.gridding_method)
    #     else:
    #         worker_function(t1, low,  dataset, select_data, label_list,
    #                         voxel_size, coefficients, F, coord_list,
    #                         omega_axis, axes_data, args.save_dir, args,
    #                         offset, full_iteration, store_paths, printing, num_cls)










if __name__ == '__main__':
    main()
