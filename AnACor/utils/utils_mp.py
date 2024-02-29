import os
import json
import time
import pdb
import numpy as np
import ctypes as ct
import multiprocessing as mp
from ast import literal_eval
from multiprocessing import Pool
try:
    from utils.utils_rt import *
    from utils.utils_ib import *
    from utils.utils_os import stacking,python_2_c_3d,kp_rotation
    from utils.utils_mp import *
except:
    from AnACor.utils.utils_rt import *
    from AnACor.utils.utils_ib import *

    from AnACor.utils.utils_os import stacking,python_2_c_3d,kp_rotation
    from AnACor.utils.utils_mp import *

def worker_function(t1, low,  dataset, selected_data, label_list,
                    voxel_size, coefficients, F, coord_list,
                    omega_axis, axes_data, save_dir, args,
                    offset, full_iteration, store_paths, printing, num_cls,logger):
    corr = []
    dict_corr = []
    arr_scattering = []
    arr_omega = []
    IsExp = 1
    xray = -np.array(axes_data[1]["direction"])
    shape = np.array(label_list.shape)
    try:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './src/ray_tracing_cpu.so'))
    except:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), './src/ray_tracing_cpu.so'))
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

    if args.partial_illumination:
        centre_point_on_axis=np.array([args.centre_point_z,
                                    args.centre_point_y, 
                                    args.centre_point_x])
        width =int(args.beam_width/1000/(args.pixel_size_x* 1e-3)/2)
        height=int(args.beam_height/1000/(args.pixel_size_x* 1e-3)/2)
        xray_region=[ centre_point_on_axis[1]-height,centre_point_on_axis[1]+height,centre_point_on_axis[0]-width,centre_point_on_axis[0]+width]   
    if args.gpu:
        try:
            anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), f'./src/ray_tracing_gpu_{args.gpu_card}.so'))
        except:
            anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f'./src/ray_tracing_gpu_{args.gpu_card}.so'))
        # anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), f'./src/ray_tracing_gpu_{args.gpu_card}.so'))
        anacor_lib_gpu.ray_tracing_gpu_overall.restype = ct.POINTER(ct.c_float)
        anacor_lib_gpu.ray_tracing_gpu_overall.argtypes = [  # crystal_coordinate_shape
            ct.c_int64,  # low
            ct.c_int64,  # up
            np.ctypeslib.ndpointer(dtype=np.int32),  # coordinate_list
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
            logger.info("\033[92m GPU  is used for ray tracing \033[0m")
            print("\033[92m GPU  is used for ray tracing \033[0m")
            # Initialize an empty list to hold the indices of non-zero elements.
           
            # abc = label_list.ctypes.data_as(ct.POINTER(ct.c_int8))
            # # Iterate over the array and check each element.
            # for i in range(660*850*850):
            #     if abc[i] != 0:
            #         nonzero_indices.append(i)
            # pdb.set_trace()
            result_list = anacor_lib_gpu.ray_tracing_gpu_overall(low, low+len(selected_data),
                                                                 coord_list.astype(np.int32), np.int64(len(
                                                                     coord_list)),
                                                                 arr_scattering.astype(np.float32), arr_omega.astype(
                np.float32), xray.astype(np.float32), omega_axis.astype(np.float32),
                F.astype(np.float32), np.int64(len(selected_data)),
                voxel_size.astype(np.float32),
                coefficients.astype(np.float32), label_list_c, label_list.ctypes.data_as(
                    ct.POINTER(ct.c_int8)), shape.astype(np.int32),
                full_iteration, store_paths, args.gpumethod)

            t2 = time.time()
            logger.info('GPU time is {}'.format(t2-t1))
            print('GPU time is {}'.format(t2-t1))
        elif args.openmp is True:
            logger.info("\033[92m Openmp/C with {} cores is used for ray tracing \033[0m".format(
                args.num_workers))
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
                if i == 0:
                    logger.info("\033[92m C with {} cores is used for bisection method \033[0m".format(
                        args.num_workers))
                    print("\033[92m C with {} cores is used for bisection method \033[0m".format(args.num_workers))
                result = anacor_lib_cpu.ib_test(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    args.full_iteration, args.store_paths, num_cls, IsExp)
            elif args.single_c:
                if i == 0:
                    logger.info("\033[92m C with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                    print("\033[92m C with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                result = anacor_lib_cpu.ray_tracing_single(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, store_paths, IsExp)

            else:

                if i == 0:
                    logger.info("\033[92m Python with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                    print("\033[92m Python with {} cores is used for ray tracing \033[0m".format(
                        args.num_workers))
                ray_direction = dials_2_myframe(rotated_s1)
                xray_direction = dials_2_myframe(xray)

                absorp = np.empty(len(coord_list))
                absorprt = np.empty(len(coord_list))

                if args.single_distribution:

                    t1 = time.time()

                    # Determine the number of processes based on the number of available CPUs
                    num_processes = os.cpu_count()
                    print('num_processes is {}'.format(num_processes))
                    chunk_size = max(len(coord_list) // num_processes, 1)  # Ensure chunk_size is at least 1

                    chunks=[]
                    for k,i in enumerate( range(0, len(coord_list), chunk_size)):
                        if k !=num_processes-1:
                            chunks.append(coord_list[i:i + chunk_size])
                        else:
                            chunks.append(coord_list[i:])
                    params =  [(chunk, xray_direction, ray_direction, theta_1, phi_1, theta, phi, shape, voxel_size, coefficients, label_list) for chunk in chunks]
                    
                    pool = mp.Pool(num_processes)
                    absorptions_list  = pool.map(process_chunk, params)
                    pool.close()
                    pool.join()
                    absorptions = [absorp for sublist in absorptions_list for absorp in sublist]
                    absorptions = np.array(absorptions)
                    
                    np.save(os.path.join(save_dir, '{}_single_distribution_{}_{}.npy'.format(dataset,low+i, args.sampling_ratio)), absorp)
                    t2 = time.time()
                    with open(os.path.join(save_dir, "{}_time_{}.json".format(dataset, up)), "w") as f1:  # Pickling
                        json.dump(t2 - t1, f1, indent=2)
                    import sys
                    sys.exit(0)


                for k, coord in enumerate(coord_list):
                    if args.partial_illumination :
                            pl = partial_illumination_selection(xray_region, total_rotation_matrix, coord, centre_point_on_axis)
                            pdb.set_trace()
                            
                            if pl is False:
                                continue
                            else:
                                pass
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

                        path_1_f = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list,full_iteration=True)  # 37
                        path_2_f = cal_coord(
                            theta, phi, coord, face_2, shape, label_list,full_iteration=True)  # 16
                        numbers_1_f = cal_path_plus(path_1_f, voxel_size)  # 3.5s
                        numbers_2_f = cal_path_plus(path_2_f, voxel_size)  # 3.5s
                        absorption_f= cal_rate(
                            (numbers_1_f + numbers_2_f), coefficients)
                        absorprt[k] = absorption_f
                        numbers_1 = cal_path_plus(path_1, voxel_size)  # 3.5s

                        numbers_2 = cal_path_plus(path_2, voxel_size)  # 3.5s       

                        # diff_1_cr= (numbers_1[2]-numbers_1_f[2])/numbers_1[2] *100
                        # diff_2_cr= (numbers_2[2]-numbers_2_f[2])/numbers_2[2] *100
                        # diff_1_li= (numbers_1[0]-numbers_1_f[0])/numbers_1[0] *100
                        # diff_2_li= (numbers_2[0]-numbers_2_f[0])/numbers_2[0] *100
                        # print('diff_1_cr is {}'.format(diff_1_cr))
                        # print('diff_2_cr is {}'.format(diff_2_cr))
                        # print('diff_1_li is {}'.format(diff_1_li))
                        # print('diff_2_li is {}'.format(diff_2_li))
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
                    # if args.DEBUG:
                    #     diff = (absorption - absorption_f)/absorption_f
                    #     print('diff is {}'.format(diff))
                    #     pdb.set_trace()
                    #     if diff > 0.01:
                    #         pdb.set_trace()
                result = absorp.mean()
            # pdb.set_trace()
            if args.DEBUG:
                result_c = anacor_lib_cpu.ray_tracing_single(
                    coord_list, len(coord_list),
                    rotated_s1, xray, voxel_size,
                    coefficients, label_list_c, shape,
                    full_iteration, store_paths, IsExp)
                diff = (result_c - result)/result_c
                print('diff is {}'.format(diff))
                try:
                    diff_2 = (result - absorprt.mean()) / absorprt.mean() *100

                    print('diff_2 is {}'.format(diff_2))
                except:
                    pass
                pdb.set_trace()
            t2 = time.time()
            if printing:
                print('[{}/{}:{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,low,
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
    logger.info('{} ({} ) process is Finish!!!!'.format(os.getpid(), up))

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
    try:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './src/ray_tracing_cpu.so'))
    except:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), './src/ray_tracing_cpu.so'))
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
        try:
            anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), f'./src/ray_tracing_gpu_{args.gpu_card}.so'))
        except:
            anacor_lib_gpu = ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f'./src/ray_tracing_gpu_{args.gpu_card}.so'))
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



