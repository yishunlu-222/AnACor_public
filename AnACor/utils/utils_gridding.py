import os
import json

import time
import pdb
import numpy as np
import re
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils.utils_rt import (
    dials_2_thetaphi,
    thetaphi_2_myframe,
    myframe_2_dials,
    dials_2_myframe,
    cube_face,
    cal_coord,
    cal_path_plus,
    cal_rate,
    cal_rate_single,
)
from utils.utils_os import python_2_c_3d, kp_rotation,stacking
import gc
import sys
import multiprocessing as mp
# from multiprocessing import Pool, sharedctypes, shared_memory, Process
import ctypes as ct
from scipy.interpolate import RegularGridInterpolator

try:
    from scipy.interpolate import (
        interp2d,
        interpn,
        RectSphereBivariateSpline,
        SmoothSphereBivariateSpline,
    )
    import psutil
    from memory_profiler import profile
except:
    pass



def loading_absorption_map(gridding_dir,afterfix):
    def sorting(s):
        return int(re.findall(r'\d+', s)[-1])
    absorption_map_list =[name for name in os.listdir(gridding_dir) if name.endswith(afterfix)]
    if len(absorption_map_list) == 0:
        return None
    absorption_map_list.sort(key=sorting)
    for i in range(len(absorption_map_list)):
        map = np.load(f'{gridding_dir}/{absorption_map_list[i]}')
        overall_map = map if i == 0 else np.concatenate((overall_map,map),axis=0)
    return overall_map

def worker_function_create_gridding(
    t1,
    low,
    dataset,
    gridding_data,
    label_list,
    voxel_size,
    coefficients,
    coord_list,
    gridding_dir,
    args,
    full_iteration,
    store_paths,
    printing,
    afterfix,
    num_cls,
    gridding_method=2,
):
    len_data = len(gridding_data)
    arr_thetaphi = []
    arr_map = []

    up = low + len(gridding_data)
    shape = np.array(label_list.shape)

    for i, row in enumerate(gridding_data):
        theta, phi = row
        # scattering_vector = thetaphi_2_dials(theta, phi)
        arr_map.append(myframe_2_dials(thetaphi_2_myframe(theta, phi)))
        arr_thetaphi.append([theta, phi])
    arr_map = np.array(arr_map)

    full_iteration = 0
    label_list_c = python_2_c_3d(label_list)

    anacor_lib_cpu = ct.CDLL(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../src/ray_tracing_cpu.so"
        )
    )
    # anacor_lib_cpu = ct.CDLL( './ray_tracing.so' )s
    # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

    anacor_lib_cpu.ray_tracing_single_gridding.restype = ct.c_double
    anacor_lib_cpu.ray_tracing_single_gridding.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coord
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        ct.c_double,  # theta
        ct.c_double,  # phi
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # index
        ct.c_int,  # IsExp
    ]
    anacor_lib_cpu.ib_single_gridding.restype = ct.c_double
    anacor_lib_cpu.ib_single_gridding.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coord
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
        ct.c_int,  # IsExp
    ]
    print("gridding method is {}".format(gridding_method))
    assert gridding_method < 3
    absorption_map = []
    for i, row in enumerate(arr_map):
        if i == 0:
            print(
                "\033[92m C with {} cores is used for gridding \033[0m".format(
                    args.num_workers
                )
            )
        rotated_s1 = row
        theta, phi = dials_2_thetaphi((rotated_s1))
        ray_direction = dials_2_myframe(rotated_s1)
        absorption_row = []
        theta_1, phi_1 = 0, 0
        numbers_1 = (0, 0, 0, 0)
        for k, coord in enumerate(coord_list):
           
            if gridding_method == 1:
                # absorption = cal_rate(numbers_2, coefficients)
                if args.gridding_bisection:
                    absorption = anacor_lib_cpu.ib_single_gridding(coord,rotated_s1,theta,phi,voxel_size,coefficients,label_list_c,shape,full_iteration,
                    i,
                    num_cls,
                    1,
                )
                else:
                    absorption = anacor_lib_cpu.ray_tracing_single_gridding(
                        coord,
                        rotated_s1,
                        theta,
                        phi,
                        voxel_size,
                        coefficients,
                        label_list_c,
                        shape,
                        full_iteration,
                        i,
                        1,
                    )

            elif gridding_method == 2:
                # absorption = cal_rate(numbers_2, coefficients, exp=False)
                if args.gridding_bisection:
                    absorption = anacor_lib_cpu.ib_single_gridding(
                    coord,
                    rotated_s1,
                    theta,
                    phi,
                    voxel_size,
                    coefficients,
                    label_list_c,
                    shape,
                    full_iteration,
                    i,
                    num_cls,
                    0,
                )
                else:
                    absorption = anacor_lib_cpu.ray_tracing_single_gridding(
                        coord,
                        rotated_s1,
                        theta,
                        phi,
                        voxel_size,
                        coefficients,
                        label_list_c,
                        shape,
                        full_iteration,
                        i,
                        0,
                    )

            if args.DEBUG:
                face_2 = cube_face(coord, ray_direction, shape)

                path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list)  # 16
                numbers_2rt = cal_path_plus(path_2, voxel_size)  # 3.5s\
                absorptionrt = cal_rate(
                    (numbers_1 + numbers_2rt), coefficients, exp=False
                )
                diff_2 = (absorption - absorptionrt) / absorptionrt * 100
                if diff_2 > 0.01:
                    print("diff_2 is {}".format(diff_2))
                    pdb.set_trace()

            absorption_row.append(np.float32(absorption))
        absorption_map.append(absorption_row)
        if printing:
            print(
                "[{}/{}] gridding map theta: {:.4f}, phi: {:.4f}".format(
                    low + i, up, theta * 180 / np.pi, phi * 180 / np.pi
                )
            )
    #     # pdb.set_trace()
    # with open(f"{gridding_dir}/{dataset}_{afterfix}_{up}.json", "w") as fz:  # Pickling
    #     json.dump(absorption_map, fz, indent=2)
    np.save(f"{gridding_dir}/{dataset}_{afterfix}_{up}.npy", absorption_map)

def mp_create_gridding(
    t1,
    low,
    label_list,
    dataset,
    voxel_size,
    coefficients,
    coord_list,
    gridding_dir,
    args,
    offset,
    full_iteration,
    store_paths,
    printing,
    afterfix,
    num_cls,
    gridding_method,
    num_processes,
):
    theta_list = np.linspace(-180, 180, args.gridding_theta, endpoint=False)
    phi_list = np.linspace(-90, 90, args.gridding_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_list, phi_list)
    # theta_grid, phi_grid = np.meshgrid(phi_list , theta_list  )
    data = np.stack((theta_grid.ravel(), phi_grid.ravel()), axis=-1)
    gridding_data = data / 180 * np.pi
    len_data = len(gridding_data)
    up = low + len(gridding_data)
    gridding_data = gridding_data[low:up]
    # pdb.set_trace()
    processes = []
    if num_processes > 1:
        each_core = int(len_data // num_processes)
        data_copies = [label_list.copy() for _ in range(num_processes)]
        for i in range(num_processes):
            # Create a new process and pass it the data copy and result queue
            if i != num_processes - 1:
                process = mp.Process(
                    target=worker_function_create_gridding,
                    args=(
                        t1,
                        low + i * each_core,
                        dataset,
                        gridding_data[i * each_core : (i + 1) * each_core],
                        data_copies[i],
                        voxel_size,
                        coefficients,
                        coord_list,
                        gridding_dir,
                        args,
                        full_iteration,
                        store_paths,
                        printing,
                        afterfix,
                        num_cls,
                        gridding_method,
                    ),
                )

            else:
                process = mp.Process(
                    target=worker_function_create_gridding,
                    args=(
                        t1,
                        low + i * each_core,
                        dataset,
                        gridding_data[i * each_core :],
                        data_copies[i],
                        voxel_size,
                        coefficients,
                        coord_list,
                        gridding_dir,
                        args,
                        full_iteration,
                        store_paths,
                        printing,
                        afterfix,
                        num_cls,
                        gridding_method,
                    ),
                )

            processes.append(process)
        # pdb.set_trace()
        # Start all worker processes
        for process in processes:
            process.start()

        # Wait for all worker processes to finish
        for process in processes:
            process.join()
    else:
        worker_function_create_gridding(
            t1,
            low,
            dataset,
            gridding_data,
            label_list,
            voxel_size,
            coefficients,
            coord_list,
            gridding_dir,
            args,
            full_iteration,
            store_paths,
            printing,
            afterfix,
            num_cls,
            args.gridding_method,
        )

    with open(
        os.path.join(
            gridding_dir, "{}_time_create_{}.json".format(dataset, args.sampling_ratio)
        ),
        "w",
    ) as fz:
        json.dump(time.time() - t1, fz, indent=2)


def create_interpolation_gridding(
    theta_list, phi_list, data, interpolation_method="linear"
):
    interp_func = RegularGridInterpolator(
        (theta_list, phi_list), data, method=interpolation_method
    )

    return interp_func


def mp_interpolation_gridding(
    t1,
    low,
    abs_gridding,
    selected_data,
    label_list,
    voxel_size,
    coefficients,
    F,
    coord_list,
    omega_axis,
    axes_data,
    gridding_dir,
    args,
    offset,
    full_iteration,
    store_paths,
    printing,
    num_cls,
    num_processes,
    interpolation_method="linear",afterfix=''
):
    abs_gridding = np.array(abs_gridding)  # .astype(np.float32)
    try:
        abs_gridding = abs_gridding.reshape(
            (args.gridding_phi, args.gridding_theta, len(coord_list))
        )
    except:
        print("error in reshape")
        print("recreating the gridding")
        mp_create_gridding(t1, low, label_list,args.dataset,
                             voxel_size, coefficients,coord_list,
                             gridding_dir, args,
                             offset, full_iteration, store_paths, printing,afterfix, num_cls, args.gridding_method,num_processes)
        abs_gridding=stacking(gridding_dir,afterfix)
    #
    # theta_list = np.linspace(-180 , 180 ,
    #                          args.gridding_theta, endpoint=False) / 180 * np.pi
    # phi_list = np.linspace(-90 , 90 ,
    #                        args.gridding_phi , endpoint=False) / 180 * np.pi
    # pdb.set_trace()
    theta_extra_padding_num = int(np.round((abs_gridding.shape[1] // 6) / 2))
    phi_extra_padding_num = int(np.round(abs_gridding.shape[0] // 6 / 2))
    # add to top  and bottom
    bot_rows = abs_gridding[:, -theta_extra_padding_num:, :]
    top_rows = abs_gridding[:, :theta_extra_padding_num, :]
    abs_gridding = np.concatenate((bot_rows, abs_gridding, top_rows), axis=1)
    # add to left and right
    right_cols = abs_gridding[-phi_extra_padding_num:, :, :]
    left_cols = abs_gridding[:phi_extra_padding_num, :, :]
    abs_gridding = np.concatenate((right_cols, abs_gridding, left_cols), axis=0)

    abs_gridding = np.transpose(abs_gridding, (2, 0, 1))
    # abs_gridding = np.transpose(abs_gridding, (1, 0, 2))
    len_theta = int(args.gridding_theta + theta_extra_padding_num * 2)
    len_phi = int(args.gridding_phi + phi_extra_padding_num * 2)
    # add 1/6  is for the padding to avoid the interpolation error on boundary
    theta_list = (
        np.linspace(
            -np.round(180 * 7 / 6), np.round(180 * 7 / 6), len_theta, endpoint=False
        )
        / 180
        * np.pi
    )
    phi_list = (
        np.linspace(
            -np.round(90 * 7 / 6), np.round(90 * 7 / 6), len_phi, endpoint=False
        )
        / 180
        * np.pi
    )
    # theta_list =theta_list.astype(np.float32)
    # phi_list =phi_list.astype(np.float32)
    # interpolation_functions=[]
    # for k in range(len(coord_list)):
    #     interpolation_functions.append(create_interpolation_gridding( phi_list, theta_list,abs_gridding[k, :, :],interpolation_method))
    # interpolation_functions=np.array(interpolation_functions)
    # del abs_gridding
    # shm = shared_memory.SharedMemory(create=True, size=abs_gridding.nbytes)
    # shared_gridding = np.ndarray(
    #     abs_gridding.shape, dtype=abs_gridding.dtype, buffer=shm.buf
    # )

    # Copy data to the shared array
    # np.copyto(shared_gridding, abs_gridding)
    # shared_gridding = sharedctypes.RawArray('d', abs_gridding.size)
    # shared_gridding_np = np.frombuffer(shared_gridding, dtype=abs_gridding.dtype).reshape(abs_gridding.shape)
    # np.copyto(shared_gridding, abs_gridding)
    # pdb.set_trace()

    len_data = len(selected_data)
    # processes = []
    # if num_processes > 1:
    #     each_core = int(len_data // num_processes)
    #     for i in range(num_processes):
    #         # Create a new process and pass it the data copy and result queue
    #         if i != num_processes - 1:
    #             process = mp.Process(
    #                 target=interpolation_gridding,
    #                 args=(
    #                     t1,
    #                     low + i * each_core,
    #                     selected_data[i * each_core : (i + 1) * each_core],
    #                     label_list,
    #                     voxel_size,
    #                     coefficients,
    #                     F,
    #                     coord_list,
    #                     omega_axis,
    #                     axes_data,
    #                     gridding_dir,
    #                     args,
    #                     offset,
    #                     full_iteration,
    #                     store_paths,
    #                     printing,
    #                     num_cls,
    #                     interpolation_method,
    #                     phi_list,
    #                     theta_list,
    #                     abs_gridding,
    #                     len_theta,
    #                     len_phi,
    #                 ),
    #             )

    #         else:
    #             process = mp.Process(
    #                 target=interpolation_gridding,
    #                 args=(
    #                     t1,
    #                     low + i * each_core,
    #                     selected_data[i * each_core :],
    #                     label_list,
    #                     voxel_size,
    #                     coefficients,
    #                     F,
    #                     coord_list,
    #                     omega_axis,
    #                     axes_data,
    #                     gridding_dir,
    #                     args,
    #                     offset,
    #                     full_iteration,
    #                     store_paths,
    #                     printing,
    #                     num_cls,
    #                     interpolation_method,
    #                     phi_list,
    #                     theta_list,
    #                     abs_gridding,
    #                     len_theta,
    #                     len_phi,
    #                 ),
    #             )

    #         processes.append(process)
    #     # pdb.set_trace()
    #     # Start all worker processes
    #     for process in processes:
    #         process.start()

    #     # Wait for all worker processes to finish
    #     for process in processes:
    #         process.join()
        # shm.close()
        # shm.unlink()
    # else:
    interpolation_gridding(
            t1,
            low,
            selected_data,
            label_list,
            voxel_size,
            coefficients,
            F,
            coord_list,
            omega_axis,
            axes_data,
            gridding_dir,
            args,
            offset,
            full_iteration,
            store_paths,
            printing,
            num_cls,
            interpolation_method=interpolation_method,
            phi_list=phi_list,
            theta_list=theta_list,
            abs_gridding=abs_gridding,
            len_theta=len_theta,
            len_phi=len_phi,
            dtype=abs_gridding.dtype,
            shape=abs_gridding.shape,
        )


def interpolation_gridding(
    t1,
    low,
    selected_data,
    label_list,
    voxel_size,
    coefficients,
    F,
    coord_list,
    omega_axis,
    axes_data,
    gridding_dir,
    args,
    offset,
    full_iteration,
    store_paths,
    printing,
    num_cls,
    interpolation_method="nearest",
    phi_list=None,
    theta_list=None,
    abs_gridding=None,
    len_theta=None,
    len_phi=None,
    dtype=np.float64,
    shape=None,
):
    # existing_shm = shared_memory.SharedMemory(name=shared_gridding.name)
    # pdb.set_trace()
    # abs_gridding = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # abs_gridding = np.frombuffer(shared_gridding, dtype=np.float64).reshape(len(coord_list),len_phi, len_theta)
    if args.inter_method == "linear":
        interpolation_method = 2
    elif args.inter_method == "nearest":
        interpolation_method = 1
    up = low + len(selected_data)
    corr = []
    dict_corr = []
    arr_scattering = []
    arr_omega = []
    IsExp = 1
    xray = -np.array(axes_data[1]["direction"])
    shape = np.array(label_list.shape)
    # print(os.path.join(os.path.dirname( os.path.dirname( os.path.abspath(__file__))), './src/gridding_interpolation.so'))
    print("gridding_dir is {}".format(gridding_dir))
    lib = ct.CDLL(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "./src/gridding_interpolation.so",
        )
    )
    lib.interpolate.restype = ct.c_double
    # Define the argument types and return type of the function

    lib.interpolate.argtypes = [
        ct.c_int64,
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
    ]
    lib.nearest_neighbor_interpolate.restype = ct.c_double
    lib.nearest_neighbor_interpolate.argtypes = [
        ct.c_int64,
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float32),
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
    ]
    lib.nearest_neighbor_interpolate.restype = ct.c_double
    lib.nearest_neighbor_interpolate_overall.argtypes = [
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
        ct.c_int,  # num_workers
        ct.c_int, # IsExp
        np.ctypeslib.ndpointer(dtype=np.float64), #theta_list
        np.ctypeslib.ndpointer(dtype=np.float64), #phi_list
        np.ctypeslib.ndpointer(dtype=np.float32), #gridding data
        ct.c_int64, #nx
        ct.c_int64, #ny
        ct.c_double, #theta_min
        ct.c_double, #theta_max
        ct.c_double, #phi_min
        ct.c_double, #phi_max
        ct.c_int, #interpolation_method
        ]
    lib.nearest_neighbor_interpolate_overall.restype = ct.POINTER(ct.c_double)

    # Define the argument types and return type of the function

    lib.interpolate_single.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
    ]
    if args.openmp:
        for i, row in enumerate(selected_data):
            intensity = float(row["intensity.sum.value"])
            # all are in x, y , z in the origin dials file
            miller_index = row["miller_index"]

            scattering_vector = literal_eval(row["s1"])
            rotation_frame_angle = literal_eval(row["xyzobs.mm.value"])[2]
            rotation_frame_angle += offset / 180 * np.pi
            arr_scattering.append(scattering_vector)
            arr_omega.append(rotation_frame_angle)

        arr_scattering = np.array(arr_scattering)
        arr_omega = np.array(arr_omega)
        print(
                "\033[92m Openmp/C with {} cores is used for ray tracing \033[0m".format(args.num_workers))
        result_list = lib.nearest_neighbor_interpolate_overall(low,
                                                                up,
                                                                coord_list,
                                                                len(coord_list),
                                                                arr_scattering,
                                                                arr_omega,
                                                                xray,
                                                                omega_axis,
                                                                F,
                                                                len(selected_data),
                                                                voxel_size,
                                                                coefficients,
                                                                args.num_workers,
                                                                IsExp,
                                                                theta_list,
                                                                phi_list,
                                                                abs_gridding.flatten(),
                                                                len_theta,
                                                                len_phi,
                                                                theta_list.min(),
                                                                theta_list.max(),
                                                                phi_list.min(),
                                                                phi_list.max(),
                                                                interpolation_method,
                                                                )
        for i in range(len(selected_data)):
            corr.append(result_list[i])
        t2 = time.time()
        lib.free(result_list)
        
    else:
        for i, row in enumerate(selected_data):
            intensity = float(row["intensity.sum.value"])
            # all are in x, y , z in the origin dials file
            scattering_vector = literal_eval(row["s1"])
            miller_index = row["miller_index"]

            rotation_frame_angle = literal_eval(row["xyzobs.mm.value"])[2]
            rotation_frame_angle += offset / 180 * np.pi
            rotation_matrix_frame_omega = kp_rotation(omega_axis, rotation_frame_angle)

            kp_rotation_matrix = np.dot(rotation_matrix_frame_omega, F)
            total_rotation_matrix = np.transpose(kp_rotation_matrix)
            xray = -np.array(axes_data[1]["direction"])

            xray = np.dot(total_rotation_matrix, xray)
            rotated_s1 = np.dot(total_rotation_matrix, scattering_vector)

            theta, phi = dials_2_thetaphi(rotated_s1)
            theta_1, phi_1 = dials_2_thetaphi(xray, L1=True)

            ray_direction = dials_2_myframe(rotated_s1)
            xray_direction = dials_2_myframe(xray)

            # grid=abs_gridding[:,:,k].flatten()
            # pdb.set_trace()
            result = lib.nearest_neighbor_interpolate(
                np.int64(len(coord_list)),
                theta_list,
                phi_list,
                abs_gridding.flatten(),
                len_theta,
                len_phi,
                theta_1,
                phi_1,
                theta,
                phi,
                theta_list.min(),
                theta_list.max(),
                phi_list.min(),
                phi_list.max(),
            )

            if args.DEBUG:
                absorp = np.empty(len(coord_list))
                absorprt = np.empty(len(coord_list))
                for k, coord in enumerate(coord_list):
                    face_1 = cube_face(coord, xray_direction, shape, L1=True)
                    face_2 = cube_face(coord, ray_direction, shape)
                    path_1 = cal_coord(
                        theta_1, phi_1, coord, face_1, shape, label_list
                    )  # 37
                    path_2 = cal_coord(
                        theta, phi, coord, face_2, shape, label_list
                    )  # 16

                    numbers_1rt = cal_path_plus(path_1, voxel_size)  # 3.5s
                    numbers_2rt = cal_path_plus(path_2, voxel_size)  # 3.5s\

                    absorptionrt = cal_rate((numbers_1rt + numbers_2rt), coefficients)
                    absorprt[k] = absorptionrt

                    # inter_1 = interpolation_v1(theta_1, phi_1, theta_list, phi_list, abs_gridding[k, :, :])
                    # inter_1 =create_interpolation_gridding(theta_list, phi_list, abs_gridding[:, :, k],interpolation_method)(np.array([theta_1, phi_1]))
                    # inter_2 =create_interpolation_gridding(theta_list, phi_list, abs_gridding[:, :, k],interpolation_method)(np.array([theta, phi]))
                    grid = abs_gridding[k, :, :].flatten()
                    inter_1 = lib.interpolate_single(
                        theta_list,
                        phi_list,
                        grid,
                        len_theta,
                        len_phi,
                        theta_1,
                        phi_1,
                    )
                    inter_2 = lib.interpolate_single(
                        theta_list,
                        phi_list,
                        grid,
                        len_theta,
                        len_phi,
                        theta,
                        phi,
                    )
                    absorption = np.exp(-(inter_1 + inter_2))
                    absorp[k] = absorption
                # inter_1 =interpolation_functions[k](np.array([ phi_1,theta_1]))
                # inter_2 =interpolation_functions[k](np.array([ phi,theta]))
                # try:

                # pdb.set_trace()
                # except:
                #     print('error in c')
                #     pdb.set_trace()
                # inter_2 = interpolation_v1(
                #     theta, phi, theta_list, phi_list, abs_gridding[k, :, :])

            t2 = time.time()
            if args.DEBUG:
                # result = absorp.mean()
                diff_2 = (result - absorprt.mean()) / absorprt.mean() * 100
                if diff_2 > 1:
                    print("diff_2 is {}".format(diff_2))

                    pdb.set_trace()

            # if i == 1000:
            #     print('time spent {}'.format(t2 - t1))
            #     pdb.set_trace()
            if printing:
                print(
                    "[{}/{}] theta: {:.4f}, phi: {:.4f} , time: {:.4f},  absorption: {:.4f}".format(
                        low + i,
                        low + len(selected_data),
                        theta * 180 / np.pi,
                        phi * 180 / np.pi,
                        t2 - t1,
                        result,
                    )
                )

            corr.append(result)
            # print( 'it spends {}'.format( t2 - t1 ) )
            # dict_corr.append(
            #     {
            #         "index": low + i,
            #         "miller_index": miller_index,
            #         "intensity": intensity,
            #         "corr": result,
            #         "theta": theta * 180 / np.pi,
            #         "phi": phi * 180 / np.pi,
            #         "theta_1": theta_1 * 180 / np.pi,
            #         "phi_1": phi_1 * 180 / np.pi,
            #     }
            # )
            if i % 1000 == 1:
                with open(
                    os.path.join(
                        args.save_dir, "{}_refl_{}.json".format(args.dataset, up)
                    ),
                    "w",
                ) as fz:  # Pickling
                    json.dump(corr, fz, indent=2)

                print("[{} /{} ]it spends {}".format(low + i, up, t2 - t1))
    
    with open(
            os.path.join(args.save_dir, "{}_refl_{}.json".format(args.dataset, up)), "w"
        ) as fz:  # Pickling
            json.dump(corr, fz, indent=2)


    with open(
        os.path.join(args.save_dir, "{}_time_{}.json".format(args.dataset, up)), "w"
    ) as f1:  # Pickling
        json.dump(t2 - t1, f1, indent=2)
    print("{} ({} ) process is Finish!!!!".format(os.getpid(), up))


def memorylog():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024} MB")
    return mem_info.rss


def coord_transform(theta, phi):
    if theta < 0:
        theta += 2 * np.pi
    if phi < 0:
        phi += np.pi
    return theta, phi


def y_concat(angles):
    theta_max, phi_max = angles.shape
    shift_theta = theta_max // 2
    shift_phi = phi_max // 2
    first_half_rows = angles[:shift_theta, :]
    rest_of_rows = angles[shift_theta:, :]
    angles = np.concatenate((rest_of_rows, first_half_rows), axis=0)

    return angles


def x_concat(angles):
    theta_max, phi_max = angles.shape
    shift_theta = theta_max // 2
    shift_phi = phi_max // 2
    first_half_cols = angles[:, :shift_phi]
    rest_of_cols = angles[:, shift_phi:]
    angles = np.concatenate((rest_of_cols, first_half_cols), axis=1)
    return angles


def unit_test_sphere_transform(detector_gridding, theta_grid, phi_grid):
    ap2 = spheretransformation(detector_gridding)
    theta_grid_2 = y_concat(theta_grid)
    phi_grid_2 = x_concat(phi_grid)
    theta_grid_2[theta_grid_2 < 0] += 2 * np.pi
    phi_grid_2[phi_grid_2 < 0] += np.pi
    df = []
    for i, phi_row in enumerate(phi_grid):
        for j, phi in enumerate(phi_row):
            theta = theta_grid[i][j]
            ap_1 = detector_gridding[i][j]
            theta_2, phi_2 = coord_transform(theta, phi)
            # j_index= np.where(np.abs(phi_grid_2-phi_2)<0.01)[1][0]
            # i_index=np.where(np.abs(theta_grid_2- theta_2)<0.01)[0][0]
            j_index = np.argmin(np.abs(phi_grid_2 - phi_2), axis=1)[0]
            i_index = np.argmin(np.abs(theta_grid_2 - theta_2), axis=0)[0]
            ap_2 = ap2[i_index][j_index]
            df.append(np.abs(ap_2 - ap_1))
    # pdb.set_trace()


def spheretransformation(absorption_map):
    if len(absorption_map.shape) == 2:
        theta_max, phi_max = absorption_map.shape
        shift_theta = theta_max // 2
        shift_phi = phi_max // 2
        first_half_rows = absorption_map[1:shift_theta, :]
        rest_of_rows = absorption_map[shift_theta:, :]
        absorption_map = np.vstack((rest_of_rows, first_half_rows))
        first_half_cols = absorption_map[:, 1:shift_phi]
        rest_of_cols = absorption_map[:, shift_phi:]
        ap = np.hstack((rest_of_cols, first_half_cols))

    elif len(absorption_map.shape) == 3:
        theta_max, phi_max, voxels_number = absorption_map.shape
        shift_theta = theta_max // 2
        shift_phi = phi_max // 2
        first_half_rows = absorption_map[:shift_theta, :, :]
        rest_of_rows = absorption_map[shift_theta:, :, :]
        absorption_map = np.concatenate((rest_of_rows, first_half_rows), axis=0)
        first_half_cols = absorption_map[:, :shift_phi, :]
        rest_of_cols = absorption_map[:, shift_phi:, :]
        ap = np.concatenate((rest_of_cols, first_half_cols), axis=1)

    return ap


def thicken_grid(absorption_map, thickness=1):
    if len(absorption_map.shape) == 2:
        ap = np.concatenate(
            (
                absorption_map[-thickness:, :],
                absorption_map,
                absorption_map[:thickness, :],
            ),
            axis=0,
        )
        ap2 = np.concatenate((ap[:, -thickness:], ap, ap[:, :thickness]), axis=1)
    else:
        ap = np.concatenate(
            (
                absorption_map[-thickness:, :, :],
                absorption_map,
                absorption_map[:thickness, :, :],
            ),
            axis=0,
        )
        ap2 = np.concatenate((ap[:, -thickness:, :], ap, ap[:, :thickness, :]), axis=1)
    return ap2


# def gridding_3D(label_list,coord_list,voxel_size,
#                 coefficients, arr_map,gridding_method):
#     shape=label_list.shape
#     assert gridding_method < 3
#     absorption_map=[]
#     t1=time.time()
#     label_list_c=python_2_c_3d(label_list)
#     for i, row in enumerate(arr_map):
#             rotated_s1 = row
#             theta, phi = dials_2_thetaphi((rotated_s1))
#             absorption_row=[]
#             ray_direction = dials_2_myframe( rotated_s1 )
#             for k , coord in enumerate( coord_list ) :

#                 face_2 = cube_face( coord , ray_direction , shape )
#                 path_2 = cal_coord( theta , phi , coord , face_2 , shape , label_list )  # 16
#                 numbers_2 = cal_path_plus( path_2 , voxel_size )
#                 if gridding_method == 1:
#                     absorption = cal_rate( numbers_2 , coefficients )
#                     # absorption_map[i][k] = absorption
#                     absorption_row.append(absorption)
#                 elif gridding_method == 2:
#                     absorption = cal_rate( numbers_2 , coefficients , exp = False )
#                     # absorption_map[i][k]=absorption
#                     absorption_row.append(absorption)

#             absorption_map.append(absorption_row)


#     t2 = time.time( )
#     print( '[{}]/[{}]'.format( i , len( arr_map ) ) )
#     print( 'time spent {}'.format( t2 - t1 ) )
#     return absorption_map
# np.save( "./gridding/{}_gridding_{}_{}_{}.npy".format( dataset,theta_num, phi_num,gridding_method) ,
#             absorption_map )

# def gridding_2D(absorption_map,label_list,coordinate_list,voxel_size,
#                 coefficients, phi_grid,theta_grid,gridding_method):
#     shape=label_list.shape
#     assert gridding_method == 3
#     t1=time.time()

#     for i, phi_row in enumerate(phi_grid):
#         for j, phi in enumerate(phi_row):
#             theta=theta_grid[i][j]
#             absorp= np.empty( len( coordinate_list ) )
#             for k , index in enumerate( coordinate_list ) :
#                 coord = crystal_coordinate[index]
#                 face_2 = which_face_2( coord , shape , theta , phi )
#                 path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
#                 numbers_2 = cal_num( path_2 , voxel_size )
#                 absorption = cal_rate( numbers_2 , coefficients )
#                 absorp[k] = absorption

#             absorption_map[i][j]= absorp.mean()
#         t2 = time.time( )
#         print( '[{}]/[{}],[{}]/[{}],[{}]/[{}]'.format( i , len( phi_grid ) , j , len( phi_row ) , k ,
#                                                        len( coordinate_list ) ) )
#         print( 'time spent {}'.format( t2 - t1 ) )
# return absorption_map


def gridding(
    label_list, coordinate_list, voxel_size, coefficients, phi_grid, theta_grid
):
    shape = label_list.shape

    m1 = memorylog()
    if gridding_method == 3:
        absorption_map = np.zeros(phi_grid.shape)
    else:
        absorption_map = np.zeros(phi_grid.shape + (len(coordinate_list),))
    m2 = memorylog()
    print("memory usage on the absorption map is {}".format((m2 - m1) / 1024 / 1024))
    t1 = time.time()
    for i, phi_row in enumerate(phi_grid):
        for j, phi in enumerate(phi_row):
            theta = theta_grid[i][j]
            absorp = np.empty(len(coordinate_list))
            for k, index in enumerate(coordinate_list):
                coord = crystal_coordinate[index]

                face_2 = which_face_2(coord, shape, theta, phi)
                path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list)
                numbers_2 = cal_num(path_2, voxel_size)
                if gridding_method == 1:
                    absorption = cal_rate(numbers_2, coefficients)
                    absorption_map[i][j][k] = absorption
                elif gridding_method == 2:
                    absorption = cal_rate(numbers_2, coefficients, exp=False)
                    absorption_map[i][j][k] = absorption
                elif gridding_method == 3:
                    absorption = cal_rate(numbers_2, coefficients)
                    absorp[k] = absorption
            if gridding_method == 3:
                absorption_map[i][j] = absorp.mean()

        t2 = time.time()
        print(
            "[{}]/[{}],[{}]/[{}],[{}]/[{}]".format(
                i, len(phi_grid), j, len(phi_row), k, len(coordinate_list)
            )
        )
        print("time spent {}".format(t2 - t1))
        np.save(
            "./gridding/{}_gridding_{}_{}_{}.npy".format(
                dataset, theta_num, phi_num, gridding_method
            ),
            absorption_map,
        )
