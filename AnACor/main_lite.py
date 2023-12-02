import os
import json
import time
import pdb
import numpy as np
import ctypes as ct
import multiprocessing as mp
# from dials.array_family import flex
from ast import literal_eval
from multiprocessing import Pool
try:
    from utils.utils_rt import *
    from utils.utils_ib import *
    from utils.utils_gridding import mp_create_gridding,mp_interpolation_gridding
    from utils.utils_os import stacking,python_2_c_3d,kp_rotation
    from utils.utils_mp import *
    from utils.utils_resolution import model3D_resize
except:
    from AnACor.utils.utils_rt import *
    from AnACor.utils.utils_ib import *
    from AnACor.utils.utils_gridding import mp_create_gridding,mp_interpolation_gridding
    from AnACor.utils.utils_os import stacking,python_2_c_3d,kp_rotation
    from AnACor.utils.utils_mp import *
    from AnACor.utils.utils_resolution import model3D_resize

from param import set_parser
# try:
#     from AnACor.RayTracing import RayTracingBasic,kp_rotation
# except:
#     from RayTracing import RayTracingBasic,kp_rotation
# from dials.util.filter_reflections import *
# from dials.algorithms.scaling.scaler_factory import *
# from dials.array_family import flex
# from dxtbx.serialize import load


def create_directory(directory_path):
    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass



def main():
    global args
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
    voxel_size = np.array([args.pixel_size_z * 1e-3,
                           args.pixel_size_y * 1e-3,
                           args.pixel_size_x * 1e-3])
    label_list = np.load(args.model_storepath).astype(np.int8)
    if args.resolution_voxel_size is not None:

        factor=args.pixel_size_x /  args.resolution_voxel_size
        factors=[factor, factor,factor ]
        label_list = model3D_resize(label_list, factors)
        print("model is resized to voxel size of {}".format(args.resolution_voxel_size))
        voxel_size = np.array([args.resolution_voxel_size * 1e-3,
                           args.resolution_voxel_size * 1e-3,
                           args.resolution_voxel_size * 1e-3])

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
        afterfix=f'gridding_{args.sampling_ratio}_{args.gridding_theta}_{args.gridding_phi}'
        print(os.path.join(os.path.dirname( os.path.abspath(__file__)), './src/gridding_interpolation.so'))
        lib = ct.CDLL(os.path.join(os.path.dirname( os.path.abspath(__file__)), './src/gridding_interpolation.so'))
        abs_gridding=stacking(gridding_dir,afterfix)
        # abs_gridding=stacking(gridding_dir,'gridding')
        


 
        if abs_gridding is None:
            print('gridding map is not found')
            print('creating gridding map...')
            mp_create_gridding(t1, low, label_list,dataset,
                             voxel_size, coefficients,coord_list,
                             gridding_dir, args,
                             offset, full_iteration, store_paths, printing,afterfix, num_cls, args.gridding_method,num_processes)
            print('gridding map is finished and created')
            abs_gridding=stacking(gridding_dir,afterfix)
            t1 = time.time()
        print('Loading gridding map')
        mp_interpolation_gridding(t1, low,  abs_gridding, select_data, label_list,
                          voxel_size, coefficients, F, coord_list,
                          omega_axis, axes_data, gridding_dir, args,
                          offset, full_iteration, store_paths, printing, num_cls,num_processes,args.interpolation_method)

    else:


        # Create a list of 48 data copies

        # Create a queue to store the results from each worker process
        # pdb.set_trace()
        # Create a list of worker processes
        processes = []
        if args.gpu is True:
            num_processes = 1
        if args.openmp is True:
            num_processes = 1

        if num_processes > 1:

            each_core = int(len_data//num_processes)
            data_copies = [label_list.copy() for _ in range(num_processes)]
            for i in range(num_processes):
                # Create a new process and pass it the data copy and result queue
                if i != num_processes-1:
                    if args.absorption_map is True:
                        process = mp.Process(target=worker_function_am,
                                            args=(t1, low+i*each_core, dataset,
                                                map_data[i*each_core:(i+1)
                                                            * each_core], select_data,  data_copies[i],
                                                voxel_size, coefficients, F, coord_list,
                                                omega_axis, axes_data, args.save_dir, args,
                                                offset, full_iteration, store_paths, printing, num_cls))
                    else:
                        process = mp.Process(target=worker_function,
                                            args=(t1, low+i*each_core, dataset,
                                                select_data[i*each_core:(i+1)
                                                            * each_core], data_copies[i],
                                                voxel_size, coefficients, F, coord_list,
                                                omega_axis, axes_data, args.save_dir, args,
                                                offset, full_iteration, store_paths, printing, num_cls))
                    # worker_function()
                else:
                    if args.absorption_map is True:
                        process = mp.Process(target=worker_function_am,
                                            args=(t1, low+i*each_core, dataset,
                                                map_data[i*each_core:], select_data,  data_copies[i],
                                                voxel_size, coefficients, F, coord_list,
                                                omega_axis, axes_data, args.save_dir, args,
                                                offset, full_iteration, store_paths, printing, num_cls))
                    else:
                        process = mp.Process(target=worker_function,
                                            args=(t1, low+i*each_core, dataset,
                                                select_data[i *
                                                            each_core:], data_copies[i],
                                                voxel_size, coefficients, F, coord_list,
                                                omega_axis, axes_data, args.save_dir, args,
                                                offset, full_iteration, store_paths, printing, num_cls))

                processes.append(process)
            # pdb.set_trace()
            # Start all worker processes
            for process in processes:
                process.start()

            # Wait for all worker processes to finish
            for process in processes:
                process.join()
        else:
            if args.absorption_map is True:
                worker_function_am(t1, low,  dataset, map_data, select_data, label_list,
                                voxel_size, coefficients, F, coord_list,
                                omega_axis, axes_data, args.save_dir, args,
                                offset, full_iteration, store_paths, printing, num_cls)
            else:
                worker_function(t1, low,  dataset, select_data, label_list,
                                voxel_size, coefficients, F, coord_list,
                                omega_axis, axes_data, args.save_dir, args,
                                offset, full_iteration, store_paths, printing, num_cls)





if __name__ == '__main__':
    main()
