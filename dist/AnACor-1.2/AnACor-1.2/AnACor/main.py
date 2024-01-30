import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from AnACor.RayTracing import RayTracingBasic
from dials.util.filter_reflections import *
from dials.algorithms.scaling.scaler_factory import *
from dials.array_family import flex
from dxtbx.serialize import load
# ===========================================
#        Parse the argument
# ===========================================

def set_parser():
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
        "--store-dir",
        type=str,
        default = "./",
        help="the store directory ",
    )
    parser.add_argument(
        "--dataset",
        type=str,required=True,
        help="dataset number default is 13304",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0,
        help="the orientation offset",
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=2000,
        help="sampling for picking crystal point to calculate",
    )

    parser.add_argument(
        "--store-lengths",
        type=bool,
        default=False,
        help="whether store the path lengths to calculate with different absorption coefficients",
    )


    parser.add_argument(
        "--crac",
        type=float, required=True,
        help="the absorption coefficient of the crystal and it is needed",
    )
    parser.add_argument(
        "--loac",
        type=float, required=True,
        help="the absorption coefficient of the loop and it is needed",
    )
    parser.add_argument(
        "--liac",
        type=float, required=True,
        help="the absorption coefficient of the liquor and it is needed",
    )
    parser.add_argument(
        "--buac",
        type=float, default = 0,
        help="the absorption coefficient of the bubble and it is not necessarily needed",
    )
    parser.add_argument(
        "--refl-filename",
        type=str,
        required=True,
        help="the filenames of the reflection table",
    )
    parser.add_argument(
        "--expt-filename",
        type=str,
        default='',
        help="the filenames of the experimental table",
    )

    global args
    args = parser.parse_args()
    return args


def main():
    args=set_parser()
    print("\n==========\n")
    print("start AAC")
    print("\n==========\n")
    dataset = args.dataset
    result_path = os.path.join( args.store_dir, 'ResultData' )
    data_dir = os.path.join( result_path, '{}_save_data'.format( dataset ) )
    save_dir =os.path.join(  data_dir,'absorption_factors')
    refl_dir = os.path.join(  data_dir,'reflections')
#    if args.vflip :
#        model_name = './{}_tomobar_cropped_f.npy'.format( dataset )
#    else :
#        model_name = './{}_tomobar_cropped.npy'.format( dataset )
    models_list=[]
    for file in os.listdir(result_path ):
          if dataset in file and ".npy" in file:
              models_list.append(file)

    if len(models_list) ==1:
        model_path = os.path.join( result_path , models_list[0] )
    elif len(models_list) ==0:
        raise RuntimeError("\n There are no 3D models of sample {} in this directory \n  Please create one by command python setup.py \n".format(dataset))
    else:
        raise RuntimeError("\n There are many 3D models of sample {} in this directory \n  Please delete the unwanted models \n".format(dataset))
    
    filename=os.path.basename(args.refl_filename)
    reflections_table= "rejected_" + str(dataset) +"_"+ filename
    data = flex.reflection_table.from_file(os.path.join( refl_dir ,reflections_table) )
    print("reflection table is loaded... \n")
    expt_table=load.experiment_list(os.path.join( refl_dir ,args.expt_filename), check_format=False)[0]
    goniometer = expt_table.goniometer.to_dict()
    print("experimental data is loaded... \n")
    pdb.set_trace()
    label_list = np.load(model_path).astype(np.int8)

    print("3D model is loaded... \n")
    mu_cr = args.crac  # (unit in mm-1) 16010
    mu_li = args.liac
    mu_lo = args.loac
    mu_bu = args.buac

    t1 = time.time()
    # with open(refl_filaname) as f1:
    #     data = json.load(f1)
    

    low = args.low
    up = args.up

    if up == -1:
        selected_data = data[low:]
    else:
        selected_data = data[low:up]
    print('The total size of this calculation is {}'.format(len(selected_data)))
    del data
    coefficients = mu_li, mu_lo, mu_cr, mu_bu
    algorithm = RayTracingBasic(selected_data,label_list,coefficients,sampling=args.sampling)
    corr = []
    dict_corr = []  
    
    for i in range(len(selected_data)):
        row = selected_data[i]
        intensity = float(row['intensity.sum.value'])
        #scattering_vector = literal_eval(row['s1'])  # all are in x, y , z in the origin dials file
        scattering_vector = row['s1']
        miller_index = row['miller_index']
        lp = row['lp']
        
        #rotation_frame_angle = literal_eval( row['xyzobs.mm.value'] )[2]
        rotation_frame_angle =  row['xyzobs.mm.value'][2]
        rotation_frame_angle += args.offset
        if rotation_frame_angle < 0 :
            rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2 * np.pi :
            rotation_frame_angle = rotation_frame_angle - 2 * np.pi

        assert rotation_frame_angle <= 2 * np.pi

        #  rotate the x-ray beam about x-axis omega degree clockwisely
        rotation_matrix_frame = np.array( [[1 , 0 , 0] ,
                                           [0 , np.cos( rotation_frame_angle ) , np.sin( rotation_frame_angle )] ,
                                           [0 , -np.sin( rotation_frame_angle ) , np.cos( rotation_frame_angle )]] )

        rotated_s1 = np.dot( rotation_matrix_frame , scattering_vector )
        xray = np.array( [0 , 0 , -1] )
        xray = np.dot( rotation_matrix_frame , xray )       
        if args.store_lengths:
            absorption_factor, path_length_arr_single  = algorithm.run(xray , rotated_s1,store_lengths=args.store_lengths )
        else:
            absorption_factor = algorithm.run(xray , rotated_s1 )
            
        print( '[{}/{}] rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,low + len(selected_data ) ,
                                                                rotation_frame_angle * 180 / np.pi ,
                                                                absorption_factor) )
        corr.append(absorption_factor)

        t2 = time.time()
        if args.store_lengths:
          if i == 0 :
              path_length_arr = np.expand_dims( path_length_arr_single , axis = 0 )
          else :
              path_length_arr = np.concatenate(
                  (path_length_arr , np.expand_dims( path_length_arr_single , axis = 0 )) , axis = 0 )
                  
        print('it spends {}'.format(t2 - t1))
        dict_corr.append({'index': low + i, 'miller_index': miller_index,
                          'intensity': intensity, 'corr': absorption_factor, 'lp': lp})
        if i % 500 == 1:
            if args.store_lengths:
                np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
            with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
                json.dump(corr, fz, indent=2)

            with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
                json.dump(dict_corr, f1, indent=2)
    if args.store_lengths:
          np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
    with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
        json.dump(corr, fz, indent=2)

    with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
        json.dump(dict_corr, f1, indent=2)
    print('Finish!!!!')
