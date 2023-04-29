import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
try:
    from AnACor.RayTracing import RayTracingBasic,kp_rotation
except:
    from RayTracing import RayTracingBasic,kp_rotation
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

    parser.add_argument(
        "--offset" ,
        type = float ,
        default = 0 ,
        help = "orientation offset" ,
    )

    parser.add_argument(
        "--dataset" ,
        type = str ,
        default = 16846 ,
        help = "1 is true, 0 is false" ,
    )
    parser.add_argument(
        "--model-storepath" ,
        type = str ,
        required = True ,
        help = "full model path" ,
    )
    parser.add_argument(
        "--store-dir" ,
        type = str ,
        default = "./" ,
        help = "full storing path" ,
    )
    parser.add_argument(
        "--refl-path" ,
        type = str ,
        required = True ,
        help = "full reflection path" ,
    )
    parser.add_argument(
        "--expt-path" ,
        type = str ,
        required = True ,
        help = "full experiment path" ,
    )
    parser.add_argument(
        "--liac" ,
        type = float ,
        required = True ,
        help = "abs of liquor" ,
    )
    parser.add_argument(
        "--loac" ,
        type = float ,
        required = True ,
        help = "abs of loop" ,
    )
    parser.add_argument(
        "--crac" ,
        type = float ,
        required = True ,
        help = "abs of crystal" ,
    )
    parser.add_argument(
        "--buac" ,
        type = float ,
        required = True ,
        help = "abs of other component" ,
    )
    parser.add_argument(
        "--sampling-num" ,
        type = int ,
        default = 5000 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--auto-sampling" ,
        type = str2bool ,
        default = False,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--full-iteration" ,
        type = int ,
        default = 0 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--pixel-size" ,
        type = float ,
        default = 0.3 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--pixel-size-x" ,
        type = float ,
        default = 0.3 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--pixel-size-y" ,
        type = float ,
        default = 0.3 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--pixel-size-z" ,
        type = float ,
        default = 0.3 ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--by-c" ,
        type = str2bool ,
        default = False,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--slicing" ,
        type = str ,
        default = 'z' ,
        help = "pixel size of tomography" ,
    )
    parser.add_argument(
        "--num-workers" ,
        type = int,
        default = 4 ,
        help = "number of workers" ,
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
    
    save_dir = os.path.join(  args.store_dir, '{}_save_data'.format( dataset ) )
    result_path  =os.path.join(  save_dir,'ResultData','absorption_factors')
    refl_dir = os.path.join(  save_dir,'ResultData','reflections')

    try:
        os.makedirs(save_dir)
        os.makedirs(result_path)
        os.makedirs(refl_dir)
    except:
        pass

    if args.model_storepath == 'None':
        models_list=[]
        for file in os.listdir(save_dir ):
              if dataset in file and ".npy" in file:
                  models_list.append(file)

        if len(models_list) ==1:
            model_path = os.path.join( save_dir, models_list[0] )
        elif len(models_list) ==0:
            raise RuntimeError("\n There are no 3D models of sample {} in this directory \n  Please create one by command python setup.py \n".format(dataset))
        else:
            raise RuntimeError("\n There are many 3D models of sample {} in this directory \n  Please delete the unwanted models \n".format(dataset))
    else:
        model_path=args.model_storepath
    
    args.model_storepath= model_path
    args.save_dir=result_path

    algorithm = RayTracingBasic(args)
    algorithm.mp_run(printing=True)
    #algorithm.run()
    # pdb.set_trace()

    # for file in os.listdir(save_dir):
    #     if '.json' in file:
    #         if 'expt' in file:
    #             expt_filename=os.path.join(save_dir,file)
    #         if 'refl' in file:
    #             refl_filename = os.path.join(save_dir,file)
    # try:
    #     with open(expt_filename) as f2:
    #         axes_data = json.load(f2)
    #     print( "experimental data is loaded... \n" )
    #     with open(refl_filename) as f1:
    #         data = json.load(f1)
    #     print( "reflection table is loaded... \n" )
    # except:
    #     try:
    #         with open(args.expt_filename) as f2:
    #             axes_data = json.load(f2)
    #         print( "experimental data is loaded... \n" )
    #         with open(args.refl_filename) as f1:
    #             data = json.load(f1)
    #         print( "reflection table is loaded... \n" )
    #     except:
    #         raise  RuntimeError('no reflections or experimental files detected'
    #                             'please use --refl_filename --expt-filename to specify')


    # label_list = np.load(model_path).astype(np.int8)

    # print("3D model is loaded... \n")
    # mu_cr = args.crac  # (unit in mm-1) 16010
    # mu_li = args.liac
    # mu_lo = args.loac
    # mu_bu = args.buac

    # t1 = time.time()

    # low = args.low
    # up = args.up

    # if up == -1:
    #     selected_data = data[low:]
    # else:
    #     selected_data = data[low:up]
    # print('The total size of this calculation is {}'.format(len(selected_data)))
    # del data
    # coefficients = mu_li, mu_lo, mu_cr, mu_bu
    # axes = axes_data[0]
    # kappa_axis = np.array( axes["axes"][1] )
    # kappa = axes["angles"][1] / 180 * np.pi
    # kappa_matrix = kp_rotation( kappa_axis , kappa )
    # phi_axis = np.array( axes["axes"][0] )
    # phi = axes["angles"][0] / 180 * np.pi
    # phi_matrix = kp_rotation( phi_axis , phi )
    # # https://dials.github.io/documentation/conventions.html#equation-diffractometer
    # omega_axis = np.array( axes["axes"][2] )
    # F = np.dot( kappa_matrix , phi_matrix )  # phi is the most intrinsic rotation, then kappa

    # algorithm = RayTracingBasic(selected_data,label_list,coefficients,sampling_threshold=args.sampling)
    # corr = []
    # dict_corr = []  
    
    # for i in range(len(selected_data)):
    #     row = selected_data[i]
    #     intensity = float(row['intensity.sum.value'])
    #     scattering_vector = literal_eval(row['s1'])  # all are in x, y , z in the origin dials file
    #     miller_index = row['miller_index']
    #     lp = row['lp']
    #     rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]
    #     rotation_frame_angle+=args.offset/180*np.pi

    #     rotation_matrix_frame_omega = kp_rotation(omega_axis, rotation_frame_angle)

    #     total_rotation_matrix = np.dot(rotation_matrix_frame_omega,F)
    #     total_rotation_matrix = np.transpose(total_rotation_matrix)

    #     xray = -np.array(axes_data[1]["direction"])
    #     xray=np.dot(total_rotation_matrix ,xray)
    #     rotated_s1 = np.dot(total_rotation_matrix, scattering_vector)


    #     if args.store_lengths:
    #         absorption_factor, path_length_arr_single  = algorithm.run(xray , rotated_s1)
    #     else:
    #         absorption_factor = algorithm.run(xray , rotated_s1 )
            
    #     print( '[{}/{}] rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,low + len(selected_data ) ,
    #                                                             rotation_frame_angle * 180 / np.pi ,
    #                                                             absorption_factor) )
    #     corr.append(absorption_factor)

    #     t2 = time.time()
    #     if args.store_lengths:
    #       if i == 0 :
    #           path_length_arr = np.expand_dims( path_length_arr_single , axis = 0 )
    #       else :
    #           path_length_arr = np.concatenate(
    #               (path_length_arr , np.expand_dims( path_length_arr_single , axis = 0 )) , axis = 0 )
                  
    #     print('it spends {}'.format(t2 - t1))
    #     dict_corr.append({'index': low + i, 'miller_index': miller_index,
    #                       'intensity': intensity, 'corr': absorption_factor, 'lp': lp})
    #     if i % 500 == 1:
    #         if args.store_lengths:
    #             np.save( os.path.join(refl_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
    #         with open(os.path.join(result_path , "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
    #             json.dump(corr, fz, indent=2)

    #         with open(os.path.join(result_path , "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
    #             json.dump(dict_corr, f1, indent=2)
    # if args.store_lengths:
    #       np.save( os.path.join(refl_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
    # with open(os.path.join(result_path , "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
    #     json.dump(corr, fz, indent=2)

    # with open(os.path.join(result_path , "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
    #     json.dump(dict_corr, f1, indent=2)
    # print('Finish!!!!')

if __name__ == '__main__':
    main()