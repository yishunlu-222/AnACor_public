import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from RayTracing import RayTracingBasic
from image_process import Image2Model
from dials.util.filter_reflections import *
from dials.algorithms.scaling.scaler_factory import *
from dials.array_family import flex
# ===========================================
#        Parse the argument
# ===========================================

parser = argparse.ArgumentParser(description="multiprocessing for batches")

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



global args
args = parser.parse_args()


def ada_sampling(crystal_coordinate ,threshold=10000):
    num=len(crystal_coordinate)
    sampling=1000
    result = num
    while result >threshold:
        result = num/sampling
        sampling+=100
 
    return sampling
    

if __name__ == "__main__":
#    print("\n==========\n")
#    print("adaptive sampling calculating")
#    print("\n==========\n")
    dataset = args.dataset
    result_path = os.path.join( args.store_dir, 'ResultData' )
    save_dir = os.path.join( result_path, '{}_save_data'.format( dataset ) )
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
    
    label_list = np.load(model_path).astype(np.int8)
    #print("3D model is loaded... \n")
    rate_list = {'li' : 1 , 'lo' : 2 , 'cr' : 3 , 'bu' : 4}
    zz , yy , xx = np.where( label_list == rate_list['cr'] )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
    sampling=ada_sampling(crystal_coordinate)

    print(sampling)