import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
import argparse
# from RayTracing import RayTracingBasic
from AnACor.image_process import Image2Model
import sys

def set_parser():

    parser = argparse.ArgumentParser(description="analytical absorption correction data preprocessing")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset number ",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default = "./",
        help="the store directory ",
    )
    parser.add_argument(
        "--segimg-path",
        type=str,
        required=True,
        help="whether store the path lengths to calculate with different absorption coefficients",
    )

    parser.add_argument(
        "--refl-filename",
        type=str,
        default='16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json',
        help="the filenames of the reflection table",
    )
    parser.add_argument(
        "--vflip",
        type=bool,
        default=False,
        help="whether the reconstruction slices need to be vertically filpped to match that in the real experiment",
    )
    parser.add_argument(
        "--create3D",
        type=str,
        default='yes',
        help="whether the reconstruction slices need to be vertically filpped to match that in the real experiment",
    )

    global args
    args = parser.parse_args()
    return  args

def preprocess_dial(reflections,reflection_path,save_dir,args):
    # from dials.util.filter_reflections import *
    from dials.algorithms.scaling.scaler_factory import ScalerFactory

    filename=os.path.basename(reflection_path)
    
    scaler = ScalerFactory( )
    refls = scaler.filter_bad_reflections( reflections )
    excluded_for_scaling = refls.get_flags( refls.flags.excluded_for_scaling )
    refls.del_selected( excluded_for_scaling )
    
    filename = "rejected_"+str(args.dataset) +"_"+ filename
    path = os.path.join(save_dir,"reflections",filename)

    refls.as_file(path)
    return refls

# if __name__ == "__main__":
def main():

    args=set_parser()
    dataset = args.dataset
#    if args.vflip :
#        model_name = './{}_tomobar_cropped_f.npy'.format( dataset )
#    else :
    model_name = './{}_tomobar_cropped.npy'.format( dataset )
    # segimg_path="D:/lys/studystudy/phd/absorption_correction/dataset/13304_segmentation_labels_tifs/dls/i23" \
    #             "/data/2019/nr23571-5/processing/tomography/recon/13304/avizo/segmentation_labels_tiffs"

    # ModelGenerator = Image2Model(segimg_path , model_name ).run()

    result_path = os.path.join( args.store_dir, 'ResultData' )
    if os.path.exists( result_path ) is False :
        os.makedirs( os.path.join(args.store_dir,"/logging") )
        os.makedirs( result_path )
    print("\nResultData directory is created... \n")
      
    save_dir = os.path.join( result_path, '{}_save_data'.format( dataset ) )
    if os.path.exists( save_dir ) is False :
        os.makedirs( save_dir )
        os.makedirs( os.path.join(save_dir,"reflections") )
        os.makedirs( os.path.join(save_dir,"absorption_factors"))
    print("\nDirectory to store absorption factors is created... \n")
    # this process can be passed in the future

    model_path = os.path.join( result_path , model_name )
 
    if args.create3D =='yes':
        ModelGenerator = Image2Model( args.segimg_path,model_path )
        ModelGenerator.run()
    print("\n3D model file is already created... \n")
    try:

        from dials.array_family import flex
    except:
        RuntimeError("Fail to load dials modules please check")
    print("\nDIALS environment is confirmed... \n")

    reflections = flex.reflection_table.from_file(os.path.join(save_dir,"reflections", args.refl_filename ))
    data = preprocess_dial(reflections,args.refl_filename,save_dir,args)
    print("total number of reflections is {}".format(len(data)))
    print("reflection table has been preprocessed... \n")
