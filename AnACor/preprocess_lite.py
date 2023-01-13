import os
import json
import time
import pdb
import numpy as np
# from dials.array_family import flex
import argparse
from AnACor.image_process import Image2Model
import sys
from AnACor.absorption_coefficient import RunAbsorptionCoefficient

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
        help="the path of segmentation images",
    )
    parser.add_argument(
        "--rawimg-path",
        type=str,
        default = None,
        help="the path of raw flat-field images",
    )
    parser.add_argument(
        "--store-calculation",
        type=str2bool,
        default = False,
        help="whether store the path lengths to calculate with different absorption coefficients",
    )

    parser.add_argument(
        "--refl-filename",
        type=str,
        required=True,
        help="the path of the reflection table",
    )
    parser.add_argument(
        "--expt-filename",
        type=str,
        required=True,
        help="the path of the experimental file",
    )
    parser.add_argument(
        "--model-storepath",
        type=str,
        default=None,
        help="the storepath of the 3D model built by other sources in .npy",
    )
    parser.add_argument(
        "--create3D",
        type=str2bool,
        default=True,
        help="whether the reconstruction slices need to be vertically filpped to match that in the real experiment",
    )
    parser.add_argument(
        "--coefficient",
        type=str2bool,
        default=False,
        help="whether the reconstruction slices need to be vertically filpped to match that in the real experiment",
    )
    parser.add_argument(
        "--coefficient-auto",
        type=str2bool,
        default=True,
        help="whether calculating the best estimate of the flat-field image to calculate absorption coefficient "
             "automatically",
    )
    parser.add_argument(
        "--coefficient-orientation",
        type=int,
        default=0,
        help="the orientation of the flat-field image to match the 3D model in degree"
             "normally this is 0 degree",
    )
    parser.add_argument(
        "--coefficient-viewing",
        type=int,
        default=0,
        help="the viewing angle of the 3D model to have the best region to determine absorption coefficient"
             "in degree",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="the optional 3D model name, otherwise it would be {dataset}_.npy",
    )
    parser.add_argument(
        "--model-storepath",
        type=str,
        default=None,
        help="the storepath of the 3D model built by other sources in .npy",
    )
    parser.add_argument(
        "--full-reflection",
        type=str2bool,
        default = False,
        help="whether cutting some unwanted data of the reflection table"
             "before calculating",
    )
    parser.add_argument(
        "--dials-dependancy",
        type=str,
        required = True,
        help="the path to execute dials package"
             "e.g. module load dials"
             "e.g. source /home/yishun/dials_develop_version/dials",
    )
    global args
    args = parser.parse_args()

    if args.coefficient is True and args.rawimg_path is None :
        parser.error( "If it calculates the absorption coefficient, "
                      "the raw image path is needed" )

    if args.coefficient_auto is False and args.coefficient_viewing is None :
        parser.error( "if the orientation of coefficient_auto is not automatically found"
                      "then --coefficient-viewing is needed" )



    return  args

# if args.a:
#     if args.b:
#         print("Both arguments are entered")
#     else:
#         print("arg b is required when arg a is entered")
# else:
#     print("arg a is not entered")

def preprocess_dial_lite ( args ) :
    # from dials.util.filter_reflections import *
    import subprocess
    with open( os.path.join(args.store_dir,"preprocess_script.sh") , "w" ) as f :
        f.write( "#!/bin/bash\n" )
        f.write( "{} \n".format(args.dials_dependancy) )
        f.write( "expt_pth={} \n".format(args.expt_filename ) )
        f.write( "refl_pth={} \n".format(args.refl_filename))
        f.write( "refl_pth={} \n".format( args.refl_filename ) )
        f.write( "store_dir={} \n".format(args.store_dir ) )
        f.write( "dials.python ./lite/refl_2_json.py "
                 " --dataset ${dataset}  --refl-filename ${refl_pth} "
                 "--expt-filename ${expt_pth} --full ${full}"
                 "--save-dir ${store_dir}\n" )

    subprocess.run( ["chmod" , "+x" , "script.sh"] )
    result = subprocess.run( ["./script.sh"], shell = True , stdout = subprocess.PIPE , stderr = subprocess.PIPE )
    print( result.returncode )
    print( result.stdout )
    print( result.stderr )


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
    if args.model_name is not None:
        model_name='./{}.npy'.format( args.model_name  )
    else:
        model_name = './{}_.npy'.format( dataset )
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
    model_storepath = args.model_storepath
    if args.create3D is True:
        ModelGenerator = Image2Model( args.segimg_path,model_path )
        model_storepath=ModelGenerator.run()
    print("\n3D model file is already created... \n")

    if args.coefficient is True:
        if model_storepath is None:
            raise RuntimeError("The 3D model is not defined and run by create3D by this program")


        coefficient_model = RunAbsorptionCoefficient( args.rawimg_path , model_storepath ,auto = True,
                                                        save_dir = os.path.join(args.store_dir,"/ResultData/absorption_coefficient"),
                                                        offset = args.coefficient_orientation,
                                                        angle=args.coefficient_viewing,
                                                        kernel_square = (5 , 5) ,
                                                        full = False ,thresholding="mean")
        coefficient_model.run( )


    # try:
    #     from dials.array_family import flex
    # except:
    #     RuntimeError("Fail to load dials modules please check")
    # print("\nDIALS environment is confirmed... \n")
    #
    # reflections = flex.reflection_table.from_file(os.path.join(save_dir,"reflections", args.refl_filename ))
    # data = preprocess_dial(reflections,args.refl_filename,save_dir,args)
    # print("total number of reflections is {}".format(len(data)))
    # print("reflection table has been preprocessed... \n")
