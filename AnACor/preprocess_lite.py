import os
import json
import yaml
import pdb
import numpy as np
import sys
parent_dir =os.path.dirname( os.path.abspath(__file__))
sys.path.append(parent_dir)


from anacor_logging import setup_logger
import argparse
try:
    from AnACor.utils.image_process import Image2Model
    from AnACor.utils.absorption_coefficient import RunAbsorptionCoefficient
except:
        from utils.image_process import Image2Model
        from utils.absorption_coefficient import RunAbsorptionCoefficient




def str2bool ( v ) :
    if isinstance( v , bool ) :
        return v
    if v.lower( ) in ('yes' , 'true' , 't' , 'y' , '1') :
        return True
    elif v.lower( ) in ('no' , 'false' , 'f' , 'n' , '0') :
        return False
    else :
        raise argparse.ArgumentTypeError( 'Boolean value expected.' )


def set_parser ( ) :
    parser = argparse.ArgumentParser( description = "analytical absorption correction data preprocessing" )

    # parser.add_argument(
    #     "--dataset" ,
    #     type = str ,
    #     help = "dataset number " ,
    # )
    # parser.add_argument(
    #     "--store-dir" ,
    #     type = str ,
    #     default = "./" ,
    #     help = "the store directory " ,
    # )
    # parser.add_argument(
    #     "--segimg-path" ,
    #     type = str ,
    #     help = "the path of segmentation images" ,
    # )
    # parser.add_argument(
    #     "--rawimg-path" ,
    #     type = str ,
    #     default = None ,
    #     help = "the path of raw flat-field images" ,
    # )
    # parser.add_argument(
    #     "--store-calculation" ,
    #     type = str2bool ,
    #     default = False ,
    #     help = "whether store the path lengths to calculate with different absorption coefficients" ,
    # )
    #
    # parser.add_argument(
    #     "--refl-filename" ,
    #     type = str ,
    #     help = "the path of the reflection table" ,
    # )
    # parser.add_argument(
    #     "--expt-filename" ,
    #     type = str ,
    #     help = "the path of the experimental file" ,
    # )
    # parser.add_argument(
    #     "--model-storepath" ,
    #     type = str ,
    #     default = None ,
    #     help = "the storepath of the 3D model built by other sources in .npy" ,
    # )
    # parser.add_argument(
    #     "--create3D" ,
    #     type = str2bool ,
    #     default = True ,
    #     help = "whether the reconstruction slices need to be vertically filpped to match that in the real experiment" ,
    # )
    # parser.add_argument(
    #     "--cal_coefficient-calculation" ,
    #     type = str2bool ,
    #     default = False ,
    #     help = "whether need to calculate coefficients" ,
    # )
    # parser.add_argument(
    #     "--coefficient-auto-orientation" ,
    #     type = str2bool ,
    #     default = True ,
    #     help = "whether automatically match the orientation of 3D model with the flat-field image to calculate absorption coefficient ",
    # )
    # parser.add_argument(
    #     "--coefficient-auto-viewing" ,
    #     type = str2bool ,
    #     default = True ,
    #     help = "whether automatically calculating the largest area of the flat-field image to calculate absorption coefficient "
    #            ,
    # )
    # parser.add_argument(
    #     "--coefficient-orientation" ,
    #     type = int ,
    #     default = 0 ,
    #     help = "the orientation offset of the flat-field image to match the 3D model in degree"
    #            "normally this is 0 degree" ,
    # )
    # parser.add_argument(
    #     "--coefficient-viewing" ,
    #     type = int ,
    #     default = 0 ,
    #     help = "the viewing angle of the 3D model to have the best region to determine absorption coefficient"
    #            "in degree" ,
    # )
    # parser.add_argument(
    #     "--coefficient-thresholding" ,
    #     type = str ,
    #     default = None ,
    #     help = "thresholding method to extract the region of interest"
    #            "options are: 'triangle', 'li', 'mean,'minimum','otsu','yen','isodata'" ,
    # )
    #
    # parser.add_argument(
    #     "--full-reflection" ,
    #     type = str2bool ,
    #     default = False ,
    #     help = "whether cutting some unwanted data of the reflection table"
    #            "before calculating based dials.scale outlier removing algorithm" ,
    # )
    # parser.add_argument(
    #     "--dials-dependancy" ,
    #     type = str ,
    #     help = "the path to execute dials package"
    #            "e.g. module load dials"
    #            "e.g. source /home/yishun/dials_develop_version/dials" ,
    # )
    # parser.add_argument(
    #     "--flat-field-name" ,
    #     type = str ,
    #       default=None,
    #     help = "the flat-field image selected to determine the absorption coefficient, "
    #            "when you use this flag, you should also fill the angle in coefficient_viewing"
    #            "to allow the 3D model to rotate to match it"
    # )
    parser.add_argument(
        "--input-file" ,
        type = str ,
        default='default_preprocess_input.yaml',
        help = "the path of the input file of all the flags" ,
    )
    global ar
    ar = parser.parse_args( )
    directory = os.getcwd( )
    # Load the YAML configuration file
    try:
        with open( ar.input_file , 'r' ) as f :
            config = yaml.safe_load( f )
    except:
        with open( os.path.join(directory,ar.input_file) , 'r' ) as f :
            config = yaml.safe_load( f )
    # Add an argument for each key in the YAML file
    for key , value in config.items( ) :
        parser.add_argument( '--{}'.format( key ) , default = value )

    global args
    args = parser.parse_args( )

    # if args.coefficient is True and args.rawimg_path is None :
    #     parser.error( "If it calculates the absorption coefficient, "
    #                   "the raw image path is needed" )
    #
    # if args.coefficient_auto is False and args.coefficient_viewing is None :
    #     parser.error( "if the orientation of coefficient_auto is not automatically found"
    #                   "then --coefficient-viewing is needed" )

    return args


# if args.a:
#     if args.b:
#         print("Both arguments are entered")
#     else:
#         print("arg b is required when arg a is entered")
# else:
#     print("arg a is not entered")

def preprocess_dial_lite ( args , save_dir,logger ) :
    # from dials.util.filter_reflections import *
    # pdb.set_trace() 
    if os.path.isfile(args.expt_path) is False:
        logger.error("The experiment file is not found")
        print("The experiment file is not found")
        return None
    if os.path.isfile(args.refl_path) is False:
        logger.error("The reflection table is not found")
        print("The reflection table is not found")
        return None
    import subprocess
    logger.info( "\npreprocessing dials data.....\n" )
    print('preprocessing dials data.....')
    with open( os.path.join( save_dir , "preprocess_script.sh" ) , "w" ) as f :
        f.write( "#!/bin/bash \n" )
        
        f.write( "expt_pth=\'{}\' \n".format( args.expt_path) )
        f.write( "refl_pth=\'{}\' \n".format( args.refl_path ) )
        f.write( "store_dir=\'{}\' \n".format( save_dir ) )
        f.write( "dataset={} \n".format( args.dataset ) )
        f.write( "full={} \n".format( args.full_reflection ) )
        f.write( "dials_dependancy=\'{}\' \n".format( args.dials_dependancy ) )
        f.write( "dials.python {}  --dataset ${{dataset}} " 
                 " --refl-filename ${{refl_pth}} " 
                 "--expt-filename ${{expt_pth}} --full ${{full}} "
                 "--save-dir ${{store_dir}}\n".format(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lite/refl_2_json.py')) )

    subprocess.run( ["chmod" , "+x" , os.path.join( save_dir , "preprocess_script.sh" )] )
    try :
        result = subprocess.run( ["bash" , os.path.join( save_dir , "preprocess_script.sh" )] , check = True ,
                                 capture_output = True )
        print( result.stdout.decode( ) )
        logger.info( result.stdout.decode( ) )
    except subprocess.CalledProcessError as e :
        logger.error( "Error: " , e )
        print( "Error: " , e )


def preprocess_dial ( reflections , reflection_path , save_dir , args ) :
    # from dials.util.filter_reflections import *
    from dials.algorithms.scaling.scaler_factory import ScalerFactory

    filename = os.path.basename( reflection_path )

    scaler = ScalerFactory( )
    refls = scaler.filter_bad_reflections( reflections )
    excluded_for_scaling = refls.get_flags( refls.flags.excluded_for_scaling )
    refls.del_selected( excluded_for_scaling )

    filename = "rejected_" + str( args.dataset ) + "_" + filename
    path = os.path.join( save_dir , "reflections" , filename )

    refls.as_file( path )
    return refls

def create_save_dir ( args ) :
    save_dir = os.path.join( args.store_dir , '{}_save_data'.format( args.dataset ) )
    if os.path.exists( save_dir ) is False :
        os.makedirs( save_dir )
        os.makedirs( os.path.join( save_dir , "Logging" ) )
    result_path = os.path.join( save_dir , 'ResultData' )

    # with open(os.path.join( save_dir , "Logging" , 'preprocess_lite.log') , 'w' ) as f :
    #     # Redirect standard output to the file
    #     sys.stdout = f
    #
    #
    #     sys.stdout = sys.__stdout__


    if os.path.exists( result_path ) is False :
        os.makedirs( os.path.join( save_dir , 'ResultData' ) )
        os.makedirs( os.path.join( result_path , "reflections" ) )
        os.makedirs( os.path.join( result_path , "absorption_factors" ) )
        os.makedirs( os.path.join( result_path , "absorption_coefficient" ) )
        os.makedirs( os.path.join( result_path , "dials_output" ) )


# if __name__ == "__main__":
def main ( ) :
    args = set_parser( )
    dataset = args.dataset
    save_dir = os.path.join( args.store_dir , '{}_save_data'.format( args.dataset ) )
    result_path = os.path.join( save_dir , 'ResultData' )




    model_name = './{}_.npy'.format( dataset )
    # segimg_path="D:/lys/studystudy/phd/absorption_correction/dataset/13304_segmentation_labels_tifs/dls/i23" \
    #             "/data/2019/nr23571-5/processing/tomography/recon/13304/avizo/segmentation_labels_tiffs"

    # ModelGenerator = Image2Model(segimg_path , model_name ).run()
    create_save_dir( args )
    save_dir = os.path.join( args.store_dir , '{}_save_data'.format( args.dataset ) )
    global logger
    logger = setup_logger( os.path.join( save_dir , "Logging" , 'preprocess.log') )
    # logger.setLevel( logging.INFO )

    # handler = logging.FileHandler( os.path.join( save_dir , "Logging" , 'preprocess_lite.log') ,
    #                                mode = "w+" , delay=True)
    # handler.setLevel( logging.INFO )

    # formatter = logging.Formatter( '%(asctime)s - %(levelname)s - %(message)s' )
    # handler.setFormatter( formatter )
    # logger.addHandler( handler )
    logger.info( "\nResultData directory is created... \n")
    print( "\nResultData directory is created... \n" )
    
    # this process can be passed in the future

    model_path = os.path.join( save_dir , model_name )
    model_storepath = args.model_storepath
    if args.create3D is True :
        ModelGenerator = Image2Model( args.segimg_path , model_path,logger )
        model_storepath = ModelGenerator.run( )

        with open('./default_preprocess_input.yaml', 'r' ) as f0 :
                pre_config = yaml.safe_load( f0 )
        try:
            model_storepath.replace('./','')
        except:
            pass

        pre_config[ 'model_storepath' ] = model_storepath

        with open( './default_preprocess_input.yaml' , 'w' ) as file :
            yaml.dump( pre_config , file, default_flow_style=False, sort_keys=False, indent=4)

    logger.info( "\n3D model file is already created... \n" )
    print( "\n3D model file is already created... \n" )

    if args.cal_coefficient is True :
        
        if args.model_storepath is not None and args.model_storepath.isspace() is not True\
                and args.model_storepath !='' :
            pass
        else:
            models_list = []
            for file in os.listdir(save_dir ):
                  if str(dataset) in file and ".npy" in file:
                      models_list.append(file)

            try:
                model_storepath = os.path.join( save_dir, models_list[0] )
            except:
                logger.error("The 3D model is not defined or run by create3D by this program")
                raise RuntimeError( "The 3D model is not defined or run by create3D by this program" )
        try:
            coefficient_viewing= args.coefficient_viewing
        except:
            coefficient_viewing=0

        if hasattr(args, 'coe_li'):
            pass
        else:
            args.coe_li = 0.0
        if hasattr(args, 'coe_lo'):
            pass
        else:
            args.coe_lo = 0.0
        if hasattr(args, 'coe_cr'):
            pass
        else:
            args.coe_cr = 0.0
        if hasattr(args, 'coe_bu'):
            pass
        else:
            args.coe_bu = 0.0
        if hasattr(args, 'abs_base_cls'):
            pass
        else:
            args.abs_base_cls = 'li'
        if hasattr(args, 'crop'):
            args.crop = list(args.crop)
        else:
            args.crop = None
        if hasattr(args, 'padding'):
            args.padding = list(args.padding)
        else:
            args.padding = None
        try:
            coefficient_model = RunAbsorptionCoefficient( args.rawimg_path , model_storepath ,
                                                     coe_li= args.coe_li ,
                                                     coe_lo= args.coe_lo ,
                                                        coe_cr= args.coe_cr ,
                                                        coe_bu= args.coe_bu ,
                                                      logger=logger,
                                                      auto_orientation = args.coefficient_auto_orientation ,
                                                      auto_viewing= args.coefficient_auto_viewing ,
                                                      save_dir = os.path.join( result_path ,
                                                                               "absorption_coefficient" ) ,
                                                      offset = args.coefficient_orientation ,
                                                      angle = coefficient_viewing ,
                                                      kernel_square = (5 , 5) ,
                                                      full = False , thresholding = args.coefficient_thresholding,
                                                      flat_fielded=args.flat_field_name,base=args.abs_base_cls, crop=args.crop,padding=args.padding)
            coefficient_model.run( )
        except Exception as e:
            logger.error("The absorption coefficient calculation is failed")
            logger.error(e)
            raise RuntimeError("The absorption coefficient calculation is failed")
    


    
    preprocess_dial_lite( args , save_dir,logger )


    with open('./default_mpprocess_input.yaml', 'r' ) as f3 :
            mp_config = yaml.safe_load( f3 )

    mp_config[ 'refl_path' ] = args.refl_path
    mp_config[ 'expt_path' ] = args.expt_path
    mp_config[ 'dataset' ] = args.dataset      
    mp_config[ 'model_storepath' ] = model_storepath

    # for file in os.listdir(save_dir):
    #     if '.json' in file:
    #         if 'expt' in file:
    #             expt_filename=os.path.join(save_dir,file)
    #         if 'refl' in file:
    #             refl_filename = os.path.join(save_dir,file)

    # mp_config[ 'refl_path' ] = refl_filename
    # mp_config[ 'expt_path' ] = expt_filename

    try:
        with open(os.path.join( result_path , "absorption_coefficient","median_coefficients_with_percentage.json" )) as f2:
            coe = json.load(f2)
        
        mp_config[ 'liac' ] =coe[2][2]
        mp_config[ 'loac' ] =coe[3][2]
        mp_config[ 'crac' ] =coe[4][2]
        try:
            mp_config[ 'buac' ] =coe[5][2]
        except:
            mp_config[ 'buac' ] =0
        logger.info( "\nAbsorption coefficients are written in the mp process input file... \n" )
    except:
        pass
    with open( 'default_mpprocess_input.yaml' , 'w' ) as file :
        yaml.dump( mp_config , file, default_flow_style=False, sort_keys=False, indent=4)

if __name__ == '__main__' :
    main( )

