import argparse
import subprocess
import json
import os
import pdb
import yaml

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
    directory = os.getcwd( )
    # Load the YAML configuration file
    with open( os.path.join(directory,'default_postprocess_input.yaml') , 'r' ) as f :
        config = yaml.safe_load( f )

    # Add an argument for each key in the YAML file
    for key , value in config.items( ) :
        parser.add_argument( '--{}'.format( key ) , default = value )
    # parser.add_argument(
    #     "--store-dir" ,
    #     type = str ,
    #     default = "./" ,
    #     help = "the store directory " ,
    # )
    # parser.add_argument(
    #     "--dataset" ,
    #     type = str , required = True ,
    #     help = "dataset number " ,
    # )
    # parser.add_argument(
    #     "--save-note" ,
    #     type = str , default = 'anacor',
    #     help = "note of the saving" ,
    # )
    # parser.add_argument(
    #     "--refl-filename" ,
    #     type = str ,
    #     default = '' ,
    #     help = "the filenames of the reflection table" ,
    # )
    # parser.add_argument(
    #     "--expt-filename" ,
    #     type = str ,
    #     default = '' ,
    #     help = "the filenames of the experimental table" ,
    # )
    # parser.add_argument(
    #     "--dials-dependancy" ,
    #     type = str ,
    #     default = '' ,
    #     help = "the dials version that is to be executed" ,
    # )
    # parser.add_argument(
    #     "--mtz2sca-dependancy" ,
    #     type = str ,
    #     default = '' ,
    #     help = "the dependancy to convert mtz into sca files" ,
    # )
    # parser.add_argument(
    #     "--full-reflection" ,
    #     type = str2bool ,
    #     default = False ,
    #     help = "prerejection for better computational efficiency no: 1, yes: 1" ,
    # )
    # parser.add_argument(
    #     "--with-scaling" ,
    #     type = str2bool ,
    #     default = True ,
    #     help = "absorption correcction within the scaling process true: 1 , false: 0" ,
    # )
    global args
    args = parser.parse_args( )

    return args


def main ( ) :
    args = set_parser( )
    dataset=args.dataset
    save_dir = os.path.join(  args.store_dir, '{}_save_data'.format( dataset ) )
    result_path  =os.path.join(  save_dir,'ResultData','absorption_factors')
    refl_dir = os.path.join(  save_dir,'ResultData','reflections')
    dials_dir = os.path.join( save_dir , 'ResultData' , 'dials_output' )


    stackingpy_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'stacking.py' )
    intoflexpy_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'into_flex.py' )
    dials_save_name='test_{}.refl'.format(args.save_note)

    with open( os.path.join( save_dir , "dialsprocess_script.sh" ) , "w" ) as f :

        f.write( "#!/bin/sh\n" )
        f.write( "{}\n".format( args.dials_dependancy ) )
        f.write( "\n" )
        f.write( "dials.python {} --save-dir {} --dataset {} \n".format( stackingpy_pth,result_path,args.dataset ) )
        f.write( "\n" )
        f.write( "dials.python {0} "
                 "--save-number {1}  --refl-filename {2}  "
                 "--full {3} --with-scaling {4} "
                 "--dataset {5} "
                 "--target-pth {6} --store-dir {7}  \n".format( intoflexpy_pth,args.save_note,args.refl_filename,args.full_reflection,
                                             args.with_scaling, dataset, dials_dir, args.store_dir
                                                  ) )
        f.write( "cd {} \n".format(dials_dir) )
        f.write( "\n" )
        f.write( "dials.scale  {0} {1} "
                 "anomalous=True  physical.absorption_correction=False physical.analytical_correction=True "
                 "output.reflections=result_{2}_ac.refl  output.html=result_{2}_ac.html "
                 "output{{log={2}_ac_log.log}} output{{unmerged_mtz={2}_unmerged_ac.mtz}} output{{merged_mtz={2}_merged_ac.mtz}} "
                 "\n".format( os.path.join(dials_dir,dials_save_name), args.expt_filename,dataset ) )
        f.write( "\n" )
        f.write( "dials.scale  {0} {1}  "
                 "anomalous=True  physical.absorption_level=high physical.analytical_correction=True "
                 "output.reflections=result_{2}_acsh.refl  output.html=result_{2}_acsh.html "
                 "output{{log={2}_acsh_log.log}}  output{{unmerged_mtz={2}_unmerged_acsh.mtz}} "
                 "output{{merged_mtz={2}_merged_acsh.mtz}} "
                 "\n".format( os.path.join(dials_dir,dials_save_name), args.expt_filename,dataset ) )
        f.write( "{} \n".format(args.mtz2sca_dependancy) )
        f.write( "mtz2sca {}_merged_acsh.mtz   \n".format(dataset ) )
        f.write( "mtz2sca {}_merged_ac.mtz   \n".format( dataset ) )

    result = subprocess.run( "bash {}".format(os.path.join( save_dir , "dialsprocess_script.sh" ) ),
                             shell = True , stdout = subprocess.PIPE , stderr = subprocess.PIPE )
    print( result.returncode )
    print( result.stdout.decode( ) )
    print( result.stderr.decode( ) )



if __name__ == '__main__' :
    main( )