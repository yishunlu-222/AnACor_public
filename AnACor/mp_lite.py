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
    parser.add_argument(
        "--input-file" ,
        type = str ,
        default='default_mpprocess_input.yaml',
        help = "the path of the input file of all the flags" ,
    )
    parser.add_argument(
        "--auto-sampling" ,
        type = str2bool ,
        default = True,
        help = "pixel size of tomography" ,
    )
    directory = os.getcwd( )
    global ar
    ar = parser.parse_args( )

    try:
        with open( ar.input_file , 'r' ) as f :
            config = yaml.safe_load( f )
    except:
        with open( os.path.join(directory,ar.input_file) , 'r' ) as f :
            config = yaml.safe_load( f )


    # # Add an argument for each key in the YAML file
    # for key, value in config.items():
    #     # Check if the key is "auto-sampling"
    #     if key == "auto-sampling":
    #         # If so, overwrite the default value with the value from the YAML file
    #         parser.add_argument(
    #             "--{}".format(key),
    #             type=str2bool,
    #             default=value,
    #             help="pixel size of tomography",
    #         )
    #     else:
    #         # Otherwise, use the default value from the add_argument method
    #         parser.add_argument("--{}".format(key), default=value)

    # Add an argument for each key in the YAML file
    for key , value in config.items( ) :
        parser.add_argument( '--{}'.format( key ) , default = value )
    # parser.add_argument(
    #     "--num-cores" ,
    #     type = int ,
    #     default = 20 ,
    #     help = "the number of cores to be distributed" ,
    # )
    # parser.add_argument(
    #     "--store-dir" ,
    #     type = str ,
    #     default = "./" ,
    #     help = "the store directory " ,
    # )
    # parser.add_argument(
    #     "--dataset" ,
    #     type = str , required = True ,
    #     help = "dataset number default is 13304" ,
    # )
    # parser.add_argument(
    #     "--offset" ,
    #     type = float ,
    #     default = 0 ,
    #     help = "the orientation offset" ,
    # )
    # parser.add_argument(
    #     "--sampling" ,
    #     type = int ,
    #     default = 5000 ,
    #     help = "sampling for picking crystal point to calculate" ,
    # )
    #
    # parser.add_argument(
    #     "--store-lengths" ,
    #     type = str2bool ,
    #     default = False ,
    #     help = "whether store the path lengths to calculate with different absorption coefficients" ,
    # )
    # parser.add_argument(
    #     "--crac" ,
    #     type = float , required = True ,
    #     help = "the absorption coefficient of the crystal and it is needed" ,
    # )
    # parser.add_argument(
    #     "--loac" ,
    #     type = float , required = True ,
    #     help = "the absorption coefficient of the loop and it is needed" ,
    # )
    # parser.add_argument(
    #     "--liac" ,
    #     type = float , required = True ,
    #     help = "the absorption coefficient of the liquor and it is needed" ,
    # )
    # parser.add_argument(
    #     "--buac" ,
    #     type = float , default = 0 ,
    #     help = "the absorption coefficient of the bubble and it is not necessarily needed" ,
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
    #     help = "the python version that is to be executed" ,
    # )
    # # parser.add_argument( "--dependancies" , nargs = '*' , type = str ,
    # #                      help = "List of dependancies to execute, they can be entered at the same time"
    # #                             "e.g. 'module load dials' 'module load global/cluster' " )
    # parser.add_argument( "--hpc-dependancies" , nargs = '*' , type = str ,
    #                      help = "List of hpc_dependancies to execute, they can be entered at the same time"
    #                             "e.g. 'module load dials' 'module load global/cluster' " )
    # parser.add_argument(
    #     "--dials-dependancy",
    #     type=str,
    #     required = True,
    #     help="the path to execute dials package"
    #          "e.g. module load dials"
    #          "e.g. source /home/yishun/dials_develop_version/dials",
    # )
    # parser.add_argument( "--time" , nargs = '*' , type = int ,
    #                      help = "List of time for the cluster job"
    #                             "e.g. 01 10 10 is 1hour 10minute 10seconds" )
    global args
    args = parser.parse_args( )

    return args


def main ( ) :
    args = set_parser( )


    save_dir = os.path.join( args.store_dir , '{}_save_data'.format( args.dataset ) )
    if args.model_storepath == 'None' or len(args.model_storepath) < 2 :
        models_list = []
        for file in os.listdir( save_dir ) :
            if args.dataset in file and ".npy" in file :
                models_list.append( file )

        if len( models_list ) == 1 :
            model_storepath = os.path.join( save_dir , models_list[0] )
        elif len( models_list ) == 0 :
            raise RuntimeError(
                "\n There are no 3D models of sample {} in this directory \n  Please create one by command python setup.py \n".format(
                    args.dataset ) )
        else :
            raise RuntimeError(
                "\n There are many 3D models of sample {} in this directory \n  Please delete the unwanted models \n".format(
                    args.dataset ) )
    else :
        model_storepath = args.model_storepath

    for file in os.listdir( save_dir ) :
        if '.json' in file :
            if args.full_reflection:
                if 'expt' in file and 'True' in file :
                    expt_path = os.path.join( save_dir , file )
                if 'refl' in file and 'True' in file:
                    refl_path = os.path.join( save_dir , file )
            else:
                if 'expt' in file and 'False' in file :
                    expt_path = os.path.join( save_dir , file )
                if 'refl' in file and 'False' in file:
                    refl_path = os.path.join( save_dir , file )
    try :
        with open( expt_path ) as f2 :
            axes_data = json.load( f2 )
        print( "experimental data is loaded... \n" )
        with open( refl_path ) as f1 :
            data = json.load( f1 )
        print( "reflection table is loaded... \n" )
    except :
        try :
            with open( args.expt_path ) as f2 :
                axes_data = json.load( f2 )
            print( "experimental data is loaded... \n" )
            with open( args.refl_path ) as f1 :
                data = json.load( f1 )
            print( "reflection table is loaded... \n" )
        except :
            raise RuntimeError( 'no reflections or experimental files detected'
                                'please use --refl_path --expt-filename to specify' )

    py_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'main_lite.py' )
    with open( os.path.join( save_dir , "mpprocess_script.sh" ) , "w" ) as f :

        f.write( "#!/bin/sh\n" )
        f.write( "{}\n".format( args.dials_dependancy ) )
        # f.write("module load python/3.9 \n")
        f.write( "num={}\n".format( args.num_cores ) )
        f.write( "sampling={}\n".format( args.sampling ) )
        f.write( "auto_sampling={}\n".format( args.auto_sampling) )
        f.write( "dataset={}\n".format( args.dataset ) )
        f.write( "offset={}\n".format( args.offset ) )
        f.write( "crac={}\n".format( args.crac ) )
        f.write( "liac={}\n".format( args.liac ) )
        f.write( "loac={}\n".format( args.loac ) )
        f.write( "buac={}\n".format( args.buac ) )
        f.write( "end={}\n".format( len( data ) ) )
        f.write( "py_file={}\n".format( py_pth ) )
        f.write( "model_storepath={}\n".format( model_storepath ) )
        f.write( "full_iter={} \n".format( 0 ) )
        f.write( "by_c={} \n".format( args.by_c ) )
        try :
            f.write( "refl_pth={}\n".format( refl_path ) )
            f.write( "expt_pth={}\n".format( expt_path ) )
        except :
            f.write( "refl_pth={}\n".format( args.refl_path ) )
            f.write( "expt_pth={}\n".format( args.expt_path ) )
        f.write( "store_dir={}\n".format(args.store_dir  ) )
        f.write( "logging_dir={}\n".format( os.path.join( save_dir , 'Logging' ) ) )
        f.write( 'nohup python -u  ${py_file}  --dataset ${dataset} --sampling ${sampling} '
                 '--loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset}'
                 ' --store-dir ${store_dir} --refl-path ${refl_pth} --expt-path ${expt_pth}  '
                 '--model-storepath ${model_storepath} --full-iteration ${full_iter} --num-workers ${num}  '
                 '--sampling-num ${sampling} --auto-sampling ${auto_sampling} --by-c ${by_c}'
                 ' > ${logging_dir}/nohup_${dataset}_${counter}.out\n' )
        
        # f.write( 'increment=$[$end / $num]\n' )
        # f.write( 'counter=0\n' )
        # f.write( 'for i in $(seq 0 $increment $end);\n' )
        # f.write( 'do\n' )
        # f.write( '  counter=$[$counter+1]\n' )
        # f.write( '  if [[ $counter -eq $[$num-1] ]]; then\n' )
        # f.write( '    x=` echo "$i + $increment*0.6" | bc`  \n' )
        # f.write( '    nohup python -u  ${py_file}   --low $i --up ${x%.*}  --dataset ${dataset} --sampling ${sampling} '
        #          '--loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset}'
        #          ' --store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
        #          '--model-storepath ${model_storepath}'
        #          ' > ${logging_dir}/nohup_${expri}_${dataset}_${counter}.out&\n' )
        # f.write( '    final=$i\n' )
        # f.write( '    break\n' )
        # f.write( '  fi\n' )
        # f.write(
        #     '  nohup python -u  ${py_file}    --low $i --up $[$i+$increment] --dataset ${dataset}  --sampling ${sampling} '
        #     ' --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}  --offset ${offset} '
        #     '--store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
        #     '--model-storepath ${model_storepath}'
        #     '> ${logging_dir}/nohup_${expri}_${dataset}_${counter}.out&\n' )
        # f.write( '  if [[ $counter -eq 1 ]]; then\n' )
        # f.write( '    sleep 20\n' )
        # f.write( '  fi\n' )
        # f.write( 'done\n' )
        # f.write( 'nohup python -u ${py_file}    --low ${x%.*} --up -1  --dataset ${dataset} --sampling ${sampling} '
        #          '--loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}   --offset ${offset} '
        #          '--store-dir ${store_dir}  --refl-filename ${refl_pth}  --expt-filename ${expt_pth}  '
        #          '--model-storepath ${model_storepath}'
        #          '> ${logging_dir}/nohup_${expri}_${dataset}_$[$counter+1].out\n' )

        if args.post_process is True:
            dataset = args.dataset
            save_dir = os.path.join( args.store_dir , '{}_save_data'.format( dataset ) )
            result_path = os.path.join( save_dir , 'ResultData' , 'absorption_factors' )
            dials_dir = os.path.join( save_dir , 'ResultData' , 'dials_output' )
            dials_save_name = 'anacor_{}.refl'.format( dataset )
            stackingpy_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'stacking.py' )
            intoflexpy_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'into_flex.py' )
            f.write( "{}\n".format( args.dials_dependancy ) )
            f.write( "\n" )
            f.write(
                "dials.python {} --save-dir {} --dataset {} \n".format( stackingpy_pth , result_path , args.dataset ) )
            f.write( "\n" )
            f.write( "dials.python {0} "
                     "--save-number {1}  --refl-filename {2}  "
                     "--full {3} --with-scaling {4} "
                     "--dataset {5} "
                     "--target-pth {6} --store-dir {7}  \n".format( intoflexpy_pth , args.dataset ,
                                                                    args.refl_path , args.full_reflection ,
                                                                    args.with_scaling , dataset , dials_dir ,
                                                                    args.store_dir
                                                                    ) )
            f.write( "cd {} \n".format( dials_dir ) )
            f.write( "\n" )
            f.write( "dials.scale  {0} {1} "
                     "anomalous={3}  physical.absorption_correction=False physical.analytical_correction=True "
                     "output.reflections=result_{2}_ac.refl  output.html=result_{2}_ac.html "
                     "output{{log={2}_ac_log.log}} output{{unmerged_mtz={2}_unmerged_ac.mtz}} output{{merged_mtz={2}_merged_ac.mtz}} "
                     "\n".format( os.path.join( dials_dir , dials_save_name ) , args.expt_path , dataset,args.anomalous ) )
            f.write( "\n" )
            f.write( "dials.scale  {0} {1}  "
                     "anomalous={3}  physical.absorption_level=high physical.analytical_correction=True "
                     "output.reflections=result_{2}_acsh.refl  output.html=result_{2}_acsh.html "
                     "output{{log={2}_acsh_log.log}}  output{{unmerged_mtz={2}_unmerged_acsh.mtz}} "
                     "output{{merged_mtz={2}_merged_acsh.mtz}} "
                     "\n".format( os.path.join( dials_dir , dials_save_name ) , args.expt_path , dataset,args.anomalous ) )
            f.write( "{} \n".format( args.mtz2sca_dependancy ) )
            f.write( "mtz2sca {}_merged_acsh.mtz   \n".format( dataset ) )
            f.write( "mtz2sca {}_merged_ac.mtz   \n".format( dataset ) )

    cluster_command = "qsub -S /bin/sh -l h_rt={0}:{1}:{2},exclusive=True  -pe openmpi {3}  -o {5} -e {6} {4}".format(
        str( args.hour ).zfill( 2 ) ,
        str( args.minute ).zfill( 2 ) ,
        str( args.second ).zfill( 2 ) ,
        args.num_cores ,
        os.path.join( save_dir , "mpprocess_script.sh" ) ,
        os.path.join( save_dir , "Logging" ) ,
        os.path.join( save_dir , "Logging" ) )

    if args.hpc_dependancies is not None :
        all_command = [args.hpc_dependancies] + [cluster_command]
    else :
        all_command = cluster_command
    command = ""
    for c in all_command :
        command += c + " " + ";" + " "

    result = subprocess.run( command , shell = True , stdout = subprocess.PIPE , stderr = subprocess.PIPE )
    # result = subprocess.run( ["qsub ","-S","","","h_rt={}:{}:{}".format(args.time[0],args.time[1],args.time[2]),
    #                           "-pe","smp", "{}".format(args.num_cores),
    #                           os.path.join(save_dir,"mpprocess_script.sh"),
    #                           "-o",os.path.join(save_dir,"Logging"),
    #                           "-e",os.path.join(save_dir,"Logging"),
    #                           ],
    #                          shell = True , stdout = subprocess.PIPE , stderr = subprocess.PIPE )
    print( result.returncode )
    print( result.stdout.decode( ) )
    print( result.stderr.decode( ) )


if __name__ == '__main__' :
    main( )