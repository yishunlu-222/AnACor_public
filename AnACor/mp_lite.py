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
    with open( os.path.join(directory,'default_mpprocess_input.yaml') , 'r' ) as f :
        config = yaml.safe_load( f )

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
            if 'expt' in file :
                expt_filename = os.path.join( save_dir , file )
            if 'refl' in file :
                refl_filename = os.path.join( save_dir , file )
    try :
        with open( expt_filename ) as f2 :
            axes_data = json.load( f2 )
        print( "experimental data is loaded... \n" )
        with open( refl_filename ) as f1 :
            data = json.load( f1 )
        print( "reflection table is loaded... \n" )
    except :
        try :
            with open( args.expt_filename ) as f2 :
                axes_data = json.load( f2 )
            print( "experimental data is loaded... \n" )
            with open( args.refl_filename ) as f1 :
                data = json.load( f1 )
            print( "reflection table is loaded... \n" )
        except :
            raise RuntimeError( 'no reflections or experimental files detected'
                                'please use --refl_filename --expt-filename to specify' )

    py_pth = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ) , 'main_lite.py' )
    with open( os.path.join( save_dir , "mpprocess_script.sh" ) , "w" ) as f :

        f.write( "#!/bin/sh\n" )
        f.write( "{}\n".format( args.dials_dependancy ) )
        f.write( "num={}\n".format( args.num_cores ) )
        f.write( "sampling={}\n".format( args.sampling ) )
        f.write( "dataset={}\n".format( args.dataset ) )
        f.write( "offset={}\n".format( args.offset ) )
        f.write( "crac={}\n".format( args.crac ) )
        f.write( "liac={}\n".format( args.liac ) )
        f.write( "loac={}\n".format( args.loac ) )
        f.write( "buac={}\n".format( args.buac ) )
        f.write( "end={}\n".format( len( data ) ) )
        f.write( "py_file={}\n".format( py_pth ) )
        f.write( "model_storepath={}\n".format( model_storepath ) )
        try :
            f.write( "refl_pth={}\n".format( refl_filename ) )
            f.write( "expt_pth={}\n".format( expt_filename ) )
        except :
            f.write( "refl_pth={}\n".format( args.refl_filename ) )
            f.write( "expt_pth={}\n".format( args.expt_filename ) )
        f.write( "store_dir={}\n".format(args.store_dir  ) )
        f.write( "logging_dir={}\n".format( os.path.join( save_dir , 'Logging' ) ) )
        f.write( 'increment=$[$end / $num]\n' )
        f.write( 'counter=0\n' )
        f.write( 'for i in $(seq 0 $increment $end);\n' )
        f.write( 'do\n' )
        f.write( '  counter=$[$counter+1]\n' )
        f.write( '  if [[ $counter -eq $[$num-1] ]]; then\n' )
        f.write( '    x=` echo "$i + $increment*0.6" | bc`  \n' )
        f.write( '    nohup python -u  ${py_file}   --low $i --up ${x%.*}  --dataset ${dataset} --sampling ${sampling} '
                 '--loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset}'
                 ' --store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
                 '--model-storepath ${model_storepath}'
                 ' > ${logging_dir}/nohup_${expri}_${dataset}_${counter}.out&\n' )
        f.write( '    final=$i\n' )
        f.write( '    break\n' )
        f.write( '  fi\n' )
        f.write(
            '  nohup python -u  ${py_file}    --low $i --up $[$i+$increment] --dataset ${dataset}  --sampling ${sampling} '
            ' --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}  --offset ${offset} '
            '--store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
            '--model-storepath ${model_storepath}'
            '> ${logging_dir}/nohup_${expri}_${dataset}_${counter}.out&\n' )
        f.write( '  if [[ $counter -eq 1 ]]; then\n' )
        f.write( '    sleep 20\n' )
        f.write( '  fi\n' )
        f.write( 'done\n' )
        f.write( 'nohup python -u ${py_file}    --low ${x%.*} --up -1  --dataset ${dataset} --sampling ${sampling} '
                 '--loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}   --offset ${offset} '
                 '--store-dir ${store_dir}  --refl-filename ${refl_pth}  --expt-filename ${expt_pth}  '
                 '--model-storepath ${model_storepath}'
                 '> ${logging_dir}/nohup_${expri}_${dataset}_$[$counter+1].out\n' )

    cluster_command = "qsub -S /bin/sh -l h_rt={0}:{1}:{2} -pe smp {3}  -o {5} -e {6} {4}".format(
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