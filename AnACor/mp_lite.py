import argparse
import subprocess
import json
import os

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
        "--num-cores",
        type=int,
        default = 20,
        help="the number of cores to be distributed",
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
        default=5000,
        help="sampling for picking crystal point to calculate",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default = "./",
        help="the store directory ",
    )
    parser.add_argument(
        "--store-lengths",
        type=str2bool,
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
    parser.add_argument(
        "--dials-dependancy",
        type=str,
        required = True,
        help="the path to execute dials package"
             "e.g. module load dials"
             "e.g. source /home/yishun/dials_develop_version/dials",
    )
    parser.add_argument( "--time" , nargs = '+' , type = int ,
                         help = "List of time for the cluster job"
                                "e.g. 01 10 10 is 1hour 10minute 10seconds" )
    global args
    args = parser.parse_args()

    return  args

def main() :
    args=set_parser()
    save_dir = os.path.join( args.store_dir , '{}_save_data'.format( args.dataset ) )
    for file in os.listdir(args.store_dir):
        if '.json' in file:
            if 'refl' in file:
                refl_filename = os.path.join(args.store_dir,file)
    try:
        with open(refl_filename) as f1:
            data = json.load(f1)
        print( "size of reflection table is {}... \n".format(len(data)) )
    except:
        raise  RuntimeError('no reflections or experimental files detected'
                            'please use --refl_filename --expt-filename to specify')
    with open( os.path.join(save_dir,"mpprocess_script.sh") , "w" ) as f :
        f.write("#!/bin/sh\n"  )
        f.write("{}\n".format(args.dials_dependancy)  )
        f.write("num={}\n".format(args.num_cores)  )
        f.write("sampling={}\n".format(args.sampling)  )
        f.write("dataset={}\n".format(args.dataset)  )
        f.write("offset={}\n".format(args.offset)  )
        f.write("crac={}\n".format(args.crac)   )
        f.write("liac={}\n".format(args.liac)    )
        f.write("loac={}\n".format(args.loac)    )
        f.write("buac={}\n".format(args.buac)    )
        f.write("end={}\n".format(len(data))    )
        f.write("refl_pth={}\n".format(args.refl_filename)  )
        f.write("expt_pth={}\n".format(args.expt_filename)   )
        f.write("store_dir={}\n".format(args.store_dir)  )
        f.write('increment=$[$end / $num]\n'  )
        f.write('counter=0\n'  )

        f.write('for i in $(seq 0 $increment $end);\n'  )
        f.write('do\n'  )
        f.write('  counter=$[$counter+1]\n'  )
        f.write('  f.write( $counter\n'  )
        f.write('  if [[ $counter -eq $[$num-1] ]]; then\n'  )
        f.write('    x=`f.write( "$i + $increment*0.6\n" | bc`  \n'  )
        f.write('    nohup python -u  main.py   --low $i --up ${x%.*}  --dataset ${dataset} --sampling ${sampling} '
        '--loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset}'
        ' --store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
                ' > ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&\n'  )
        f.write('    final=$i\n'  )
        f.write('    break\n'  )
        f.write('  fi\n'  )
        f.write('  nohup python -u  main.py   --low $i --up $[$i+$increment] --dataset ${dataset}  --sampling ${sampling} '
        ' --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}  --offset ${offset} '
        '--store-dir ${store_dir} --refl-filename ${refl_pth} --expt-filename ${expt_pth}  '
                '> ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&\n'  )
        f.write('  if [[ $counter -eq 1 ]]; then\n'  )
        f.write('    sleep 20\n'  )
        f.write('  fi\n'  )
        f.write('done\n'  )
        f.write('nohup python -u main.py   --low ${x%.*} --up -1  --dataset ${dataset} --sampling ${sampling} '
        '--loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}   --offset ${offset} '
        '--store-dir ${store_dir}  --refl-filename ${refl_pth}  --expt-filename ${expt_pth}  '
                '> ${store_dir}/logging/nohup_${expri}_${dataset}_$[$counter+1].out\n'  )

    # subprocess.run( ["chmod" , "+x" , "script.sh"] )

    result = subprocess.run( ["qsub ","-S","/bin/sh","-l",
                              "h_rt={}:{}:{}".format(args.time[0],args.time[1],args.time[2]),
                              "-pe","smp", "{}".format(args.num_cores),
                              os.path.join(save_dir,"mpprocess_script.sh"),
                              "-o",os.path.join(save_dir,"Logging"),
                              "-e",os.path.join(save_dir,"Logging"),
                              ""],
                             shell = True , stdout = subprocess.PIPE , stderr = subprocess.PIPE )
    print( result.returncode )
    print( result.stdout )
    print( result.stderr )