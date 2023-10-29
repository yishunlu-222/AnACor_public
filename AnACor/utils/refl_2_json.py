
import os
import json
import pdb
import  numpy as np
from dxtbx.serialize import load
from dials.util.filter_reflections import *
from dials.algorithms.scaling.scaler_factory import *
from dials.array_family import flex
import  argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="putting corrected values files into flex files")

parser.add_argument(
    "--save-dir",
    type=str,
    default="./",
    help="save-dir for stacking",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=0,
    help="the name of the dataset",
)
parser.add_argument(
    "--refl-filename",
    type=str,
    required=True,
    help="save-dir for stacking",
)
parser.add_argument(
    "--expt-filename",
    type=str,
    required=True,
    help="save-dir for stacking",
)
parser.add_argument(
    "--full",
    type=str2bool,
    default=False,
    help="save-dir for stacking",
)
#parser.add_argument(
#    "--save-dir",
#    type=str,
#    required=True,
#    help="1 is true, 0 is false",
#)
global args
args = parser.parse_args()
# pdb.set_trace()

dictionary=[]

filename=args.refl_filename
expt_filename=args.expt_filename
reflections= flex.reflection_table.from_file(filename)
a_file = open( os.path.join(args.save_dir, os.path.basename(filename)
                            + args.dataset+"_"+str(args.full) +".json"),"w")
print("len(reflections)")
print(len(reflections))
if args.full is True:
  pass
else:
  scaler = ScalerFactory()
  refls =  scaler.filter_bad_reflections(reflections)
  excluded_for_scaling =  refls.get_flags( refls.flags.excluded_for_scaling)
  refls.del_selected(excluded_for_scaling)
  
print("len(reflections)")
print(len(reflections))

#select=reflections.get_flags(reflections.flags.scaled)
target=['intensity.sum.value','s1','miller_index','xyzobs.mm.value']
for i in range(len(reflections)):
    dictt={}
#    if  int(bin(reflections[i]['flags'])[-1]) ==0:
#        continue
    for key in reflections[i]:
        try:
#            if key == 'miller_index':slee
#                if reflections[i][str(key)] == (0,0,0):
#                    continue
#
#            if key == 's1':
#                if reflections[i][str(key)] == (0,0,0):
#                    pdb.set_trace()
            if any(item in key for item in target):
                dictt[str(key)]= str(reflections[i][str(key)])
                
        except:
            print(i)
            pass
    #dictt['valid']=select[i]
    dictionary.append(dictt)
#    if select[i] is False:
#      print(i)

json.dump(dictionary, a_file)
a_file.close()

expt = load.experiment_list(expt_filename, check_format=False)[0]
axes =expt.goniometer.to_dict()
beam=expt.beam.to_dict()

expt_=os.path.basename( expt_filename)
with open( os.path.join(args.save_dir, expt_ + args.dataset+"_"+str(args.full)+'.json'), "w") as fz:  # Pickling
    json.dump([axes,beam], fz, indent=2)
