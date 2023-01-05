import json
import numpy as np
import pdb
import random
import  argparse
from dials.util.filter_reflections import *
from dials.algorithms.scaling.scaler_factory import *
from dials.array_family import flex
import os
parser = argparse.ArgumentParser(description="putting corrected values files into flex files")

parser.add_argument(
    "--save-number",
    type=str,
    default=0,
    help="save-dir for stacking",
)

parser.add_argument(
    "--dataset",
    type=str,
    default=0,
    help="save-dir for stacking",
)
parser.add_argument(
    "--refl-filename",
    type=str,
    default="",
    help="save-dir for stacking",
)
parser.add_argument(
    "--full",
    type=int,
    default=0,
    help="prerejection for better computational efficiency no: 1, yes: 1",
)
parser.add_argument(
    "--with-scaling",
    type=int,
    default=1,
    help="absorption correcction within the scaling process true: 1 , false: 0",
)
parser.add_argument(
    "--data-pth",
    type=str,
    default = "./",
    help="the data directory ",
)
parser.add_argument(
    "--store-pth",
    type=str,
    default = "./",
    help="the store directory ",
)
global args
args = parser.parse_args()

from dials.array_family import flex
#refl_filename=args.refl_fileanme
reflections= flex.reflection_table.from_file(os.path.join(args.store_pth,args.refl_filename))

print("len(reflections)")
print(len(reflections))
corr = np.ones(len(reflections))
p=[]
dataset=args.dataset
filename=os.path.join(args.data_pth,'{}_refl_overall.json'.format(dataset))

if args.full ==1:
    pass
else:  
  scaler = ScalerFactory()
  refls =  scaler.filter_bad_reflections(reflections)
  excluded_for_scaling =  refls.get_flags( refls.flags.excluded_for_scaling)
  refls.del_selected(excluded_for_scaling)
  
corr = np.ones(len(reflections))
with open(filename) as f1:
  data = json.load(f1)
for i,row in enumerate(data):
    corr[i] =row

print(len(data))
#pdb.set_trace()

#print("len(reflections)")
#print(len(reflections))
if args.with_scaling ==1 :
  print("\n the absorption correction factors are combined with scaling \n ")
  ac = flex.double(list(corr))
  reflections["analytical_absorption_correction"] = ac
  reflections.as_file(os.path.join(args.store_pth,"test_{}.refl".format(args.save_number)))
else:
  print("\n  the absorption correction factors are applied directly on the reflection table \n ")
  after = np.array(reflections['intensity.sum.value'])/corr
  #varafter = np.array(reflections['intensity.sum.variance'])/corr
  varafter = np.array(reflections['intensity.sum.variance'])/np.square(corr)
  prf_after = np.array(reflections['intensity.prf.value'])/corr
  #prf_varafter = np.array(reflections['intensity.prf.variance'])/corr
  prf_varafter = np.array(reflections['intensity.prf.variance'])/np.square(corr)
  afterr=flex.double(list(after))
  varafterr = flex.double(list(varafter))
  prf_afterr=flex.double(list(prf_after))
  prf_varafterr = flex.double(list(prf_varafter))
  reflections['intensity.sum.value'] = afterr
  reflections['intensity.sum.variance'] = varafterr
  reflections['intensity.prf.value'] = prf_afterr
  reflections['intensity.prf.variance'] = prf_varafterr
  reflections.as_file(os.path.join(args.store_pth,"test_in_{}.refl".format(args.save_number)))



