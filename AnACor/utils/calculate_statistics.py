from dials.algorithms.scaling.scaling_library import \
    merging_stats_from_scaled_array,scaled_data_as_miller_array
from dials.array_family import flex
from dxtbx.serialize import load
import pdb
import numpy as np
import  argparse
import json
import os
import re
parser = argparse.ArgumentParser(description="putting corrected values files into flex files")

parser.add_argument(
    "--start",
    type=str,
    default='AUTOMATIC_DEFAULT_SAD_SWEEP1.expt',
    help="save-dir for stacking",
)
parser.add_argument(
    "--end",
    type=str,
    default='raw_ordering.refl',
    help="save-dir for stacking",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="save-dir for stacking",
)
parser.add_argument(
    "--save-name",
    type=str,
    default="",
    help="save-dir for stacking",
)
parser.add_argument(
    "--savepth",
    type=str,
    default="./",
    help="save-dir for stacking",
)
parser.add_argument(
    "--exppth",
    type=str,
    default="1",
    help="save-dir for stacking",
)
parser.add_argument(
    "--reflpth",
    type=str,
    default="1",
    help="save-dir for stacking",
)
parser.add_argument(
    "--pth",
    type=str,
    default="./",
    help="save-dir for stacking",
)
parser.add_argument(
    "--data-pth",
    type=str,
    default="./",
    help="save-dir for stacking",
)

def sort_key(s):

    if s:
        try:
            c = re.findall('(\d+)', s)[-1]
        except:
            c = -1
        return int(c)

def sort_key_2(s):

    if s:
        try:
            c = re.findall('(\d+\.\d+)', s)[-1]
        except:
            c = -1
        return float(c)

def stacking(dataset,path):
    files=[]

    refl_filaname_list_dict=[]
    refl_filaname_list=[]
    refl_last=None
    for file in os.listdir(path):
        # number = re.findall('(\d+)', file)
        # pdb.set_trace()
        if 'json' not in file:
            continue
        if 'overall' in file:
            continue
    
        if 'dict' in file:
            if '-1' in file:
                dict_last=file
                continue
            refl_filaname_list_dict.append(file)
        else:
            if '-1' in file:
                refl_last = file
                continue
            refl_filaname_list.append(file)
    refl_filaname_list.sort(key=sort_key)
    print(refl_filaname_list)
    
    if refl_last:
        refl_filaname_list.append(refl_last)
        
    for j,i in enumerate(refl_filaname_list):
      filename=os.path.join(path,i)
  
      with open(filename) as f1:
          data = json.load(f1)
  
      if j ==0:
          corr = data
      else:
          corr+=data
  
      f1.close()
    return corr


global args
args = parser.parse_args()
dataset = args.dataset
save_dir = args.savepth
dir=args.pth
expt_file =args.exppth
refl_file=args.reflpth
save_name_rmerge_ac = '{}_resolution_rmerge_plot_ac.json'.format(dataset)
save_name_rmerge_acsh = '{}_resolution_rmerge_plot_acsh.json'.format(dataset)
save_name_isigma_ac = '{}_resolution_isigma_plot_ac.json'.format(dataset)
save_name_isigma_acsh = '{}_resolution_isigma_plot_acsh.json'.format(dataset)
rmerge=[]
isigma=[]
ccanom=[]
ranom=[]
order=[1,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
refer="_1"
for dim in ["x","y","z"]:
  for i, prop in enumerate(order): 
    refl_pth = os.path.join( dir  ,"result_{}_general_resolution_{}{}.refl".format(dataset,dim,prop))
    

    try:
      refls = [flex.reflection_table.from_file(refl_pth )]
      expt = load.experiment_list( os.path.join( dir  ,expt_file) , check_format=False)[0]
      experiments=[expt,expt]
      scaled_miller_array = scaled_data_as_miller_array(refls,experiments)
      stats= merging_stats_from_scaled_array(scaled_miller_array)
      stats_dict=stats[0].as_dict()
      print('ACSH processing {}'.format(refl_pth))
    except:
      continue
    ranom.append(stats_dict['overall']['r_anom']   )
    rmerge.append(stats_dict['overall']['r_merge'])
    isigma.append(stats_dict['overall']['i_over_sigma_mean'])
    ccanom.append(stats_dict['overall']['cc_anom'])
all=[["r_merge", "isigma","ccanom","r_anom"],order,[rmerge,isigma, ccanom,ranom ] ]

with open(os.path.join(save_dir,str(dataset)+"_resolution_plot_ac.json"), "w") as fz:  # Pickling
    json.dump(all, fz, indent=2)

rmerge=[]
isigma=[]
ccanom=[]
ranom=[]
for dim in ["x","y","z"]:
  for i, prop in enumerate(order): 
    refl_pth = os.path.join( dir  ,"result_{}_general_resolution_{}{}.refl".format(dataset,dim,prop))
    

    try:
      refls = [flex.reflection_table.from_file(refl_pth )]
      expt = load.experiment_list( os.path.join( dir  ,expt_file) , check_format=False)[0]
      experiments=[expt,expt]
      scaled_miller_array = scaled_data_as_miller_array(refls,experiments)
      stats= merging_stats_from_scaled_array(scaled_miller_array)
      stats_dict=stats[0].as_dict()
      print('ACSH processing {}'.format(refl_pth))
    except:
      continue
    ranom.append(stats_dict['overall']['r_anom']   )
    rmerge.append(stats_dict['overall']['r_merge'])
    isigma.append(stats_dict['overall']['i_over_sigma_mean'])
    ccanom.append(stats_dict['overall']['cc_anom'])
all_sh=[["r_merge", "isigma","ccanom","r_anom"],order,[rmerge,isigma, ccanom,ranom ] ]

with open(os.path.join(save_dir,str(dataset)+"_resolution_plot_acsh.json"), "w") as fs:  # Pickling
    json.dump(all_sh, fs, indent=2)
pdb.set_trace()
files=[]
path=args.data_pth
dataset = args.dataset
refl_filaname_list_dict=[]
refl_filaname_list=[]
refl_last=None

corr_diff=[]
corr_max=[]
corr_min=[]
for roots in os.walk(path):
    # number = re.findall('(\d+)', file)
    directs=roots[1]
    directs.sort(key=sort_key_2)
    for direct in directs:
      if refer in direct:
          base_data=np.array(stacking(dataset,os.path.join(roots[0],direct)))

    for i, direct in enumerate(directs):
      if refer in direct:
          continue
      else:
          index=float( re.findall('(\d+\.\d+)', direct)[-1])
          try:
              assert index == order[i]
          except:
              pdb.set_trace()
          data=np.array(stacking(dataset,os.path.join(roots[0],direct)))
          diff= np.abs(base_data-data)
          corr_diff.append(diff.mean())
          corr_max.append(diff.max())
          corr_min.append(diff.min())

all_length=[order,["mean","max","min"],[corr_diff,corr_max,corr_min]]
with open(os.path.join(save_dir,str(dataset)+"_resolution_plot_length.json"), "w") as f1:  # Pickling
    json.dump(all_length, f1, indent=2)