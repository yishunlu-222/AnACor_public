import math as m
import numpy as np
import pdb
# from scipy import integrate
import multiprocessing
import os
import json
import re
import  argparse

parser = argparse.ArgumentParser(description="multiprocessing for batches")

parser.add_argument(
    "--save-dir",
    type=str,
    help="save-dir for stacking",
)
parser.add_argument(
    "--dataset",
    type=int,
    default=16010,
    help="dataset number default is 16010",
)

global args
args = parser.parse_args()

def sort_key(s):

    if s:
        try:
            c = re.findall('(\d+)', s)[-1]
        except:
            c = -1
        return int(c)


files=[]
path=args.save_dir
dataset = args.dataset
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
try:
  refl_filaname_list_dict.sort(key=sort_key)
  refl_filaname_list_dict.append(dict_last)
  dict_corr=[]
  for j,i in enumerate(refl_filaname_list_dict):
    filename=os.path.join(path,i)

    with open(filename) as fz:
        data = json.load(fz)
    if j == 0:
        dict_corr = data
    else:
        dict_corr+=data
    # if j == len(refl_filaname_list_dict)-1:
    #     pdb.set_trace()
    fz.close()
  with open(os.path.join(path,'{}_refl_dict_overall.json'.format(dataset)), "w+") as f2:  # Pickling
    json.dump(dict_corr, f2, indent=2)
except:
  pass
corr=[]

for j,i in enumerate(refl_filaname_list):
    filename=os.path.join(path,i)

    with open(filename) as f1:
        data = json.load(f1)

    if j ==0:
        corr = data
    else:
        corr+=data

    f1.close()
#pdb.set_trace()
with open(os.path.join(path,'{}_refl_overall.json'.format(dataset)), "w+") as f1:  # Pickling
    json.dump(corr, f1, indent=2)



