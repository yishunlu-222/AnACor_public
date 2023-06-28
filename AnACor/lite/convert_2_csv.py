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
import csv

def extract_numeric_part(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return float('inf')

single_pth="/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/single_datasets/"
multi_pth="/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/multiple_datasets/"
prefix='auto'
data_list=[]
data_multi_list=[]
for dir_pth in os.listdir(single_pth):
    if os.path.isdir(os.path.join(single_pth, dir_pth)) is False:
      continue
    data_list.append(dir_pth)
for dir_pth in os.listdir(multi_pth):
    if os.path.isdir(os.path.join(multi_pth, dir_pth)) is False:
      continue     
    data_multi_list.append(dir_pth)

sorted_data_list = sorted(data_list, key=extract_numeric_part)
sorted_data_multi_list = sorted(data_multi_list, key=extract_numeric_part)
dataset_list= sorted_data_list+sorted_data_multi_list
single_num=len(sorted_data_list)
for target in ['acsh','ac']:
  final_result=[['dataset_name']]
  counter=0

  
#  for dir_pth in os.listdir(base_pth):
#    if os.path.isdir(os.path.join(base_pth, dir_pth)) is False:
#        continue

  for i, dir_pth in enumerate(dataset_list):
    try:
      if i < single_num:
        refl_pth = os.path.join( single_pth ,dir_pth ,"{0}_save_data/ResultData/dials_output/".format(dir_pth),"result_{0}_{1}.refl".format(dir_pth,target))
        expt_pth=os.path.join( single_pth  ,dir_pth,"{0}_save_data/ResultData/dials_output/".format(dir_pth),"scaled.expt")
        
        expt = load.experiment_list( expt_pth , check_format=False)[0]
        refls = [flex.reflection_table.from_file(refl_pth )]
      else:
        refl_pth = os.path.join( multi_pth ,dir_pth ,"result_{}_{}.refl".format(prefix,target))
        expt_pth=os.path.join( multi_pth  ,dir_pth,"scaled.expt")
        expt = load.experiment_list( expt_pth , check_format=False)[0]
        refls = [flex.reflection_table.from_file(refl_pth )]
    except:
      print("{} has wrong names of the result_(dataset)_(method).refl file \n it may be because the wrong dataset name in ".format(dir_pth))
      continue
    
    experiments=[expt,expt]
    scaled_miller_array = scaled_data_as_miller_array(refls,experiments)
    stats= merging_stats_from_scaled_array(scaled_miller_array)
    stats_dict=stats[0].as_dict()['overall']
    stats_dict_anom=stats[1].as_dict()['overall']
    row=[dir_pth]
    for key in stats_dict:
        if counter==0:
            final_result[0].append(key)
            
        row.append(stats_dict[key])
    for j,heading in enumerate( final_result[0]):
          if row[j] is None:
              row[j]=stats_dict_anom[heading]
    counter+=1
    #pdb.set_trace()
    final_result.append(row)
    print('{} is finished'.format(dir_pth))
    filename='cld_1704_7_3p5kev_anom_false_{}.csv'.format(target)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write each row of data to the CSV file
        for r in final_result:
            writer.writerow(r)
