
import pdb
import numpy as np

import json
import os
import re
import csv

def extract_numeric_part(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return float('inf')

single_pth="/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/single_datasets"
multi_pth="/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/multiple_datasets"
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
  final_result=[['dataset_name','SG_A:CYS132','SG_B:CYS132','FE_A:HEM201','SD_A:MET76']]
  counter=0

  
  for i, dir_pth in enumerate(dataset_list):
    try:
      if i < single_num:
          with open(os.path.join( single_pth ,dir_pth ,'dimple',target,'anode.lsa'), 'r') as file:
           lines = file.readlines()
      else:
        with open(os.path.join( multi_pth ,dir_pth ,'dimple',target,'anode.lsa'), 'r') as file:
              lines = file.readlines()
    except:
        continue  
        
    # Find the start and end indices of the table
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if 'X        Y        Z   Height(sig)  SOF     Nearest atom' in line:
            start_index = i + 1
        elif 'Peaks output to file' in line:
            end_index = i

    
    table_data = []
    for line in lines[start_index:end_index]:
        # Remove leading/trailing whitespaces and split the line into columns
        columns = line.strip().split()
        # Convert the columns to floats
        row_data = [col for col in columns]
        table_data.append(row_data)
    
    # Convert the table data to a NumPy array
    #table_array = np.array(table_data)

    row=[dir_pth]
    for r in table_data:
        try:
          if r[-1] == 'SG_A:CYS132':
              row.append(r[4])
              break
        except:
            pass
    for r in table_data:
        try:
          if r[-1] == 'SG_B:CYS132':
              row.append(r[4])
              break   
        except:
            pass 
    #pdb.set_trace()  
    for r in table_data:
        try:
          if r[-1] == 'FE_A:HEM201':
              row.append(r[4])
              break   
        except:
            pass 
    for r in table_data:
        try:
          if r[-1] == 'SD_A:MET76':
              row.append(r[4])
              break   
        except:
            pass 
    final_result.append(row)
    
    filename='peak_heights_cld_1704_7_3p5kev_anom_false_{}.csv'.format(target)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write each row of data to the CSV file
        for r in final_result:
            writer.writerow(r)
