
import pdb
import numpy as np

import json
import os
import re
import csv

def find_file(directory, file_name):
    # this function iteratve the current directory and all subdirectories
    # until it finds the file with the given name
    for root, directories, files in os.walk(directory):
        
        for file in files:
            
            if file_name in file:
                file_path = os.path.join(root, file)
                # Perform operations on the found file
                return file_path
        
        for subdirectory in directories:
            subdirectory_path = os.path.join(root, subdirectory)
            # Recursively call the function for the subdirectory
            file_path = find_file(subdirectory_path, file_name)
            if file_path is not None:
                return file_path

def sorting_path(single_pth):

    data_list=[]
    data_multi_list=[]
    for dir_pth in os.listdir(single_pth):
        if os.path.isdir(os.path.join(single_pth, dir_pth)) is False:
            continue
        data_list.append(dir_pth)
    

    sorted_data_list = sorted(data_list, key=extract_numeric_part)
    
    return sorted_data_list

def extract_numeric_part(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return float('inf')



# final_result=[['dataset_name','anom=True','a','b']]
finally_result=[]
list_ac_at=[]
list_acsh_at=[]
list_acsh_af=[]
list_ac_af=[]
list_sh_af=[]
list_sh_at=[]
l_ac_at=[]
l_acsh_at=[]
l_acsh_af=[]
l_ac_af=[]
l_sh_af=[]
l_sh_at=[]
para_dict={'list_ac_at':list_ac_at, 'list_acsh_at':list_acsh_at, 
           'list_acsh_af':list_acsh_af, 'list_ac_af':list_ac_af,
           'list_sh_af':list_sh_af, 'list_sh_af':list_sh_at,}
para_dict_l={'l_ac_at':l_ac_at, 'l_acsh_at':l_acsh_at, 
           'l_acsh_af':l_acsh_af, 'l_ac_af':l_ac_af,
           'l_sh_af':l_sh_af, 'l_sh_af':l_sh_at,}
final_result=[]
final_result_l=[]
base='/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_12_3kev/'
heading=[]
heading_l=[]
for anom in ['af', 'at']:

    for target in ['acsh','ac']:
        if anom == 'at': 
            if target =='sh':
                single_pth =os.path.join(base, 'dials_SH_anom_false','single_datasets')
            else:
                single_pth =os.path.join(base, 'anom_false','single_datasets')
        else:
            if target =='sh':
                single_pth =os.path.join(base, 'dials_SH_anom_true','single_datasets')
            else:
                single_pth = os.path.join(base, 'anom_true','single_datasets')

        dataset_list=sorting_path(single_pth)
        
        if anom == 'at':
            heading+=[target,'anom=False','a','b','']
            heading_l+=[target,'anom=False','']

        else:
            heading+=[target,'anom=True','a','b','']
            heading_l+=[target,'anom=True','']
        # tmp = para_dict['list_{}_{}'.format(target, anom)]
        # tmp_1=para_dict_l['l_{}_{}'.format(target, anom)]
        # print('list_{}_{}'.format(target, anom))
        
        counter=0
        pattern = r"Parameters: a = (\d+\.\d+), b = (\d+\.\d+)" 
        
        for i, dir_pth in enumerate(dataset_list):
            data_pth=find_file(os.path.join(single_pth, dir_pth), '_{}_log.log'.format(target))
            # for root, directories, files in os.walk(os.path.join( single_pth ,dir_pth )):

            with open(data_pth, 'r') as f:
                text = f.read()
            
            matches =  re.findall(pattern, text)
            # Find the start and end indices of the table
            final_a=float(matches[-1][0])
            final_b=float(matches[-1][1])
            if final_a ==1 or final_b ==0.02:
                fail ='Fail'
            else:
                fail ='Success'

            para_dict['list_{}_{}'.format(target, anom)].append([dir_pth, fail, final_a, final_b,''])
            para_dict_l['l_{}_{}'.format(target, anom)].append([dir_pth, fail,''])

final_result.append(heading)
final_result_l.append(heading_l)

for i in range(len(para_dict['list_acsh_af'])):
    tmp=[]
    tmp_1=[]
    for anom in ['af', 'at']:
        for target in ['acsh','ac']:
            # for j in range(len(para_dict['list_{}_{}'.format(target, anom)][i])):
                tmp+=para_dict['list_{}_{}'.format(target, anom)][i]

    final_result.append(tmp)
    final_result_l.append(para_dict_l['l_{}_{}'.format(target, anom)][i])
pdb.set_trace()
filename='errormodel_cld_1704_12_3kev_para.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each row of data to the CSV file
    for r in final_result:
        writer.writerow(r)
filename_l='errormodel_cld_1704_12_3kev_lite.csv'
with open(filename_l, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each row of data to the CSV file
    for r in final_result_l:
        writer.writerow(r)
