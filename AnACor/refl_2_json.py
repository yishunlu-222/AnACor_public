

"""
from dials.array_family import flex
reflections = flex.reflection_table.from_file("example_filename.refl")

reflections = flex.reflection_table.from_file("example_filename.refl")
reflections[1]
{'background.mean': 3.0067946910858154, 'background.sum.value': 93.21065521240234, 
 'background.sum.variance': 118.78172302246094, 'bbox': (1310, 1322, 1520, 1532, 0, 1), 
 'd': 4.050291949582959, 'entering': False, 'flags': 1048833, 'id': 0, 'imageset_id': 0, 
 'intensity.prf.value': 0.0, 'intensity.prf.variance': -1.0, 'intensity.sum.value': 7.789344787597656, 
 'intensity.sum.variance': 126.5710678100586, 'lp': 0.30011041432316177, 'miller_index': (1, -5, 16), 
 'num_pixels.background': 113, 'num_pixels.background_used': 113, 'num_pixels.foreground': 31, 'num_pixels.valid': 144, 
 'panel': 0, 'partial_id': 1, 'partiality': 0.0, 'profile.correlation': 0.0, 'qe': 0.9300963202028352, 
 's1': (0.03337449526869561, -0.24160215159383425, -0.7687887369744076),
 'xyzcal.mm': (226.38154049755676, 262.4721810886861, -0.01511517970316119), 
 'xyzcal.px': (1316.2015583386494, 1526.169962473035, -1.7320720071458813), 
 'xyzobs.mm.value': (226.4564177359127, 262.41171771334706, 0.0043633227029589395), 
 'xyzobs.mm.variance': (0.005281526683090885, 0.005847856615085115, 6.346196245556437e-06), 
 'xyzobs.px.value': (1316.6371377429057, 1525.8183361999768, 0.4999999510663235),
 'xyzobs.px.variance': (0.17852645629701475, 0.19766957189984835, 0.08333333333333345), 
 'zeta': 0.9902352872689145}
 
 table.as_file("temp2.refl") save as refl

 """

# import gemmi
import pdb
#
# mtz = gemmi.read_mtz_file('D:/lys/studystudy/phd/absorption_correction/dials/16000_wxz_7_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1_INTEGRATE.mtz')
# data = mtz.datasets[0]
# pdb.set_trace()
import json
import pdb
import  numpy as np
from dials.array_family import flex
from dxtbx.serialize import load
# exp= flex.reflection_table.from_file("16000_wxz_7_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt")
dictionary=[]
filename="sorted_asu.refl"
filename="sorted_asu_result_no.refl.refl"
filename="16010_ompk_10_3p5keV_km70_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl"
expt_filename='16010_ompk_10_3p5keV_km70_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt'

reflections= flex.reflection_table.from_file(filename)
a_file = open(filename + ".json","w")
select=reflections.get_flags(reflections.flags.scaled)

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
pdb.set_trace()
expt = load.experiment_list(expt_filename, check_format=False)[0]
axes=str(expt.goniometer) 
axes=axes.replace('\n','').split('    ')

phi_axis=[]
kappa_axis=[]
omega_axis=[]

angles=[]
fixed_rotation=[]
for  par in axes:
    if 'PHI' in par:
        par = par.split(':')[1].replace('{','').replace('}','').split(',')
        for i in par:
            phi_axis.append(float(i))
    if 'KAPPA' in par:
        par = par.split(':')[1].replace('{','').replace('}','').split(',')
        for i in par:
            kappa_axis.append(float(i))
    if 'OMEGA' in par:
        
        par = par.split(':')[1].replace('{','').replace('}','').split(',')
  
        for i in par:
            omega_axis.append(float(i))
    if 'Angles' in par:
        par = par.split(':')[1].split(',')
        for i in par:
            angles.append(float(i))
        break
    if 'Fixed' in par:
        
        par = par.split(':')[1].replace('{','').replace('}','').split(',')
  
        for i in par:
            fixed_rotation.append(float(i))
matrices=str(expt.crystal) 
matrices=matrices.split('\n')
A_matrix=[]
B_matrix=[]
U_matrix=[]
flag=0
for  mat in matrices:
    
    if 'U matrix' in mat:
        
        mat1 = mat.strip().split(':')[1].replace('{{','').replace('}','').split(',')
        
        for i in mat1:
            try:
              U_matrix.append(float(i))
            except:
              pass
        flag=1
        continue

    if 'B matrix' in mat:
        
        mat1 = mat.strip().split(':')[1].replace('{{','').replace('}','').split(',')
        for i in mat1:
            try:
              B_matrix.append(float(i))
            except:
              pass
        flag=2
        continue

    if 'A = UB' in mat:
        
        mat1 = mat.strip().split(':')[1].replace('{{','').replace('}','').split(',')
        for i in mat1:
            try:
              A_matrix.append(float(i))
            except:
              pass
        flag=3
        

        continue
        
    if flag ==1:
        
        mat = mat.strip().replace('{','').replace('}','').split(',')
        for i in mat:
            try:
              U_matrix.append(float(i))
            except:
              continue
    elif flag == 2:
        mat = mat.strip().replace('{','').replace('}','').split(',')
        for i in mat:
            try:
              B_matrix.append(float(i))
            except:
              continue
    elif flag == 3:
        mat = mat.strip().replace('{','').replace('}','').split(',')
        for i in mat:
            try:
              A_matrix.append(float(i))
            except:
              continue


   
save_expt_axes=[['phi','kappa','omega','U matrix','B matrix','A matrix','fixed rotation']]
save_expt_axes.append(phi_axis)
save_expt_axes.append(kappa_axis)
save_expt_axes.append(omega_axis)
save_expt_axes.append(angles)
save_expt_axes.append(U_matrix)
save_expt_axes.append(B_matrix)
save_expt_axes.append(A_matrix)
save_expt_axes.append(fixed_rotation)
#save_dir=expt_filename + '.json'
with open(expt_filename + '.json', "w") as fz:  # Pickling
    json.dump(save_expt_axes, fz, indent=2)