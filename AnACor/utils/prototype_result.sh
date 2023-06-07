#!/bin/sh
source /dls/science/groups/i23/yishun/dials_yishun/dials

dataset=17520
refl_filename="nt27314v37_xdata1_SAD_SWEEP1.refl"
expt_filename="nt27314v37_xdata1_SAD_SWEEP1.expt"
store_dir="/dls/science/groups/i23/yishun/save_data/"
experi=0 # experiment number 

refl_filename="rejected_${dataset}_${refl_filename}"

expt_whole_path="${store_dir}/${expt_filename}"
data_pth="${store_dir}/ResultData/${dataset}_save_data/absorption_factors"
store_pth="${store_dir}/ResultData/${dataset}_save_data/reflections"
refl_whole_path="${store_pth}/${refl_filename}"
python stacking.py --save-dir ${data_pth} --dataset ${dataset}

python into_flex.py --save-number ${experi}  --refl-filename ${refl_whole_path} --store-pth  ${store_pth} --dataset ${dataset} --data-pth ${data_pth}
cd  ${store_pth}
dials.scale test_$experi.refl ${expt_whole_path} \
       	 anomalous=True  output.reflections=result_${experi}.refl  output.html=result_${experi}_var_${sq}.html  model=analytical_absorption physical.absorption_correction=False \
          output{unmerged_mtz=${dataset}_unmerged_ac_with_dials.mtz} output{merged_mtz=${dataset}_merged_ac_with_dials.mtz}  # \
#          
dials.scale test_$experi.refl nt27314v37_xdata1_SAD_SWEEP1.expt \
       physical.absorption_level=high anomalous=True  output.reflections=result_${experi}_sh.refl  output.html=result_${experi}_var_${sq}_sh.html model=analytical_absorption  \
       output{unmerged_mtz=${dataset}_unmerged_acsh_with_dials.mtz} output{merged_mtz=${dataset}_merged_acsh.mtz} 

