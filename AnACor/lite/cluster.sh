#!/bin/sh
source /dls/science/groups/i23/yishun/dials_yishun/dials
num=20
sampling=2000
expri=0
dataset=17520
offset=0
crac=0.01194e3
liac=0.01124e3
loac=0.01087e3
buac=0
end=53023
increment=$[$end / $num]
counter=0
refl_pth=/dls/i23/data/2022/nt27314-37/processed/WNK1/20220429/WNK1_K7_3p65keV/data_1/db552fa5-777e-40fa-8eb0-1d393fe28d1a/xia2-dials/DataFiles//nt27314v37_xdata1_SAD_SWEEP1.refl
store_dir=/dls/science/groups/i23/yishun/save_data/
for i in $(seq 0 $increment $end);
do
  counter=$[$counter+1]
  echo $counter
  if [[ $counter -eq $[$num-1] ]]; then
    x=`echo "$i + $increment*0.6" | bc`  
    nohup python -u  main.py   --low $i --up ${x%.*}  --dataset ${dataset} --sampling ${sampling} --loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset} --store-dir ${store_dir} --refl-filename ${refl_pth} > ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&
    final=$i
    break
  fi
  nohup python -u  main.py   --low $i --up $[$i+$increment] --dataset ${dataset}  --sampling ${sampling}  --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}  --offset ${offset} --store-dir ${store_dir} --refl-filename ${refl_pth} > ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&
  if [[ $counter -eq 1 ]]; then
    sleep 20
  fi
done
nohup python -u main.py   --low ${x%.*} --up -1  --dataset ${dataset} --sampling ${sampling} --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}   --offset ${offset} --store-dir ${store_dir}  --refl-filename ${refl_pth}  > ${store_dir}/logging/nohup_${expri}_${dataset}_$[$counter+1].out
