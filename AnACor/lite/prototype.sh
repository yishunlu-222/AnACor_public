#!/bin/sh

#load the dials module
source /dls/science/groups/i23/yishun/dials_yishun/dials

# (unit in mm-1)
crac=0.01194e3 
liac=0.01124e3
loac=0.01087e3
buac=0

dataset=17520
store_dir="/dls/science/groups/i23/yishun/save_data/"
flat_field_path="/dls/i23/data/2019/nr23571-5/processing/tomography/rotated/13304/tiffs/"
segimg_path="/dls/i23/data/2022/nt27314-37/processing/tomography/recon/17520/avizo/segmentation_tiffs"
refl_path="/dls/i23/data/2022/nt27314-37/processed/WNK1/20220429/WNK1_K7_3p65keV/data_1/db552fa5-777e-40fa-8eb0-1d393fe28d1a/xia2-dials/DataFiles/"
refl_filename="nt27314v37_xdata1_SAD_SWEEP1.refl"
expt_filename="nt27314v37_xdata1_SAD_SWEEP1.expt"
refl_whole_path="${refl_path}/${refl_filename}"

#reflections=${refl_path}/${refl_filename}
offset=0
num_core=20
create3D=yes
experiement=0 # the index number of the experiment 



##install necessary package
dials.python -m pip install numba
dials.python -m pip install -U scikit-image
  dials.python -m pip install opencv-python
dials.python -m pip install imagecodecs
dials.python setup.py --dataset ${dataset} --segimg-path ${segimg_path}    --store-dir ${store_dir} \
                    --refl-filename ${refl_whole_path} --create3D ${create3D}
                    
lenrefl=$(python size_of_reflection.py --store-dir ${store_dir} --dataset ${dataset})
echo "the total number of reflections is ${lenrefl}"
echo "They are distributed to different cores"
sampling=$(python adapative_sampling.py --store-dir ${store_dir} --dataset ${dataset})
#python adapative_sampling.py --store-dir ${store_dir} --dataset ${dataset}
echo " sampling is ${sampling} "
cp  "${refl_path}/${expt_filename} ${store_dir}/ResultData/${dataset}_save_data/reflections"
sleep 6000

sampling=2000
#sleep 6000
module load global/cluster
rm cluster.sh
touch cluster.sh

#lenrefl=287555

# writing the parallelisation script to distribute into codes
echo "#!/bin/sh" >> cluster.sh
echo "source /dls/science/groups/i23/yishun/dials_yishun/dials" >> cluster.sh
echo "num=${num_core}" >> cluster.sh
echo "sampling=${sampling}" >> cluster.sh
echo "expri=${experiement}" >> cluster.sh
echo "dataset=${dataset}" >> cluster.sh
echo "offset=${offset}" >> cluster.sh
echo "crac=${crac}" >> cluster.sh
echo "liac=${liac}" >> cluster.sh
echo "loac=${loac}" >> cluster.sh
echo "buac=${buac}" >> cluster.sh
echo "end=${lenrefl}" >> cluster.sh
echo 'increment=$[$end / $num]' >> cluster.sh
echo "counter=0" >> cluster.sh
echo "refl_pth=${refl_whole_path}" >> cluster.sh
echo "store_dir=${store_dir}" >> cluster.sh
echo 'for i in $(seq 0 $increment $end);' >> cluster.sh
echo 'do' >> cluster.sh
echo '  counter=$[$counter+1]' >> cluster.sh
echo '  echo $counter' >> cluster.sh
echo '  if [[ $counter -eq $[$num-1] ]]; then' >> cluster.sh
echo '    x=`echo "$i + $increment*0.6" | bc`  ' >> cluster.sh
echo '    nohup python -u  main.py   --low $i --up ${x%.*}  --dataset ${dataset} --sampling ${sampling} --loac ${loac} --liac ${liac} --crac ${crac}  --buac ${buac} --offset ${offset} --store-dir ${store_dir} --refl-filename ${refl_pth} > ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&' >> cluster.sh
echo '    final=$i' >> cluster.sh
echo '    break' >> cluster.sh
echo '  fi' >> cluster.sh
echo '  nohup python -u  main.py   --low $i --up $[$i+$increment] --dataset ${dataset}  --sampling ${sampling}  --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}  --offset ${offset} --store-dir ${store_dir} --refl-filename ${refl_pth} > ${store_dir}/logging/nohup_${expri}_${dataset}_${counter}.out&' >> cluster.sh
echo '  if [[ $counter -eq 1 ]]; then' >> cluster.sh
echo '    sleep 20' >> cluster.sh
echo '  fi' >> cluster.sh
echo 'done' >> cluster.sh
echo 'nohup python -u main.py   --low ${x%.*} --up -1  --dataset ${dataset} --sampling ${sampling} --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac}   --offset ${offset} --store-dir ${store_dir}  --refl-filename ${refl_pth}  > ${store_dir}/logging/nohup_${expri}_${dataset}_$[$counter+1].out' >> cluster.sh

#python main.py  --dataset ${dataset}    --offset ${offset}  --refl-filename ${refl_whole_path} --loac ${loac} --liac ${liac} --crac ${crac} --buac ${buac} --store-dir ${store_dir} 
#sleep 6000
qsub -S /bin/sh -l h_rt=03:00:00  -pe smp ${num_core} cluster.sh  -o ${store_dir}/cluster_logging -e ${store_dir}/cluster_logging
echo "The logging files of AnACor are in ${store_dir}/logging"
echo "The cluster_logging files of AnACor are in ${store_dir}/cluster_logging"

#    # Get a python value from BASH.
#pythonval="$(dials.python <<END
#from dials.array_family import flex
#refl_filename="/home/yishun/projectcode/dials_develop/13304_tlys_0p1_4p0keV_dials/tlys_0p1_4p0keV_dials/DataFiles/AUTOMATIC_DEFAULT_SAD_SWEEP1.refl"
#reflections = flex.reflection_table.from_file( refl_filename )
#print(len(reflections))
#END
#)"
#echo "Python value is: $pythonval"

