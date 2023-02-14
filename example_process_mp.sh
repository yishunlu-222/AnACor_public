#!/bin/sh 
#cd /home/eaf28336/absorption_correction/cluster/AnACor/
#python3 mp_lite.py \

source /dls/science/groups/i23/yishun/dials_yishun/dials
anacor.mp_lite \
--store-dir /dls/science/groups/i23/yishun/save_data/ \
--dataset  16846 \
--crac 0.01918e3 \
--loac 0.01772e3 \
--liac 0.01981e3 \
--num-cores 20 \
--store-lengths False \
--time 3 0 0  \
--dependancies "module load global/cluster" "source /dls/science/groups/i23/yishun/dials_yishun/dials" \
--python-dependancy "source /dls/science/groups/i23/yishun/dials_yishun/dials"

#--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
#--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
#--dials-dependancy  /data/dataset/segment/16846/ \
#--full-reflection False \

#anacor.preprocesslite