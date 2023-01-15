#!/bin/sh 
#cd /home/yishun/projectcode/AnACor/AnACor/
#python3 post_process_lite.py \

source /dls/science/groups/i23/yishun/dials_yishun/dials
anacor.postprocess_lite \
--store-dir /dls/science/groups/i23/yishun/save_data// \
--dataset  16846 \
--refl-filename  /dls/i23/data/2022/cm31108-1/processed/thaum/20220119/th_1_3keV_2/data_1/1023e415-307e-4bb7-93e2-463316d23553/xia2-dials/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.refl \
--expt-filename  /dls/i23/data/2022/cm31108-1/processed/thaum/20220119/th_1_3keV_2/data_1/1023e415-307e-4bb7-93e2-463316d23553/xia2-dials/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.expt \
--dials-dependancy  'source /dls/science/groups/i23/yishun/dials_yishun/dials' \
--mtz2sca-dependancy 'module load ccp4'


#--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
#--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
#--dials-dependancy  /data/dataset/segment/16846/ \
#--full-reflection False \

#anacor.preprocesslite