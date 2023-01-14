#!/bin/sh 
cd /home/eaf28336/absorption_correction/cluster/AnACor/
python main_lite.py --store-dir /data/absorption_correction_results/ \
--dataset  16846 \
-store-lengths False \
--crac 0.01918 \
--loac 0.01772 \
--liac 0.01981 \


#--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
#--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
#--dials-dependancy  /data/dataset/segment/16846/ \
#--full-reflection False \

#anacor.preprocesslite