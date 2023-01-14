#!/bin/sh 
cd /home/yishun/projectcode/AnACor/AnACor/
python3 post_process_lite.py --store-dir /data/absorption_correction_results/ \
--dataset  16846 \
--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
--dials-dependancy  'source /home/yishun/dials_develop_version/dials ' \
--mtz2sca-dependancy 'module load ccp4'


#--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
#--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
#--dials-dependancy  /data/dataset/segment/16846/ \
#--full-reflection False \

#anacor.preprocesslite