#!/bin/sh 

cd /home/yishun/projectcode/AnACor/AnACor/
python preprocess_lite.py --store-dir /data/absorption_correction_results/ \
--dataset  16846 \
--coefficient False \
--create3D False \
--segimg-path  /data/dataset/segment/16846/ \
--rawimg-path  '/data/dataset/flat_field/16846 flat field 3keV/' \
--coefficient-auto True \
--refl-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
--expt-filename  /home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
--dials-dependancy  'source /home/yishun/dials_develop_version/dials ' \
--full-reflection false \
--model-storepath  /data/absorption_correction_results/16846_save_data/16846_.npy

#anacor.preprocesslite