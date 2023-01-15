#!/bin/sh
#cd /home/yishun/projectcode/AnACor/AnACor/
#python preprocess_lite.py  \

source /home/yishun/dials_develop_version/dials
anacor.preprocess_lite \
--store-dir /data/absorption_correction_results/ \
--dataset  16846 \
--coefficient True \
--create3D True \
--segimg-path  /data/dataset/segment/16846/ \
--rawimg-path  '/data/dataset/flat_field/16846 flat field 3keV/' \
--coefficient-auto True \
--refl-filename  /home/yishun/projectcode/dials_develop/16846/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.refl \
--expt-filename  /home/yishun/projectcode/dials_develop/16846/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.expt \
--dials-dependancy  'source /home/yishun/dials_develop_version/dials ' \
--full-reflection false \
#--model-storepath  /data/absorption_correction_results/16846_save_data/16846_.npy

#anacor.preprocesslite