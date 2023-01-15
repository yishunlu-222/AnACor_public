#!/bin/sh
#cd /home/yishun/projectcode/AnACor/AnACor/
#python preprocess_lite.py  \

source /dls/science/groups/i23/yishun/dials_yishun/dials
anacor.preprocess_lite \
--store-dir /dls/science/groups/i23/yishun/save_data \
--dataset  16846 \
--create3D False \
--segimg-path  /dls/i23/data/2022/cm31108-1/processing/tomography/recon/16848/avizo/segmentation_labels  \
--rawimg-path  /dls/i23/data/2022/cm31108-1/processing/tomography/recon/16846/flats_4_abs_coeff \
--coefficient True \
--coefficient-auto True \
--refl-filename  /home/yishun/projectcode/dials_develop/16846/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.refl \
--expt-filename  /home/yishun/projectcode/dials_develop/16846/DataFiles/cm31108v1_xdata1_SAD_SWEEP1.expt \
--dials-dependancy  'source /dls/science/groups/i23/yishun/dials_yishun/dials' \
--full-reflection false \
--model-storepath  /dls/science/groups/i23/yishun/save_data/16846_save_data/16846_.npy

#anacor.preprocesslite