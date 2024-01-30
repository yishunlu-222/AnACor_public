from dials.array_family import flex
import argparse
import os
parser = argparse.ArgumentParser(description="analytical absorption correction data preprocessing")

parser.add_argument(
    "--dataset",
    type=str,
    help="dataset number ",
)
parser.add_argument(
    "--store-dir",
    type=str,
    default = "./",
    help="the store directory ",
)
global args
args = parser.parse_args()
refl_list=[]
dataset = args.dataset
pth = os.path.join( args.store_dir, 'ResultData' )
for file in os.listdir(pth):
    if "rejected" in file and dataset in file:
        refl_list.append(file)
if len(refl_list) ==1:
    refl_path = os.path.join( pth , refl_list[0] )
elif len(refl_list) ==0:
    raise RuntimeError("\n There are no reflections table of sample {} in this directory \n  Please create one by command python setup.py \n".format(dataset))
else:
    raise RuntimeError("\n There are many reflections table of sample {} in this directory \n  Please delete the unwanted reflection tables \n".format(dataset))          
reflections = flex.reflection_table.from_file(refl_path)
print(len(reflections))