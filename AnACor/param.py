import argparse
import os
import sys



# ===========================================
#        Parse the argument
# ===========================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_parser_init():
    parser = argparse.ArgumentParser(prog="AnACor initialization" )
    parser.add_argument(
        "--post",
        type=str2bool,
        default=False,
        help="initialize the postprocessing configuation file", 
    )
    parser.add_argument(
        "--pre",
        type=str2bool,
        default=True,
        help="initialize the preprocessing configuation file",
    )  
    parser.add_argument(
        "--mp",
        type=str2bool,
        default=True,
        help="initialize the multiprocessing configuation file", 
    )
    args = parser.parse_args()
    return args
def set_parser():

    parser = argparse.ArgumentParser(prog="AnACor" ,
                                     description="AnACor is an accelerated absorption correction software for crystallography by written in Python, C and Cuda. It's currently built at the I23 beamline of Diamond Light Source.\n source link= https://github.com/yishunlu-222/AnACor_public \n documentation is on https://yishunlu-222.github.io/anacor.github.io/")

    parser.add_argument(
        "--low",
        type=int,
        default=0,
        help="The starting point of the batch",
    )
    parser.add_argument(
        "--up",
        type=int,
        default=-1,
        help="The ending point of the batch",
    )
    parser.add_argument(
        "--store-paths",
        type=int,
        default=0,
        help="Flag to store paths (1 for true, 0 for false)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0,
        help="Orientation offset value",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset identifier or name",
    )
    parser.add_argument(
        "--model-storepath",
        type=str,
        required=True,
        help="Path to store the full model",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default="./",
        help="Directory path for storing output files",
    )
    parser.add_argument(
        "--refl-path",
        type=str,
        default="None",
        help="Path to the reflection data file",
    )
    parser.add_argument(
        "--expt-path",
        type=str,
        default="None",
        help="Path to the experiment data file",
    )
    parser.add_argument(
        "--absorption-map",
        type=str2bool,
        default=False,
        help="Flag to produce an absorption map (True/False)",
    )
    parser.add_argument(
        "--map-theta",
        type=int,
        default=360,
        help="Number of divisions in theta for absorption map",
    )
    parser.add_argument(
        "--map-phi",
        type=int,
        default=180,
        help="Number of divisions in phi for absorption map",
    )
    parser.add_argument(
        "--gridding-theta",
        type=int,
        default=360,
        help="Number of gridding divisions in theta",
    )
    parser.add_argument(
        "--gridding-phi",
        type=int,
        default=180,
        help="Number of gridding divisions in phi",
    )
    parser.add_argument(
        "--liac",
        type=float,
        required=True,
        help="Absorption coefficient of liquor (um-1)",
    )
    parser.add_argument(
        "--loac",
        type=float,
        required=True,
        help="Absorption coefficient of loop (um-1)",
    )
    parser.add_argument(
        "--crac",
        type=float,
        required=True,
        help="Absorption coefficient of crystal (um-1)",
    )
    parser.add_argument(
        "--buac",
        type=float,
        required=True,
        help="Absorption coefficient of other components (um-1)",
    )
    parser.add_argument(
        "--sampling-num",
        type=int,
        default=5000,
        help="Number of samples for crystal point calculation",
    )
    parser.add_argument(
        "--auto-sampling",
        type=str2bool,
        default=True,
        help="Flag to automatically determine sampling number (True/False)",
    )
    parser.add_argument(
        "--full-iteration",
        type=int,
        default=0,
        help="Flag for full iteration (break on encountering an air point)",
    )
    parser.add_argument(
        "--pixel-size-x",
        type=float,
        default=0.3,
        help="Pixel size in the x dimension of tomography (um)",
    )
    parser.add_argument(
        "--pixel-size-y",
        type=float,
        default=0.3,
        help="Pixel size in the y dimension of tomography (um)",
    )
    parser.add_argument(
        "--pixel-size-z",
        type=float,
        default=0.3,
        help="Pixel size in the z dimension of tomography (um)",
    )
    parser.add_argument(
        "--openmp",
        type=str2bool,
        default=True,
        help="Flag to enable OpenMP for computation (True/False)",
    )
    parser.add_argument(
        "--gpu",
        type=str2bool,
        default=False,
        help="Flag to enable GPU computation (True/False)",
    )
    parser.add_argument(
        "--single-c",
        type=str2bool,
        default=False,
        help="Flag to use C for single-threaded computation (True/False)",
    )
    parser.add_argument(
        "--slicing",
        type=str,
        default='z',
        help="Direction for slicing sampling (x, y, or z)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads/processes",
    )
    parser.add_argument(
        "--test-mode",
        type=str2bool,
        default=False,
        help="Flag to enable test mode (True/False)",
    )
    parser.add_argument(
        "--bisection",
        type=str2bool,
        default=False,
        help="Flag to activate the bisection method (True/False)",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default='evenrandom',
        help="Method for sampling (e.g., 'even', 'random','evenrandom')",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="Ratio for sampling (relevant for certain sampling methods)",
    )
    parser.add_argument(
        "--gpumethod",
        type=int,
        default=1,
        help="Method identifier for GPU computation",
    )
    parser.add_argument(
        "--gpu-card",
        type=str,
        default='a100',
        help="Type of GPU card used (e.g., 'a100', 'v100')",
    )
    parser.add_argument(
        "--gridding",
        type=str2bool,
        default=False,
        help="Flag to enable gridding (True/False)",
    )
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default='linear',
        help="Method for interpolation (e.g., 'linear', 'cubic')",
    )
    parser.add_argument(
        "--bisection-py",
        type=str2bool,
        default=False,
        help="Flag to use Python for bisection method (True/False)",
    )
    parser.add_argument(
        "--DEBUG",
        type=str2bool,
        default=False,
        help="Flag to enable debug mode (True/False)",
    )
    parser.add_argument(
        "--gridding-method",
        type=int,
        default=2,
        help="Method identifier for gridding",
    )
    parser.add_argument(
        "--printing",
        type=str2bool,
        default=True,
        help="Flag to enable printing of output (True/False)",
    )
    parser.add_argument(
        "--single-distribution",
        type=str2bool,
        default=False,
        help="Flag to use a single distribution (True/False)",
    )
    parser.add_argument(
        "--inter-method",
        type=str,
        default='nearest',
        help="Method for intermediate processing (e.g., 'nearest', 'linear')",
    )
    parser.add_argument(
        "--gridding-bisection",
        type=str2bool,
        default=False,
        help="Flag to enable gridding with bisection (True/False)",
    )
    parser.add_argument(
        "--resolution-factor",
        type=float,
        default=None,
        help="Flag to enable gridding with bisection (True/False)",
    )
    parser.add_argument(
        "--partial-illumination",
        type=str2bool,
        default=False,
        help="whether to use partial illumination",
    )
    parser.add_argument(
        "--centre-point-x",
        type=int,
        default=500,
        help="centre point of the beam in x direction",
    )
    parser.add_argument(
        "--centre-point-y",
        type=int,
        default=500,
        help="centre point of the beam in y direction",
    )
    parser.add_argument(
        "--centre-point-z",
        type=int,
        default=500,
        help="centre point of the beam in z direction",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=200,
        help="beam width",
    )
    parser.add_argument(
        "--beam-height",
        type=int,
        default=200,
        help="beam height",
    )
    args = parser.parse_args()
    return args