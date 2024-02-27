import numpy as np
import pdb 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import time
import ctypes as ct
import argparse
import os
import sys
from analytical import *
parent_dir = os.path.dirname(os.path.dirname( os.path.abspath(__file__)))
utils_path = os.path.join(parent_dir, 'utils')
sys.path.append(utils_path)

try:
    from utils_rt import *
    from utils_os import *

except:
    from AnACor.utils.utils_rt import *
    from AnACor.utils.utils_os import stacking,python_2_c_3d,kp_rotation




parser = argparse.ArgumentParser(description="multiprocessing for batches")
parser.add_argument(
    "--mur",
    type=float,
    default=1,
    help="coordinate setting",

)
parser.add_argument(
    "--sampling-method",
    type=str,
    default='even',
    help="coordinate setting",

)
parser.add_argument(
    "--sampling-ratio",
    type=float,
    default=0.05,
    help="coordinate setting",

)
parser.add_argument(
    "--shape",
    type=str,
    required=True,
    help="coordinate setting",

)
parser.add_argument(
    "--save-dir",
    type=str,
    default='./',
    help="coordinate setting",

)
parser.add_argument(
    "--voxel-size",
    type=float,
    default=0.3,
    help="coordinate setting",

)
parser.add_argument(
    "--num-cores",
    type=int,
    default=4,
    help="coordinate setting",

)
parser.add_argument(
    "--scale-factor",
    type=float,
    default=1,
    help="coordinate setting",

)
global args
args = parser.parse_args()

def extract_from_table(numbers):
    numbers = numbers[0::2]
    numbers=1/np.array(numbers)
    return numbers

cylinder_correct_p5=[2.2996, 2.2979, 2.2926, 2.2840, 2.2721, 2.2572, 2.2398, 2.2204, 2.1996, 2.1781, 2.1564, 2.1352, 2.1152, 2.0969, 2.0809, 2.0677, 2.0579, 2.0518, 2.0497]
cylinder_correct_1=[5.0907, 5.0724, 5.0185, 4.9323, 4.8196, 4.6877, 4.5439, 4.3948, 4.2461, 4.1022, 3.9664, 3.8413, 3.7286, 3.6298, 3.5462, 3.4790, 3.4295, 3.3990, 3.3886]
cylinder_correct_1p5=[10.746, 10.643, 10.349, 9.907, 9.372, 8.800, 8.230, 7.689, 7.192, 6.744, 6.348, 6.002, 5.7036, 5.4516, 5.2441, 5.0804, 4.9609, 4.8875, 4.8625]
cylinder_correct_p5=extract_from_table(cylinder_correct_p5)
cylinder_correct_1=extract_from_table(cylinder_correct_1)
cylinder_correct_1p5=extract_from_table(cylinder_correct_1p5)



sphere_correct_p5=[2.0755, 2.0743, 2.0706, 2.0647, 2.0565, 2.0462, 2.0340, 2.0204, 2.0056, 1.9901, 1.9745, 1.9592, 1.9445, 1.9311, 1.9194, 1.9097, 1.9024, 1.8979, 1.8964]
sphere_correct_1=[4.1237, 4.1131, 4.0815, 4.0304, 3.9625, 3.8816, 3.7917, 3.6966, 3.6001, 3.5048, 3.4135, 3.3280, 3.2499, 3.1807, 3.1216, 3.0738, 3.0383, 3.0163, 3.0090]
sphere_correct_1p5=[7.801, 7.750, 7.604, 7.377, 7.092, 6.775, 6.447, 6.123, 5.8143, 5.5273, 5.2666, 5.0333, 4.8281, 4.6520, 4.5052, 4.3883, 4.3024, 4.2495, 4.2315]
sphere_correct_p5=extract_from_table(sphere_correct_p5)
sphere_correct_1=extract_from_table(sphere_correct_1)
sphere_correct_1p5=extract_from_table(sphere_correct_1p5)



def Sphere ( radius , pixel_size , sphere_value ) :
    
    # https://stackoverflow.com/questions/64212348/creating-a-sphere-at-center-of-array-without-a-for-loop-with-meshgrid-creates-sh

    num_pix =  int((radius/pixel_size) *3 )
    Radius_sq_pixels = int((radius/pixel_size) ** 2)

    center_pixel = int( num_pix / 2 - 1 )
    new_array = np.zeros( (num_pix , num_pix , num_pix) ,dtype=np.int8)

    m , n , r = new_array.shape
    x = np.arange( 0 , m , 1 )
    y = np.arange( 0 , n , 1 )
    z = np.arange( 0 , r , 1 )

    xx , yy , zz = np.meshgrid( x , y , z , indexing = 'ij' , sparse = True )
    X = (xx - center_pixel)
    Y = (yy - center_pixel)
    Z = (zz - center_pixel)

    mask = ((X ** 2) + (Y ** 2) + (Z ** 2)) < Radius_sq_pixels  # create sphere mask
    new_array = sphere_value * mask  # assign values
    # new_array = new_array.astype( np.uint16 )  # change datatype
  
    # import matplotlib.pyplot as plt
    # from skimage import measure
    #
    # fig = plt.figure( )
    # ax = fig.add_subplot( 1 , 1 , 1 , projection = '3d' )
    #
    # verts , faces , normals , values = measure.marching_cubes( new_array,0.1)
    #
    # ax.plot_trisurf(
    #     verts[: , 0] , verts[: , 1] , faces , verts[: , 2] , cmap = 'Spectral' ,
    #     antialiased = False , linewidth = 0.0 )
    # ax.set_ylabel('y')
    # ax.set_zlabel( 'z' )
    # ax.set_xlabel( 'x' )
    # ax.view_init( 0 , 90 )
    # plt.show( )


    return new_array.astype(np.int8)


def Cylinder ( radius , pixel_size , sphere_value,length ) :
    # https://stackoverflow.com/questions/64212348/creating-a-sphere-at-center-of-array-without-a-for-loop-with-meshgrid-creates-sh
    length= int((length/pixel_size) )
    num_pix =  int((radius/pixel_size) *3 )
    Radius_sq_pixels = int((radius/pixel_size) ** 2)

    center_pixel = int( num_pix / 2 - 1 )
    new_array = np.zeros( (num_pix , num_pix , num_pix) )

    m , n , r = new_array.shape
    x = np.arange( 0 , m , 1 )
    y = np.arange( 0 , n , 1 )
    #z = np.arange( 0 , r , 1 )
    z = np.arange(center_pixel - int(np.floor(length/2)), center_pixel + int(np.ceil(length/2)), 1)
    print("len of length is ",len(z))
    xx , yy , zz = np.meshgrid( x , y , z , indexing = 'ij' , sparse = True )
    xx , yy = np.meshgrid( x , y ,indexing = 'ij' , sparse = True )

    X = (xx - center_pixel)
    Y = (yy - center_pixel)
    Z = (zz - center_pixel)

    mask = ((X ** 2) + (Y ** 2) ) < Radius_sq_pixels  # create sphere mask
    new_array = sphere_value * np.stack([mask for _ in range(len(z))], axis=0)
    new_array = new_array.astype( np.uint8 )  # change datatype
    
#    import matplotlib.pyplot as plt
#    from skimage import measure
#    
#    fig = plt.figure( )
#    ax = fig.add_subplot( 1 , 1 , 1 , projection = '3d' )
#    
#    verts , faces , normals , values = measure.marching_cubes( new_array , 0.5 )
#    
#    ax.plot_trisurf(
#       verts[: , 0] , verts[: , 1] , faces , verts[: , 2] , cmap = 'Spectral' ,
#       antialiased = False , linewidth = 0.0 )
#    plt.show( )

    
    # pdb.set_trace( )
    return new_array


def cuboid(length, width, height, voxel_size):
    new_array = np.ones((length+2, int(height/voxel_size[0])+2, int(width/voxel_size[0])+2)).astype(np.int8) *3


    new_array[0:1, :, :] = 0  # Layer at the bottom
    new_array[-1:, :, :] = 0  # Layer at the top
    new_array[:, 0:1, :] = 0  # Layer at the front
    new_array[:, -1:, :] = 0  # Layer at the back
    new_array[:, :, 0:1] = 0  # Layer at the left
    new_array[:, :, -1:] = 0  # Layer at the right
    
    return new_array


def veri_ray_tracing(angle,model, coord_list,voxel_size,coefficients):
    try:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './src/ray_tracing_cpu.so'))
    except:
        anacor_lib_cpu = ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), './src/ray_tracing_cpu.so'))

    # angle = angle / 180 * np.pi
    anacor_lib_cpu.ray_tracing_single_mp.restype = ct.c_double
    anacor_lib_cpu.ray_tracing_single_mp.argtypes = [  # crystal_coordinate_shape
        np.ctypeslib.ndpointer(dtype=np.int64),  # coordinate_list
        ct.c_int,  # coordinate_list_length
        np.ctypeslib.ndpointer(dtype=np.float64),  # rotated_s1
        np.ctypeslib.ndpointer(dtype=np.float64),  # xray
        np.ctypeslib.ndpointer(dtype=np.float64),  # voxel_size
        np.ctypeslib.ndpointer(dtype=np.float64),  # coefficients
        ct.POINTER(ct.POINTER(ct.POINTER(ct.c_int8))),  # label_list
        np.ctypeslib.ndpointer(dtype=np.int64),  # shape
        ct.c_int,  # full_iteration
        ct.c_int,  # store_paths
        ct.c_int,                      # IsExp
        ct.c_int,                      # num_cores
    ]
    label_list_c =python_2_c_3d(model)
    rotated_s1 = myframe_2_dials(thetaphi_2_myframe(angle / 180 * np.pi, 0))
    xray =  -myframe_2_dials(thetaphi_2_myframe(180 / 180 * np.pi, 0))
    shape = np.array(model.shape).astype(np.int64)
    coefficients=np.array([0,0,coefficients,0])
    # pdb.set_trace()
    result = anacor_lib_cpu.ray_tracing_single_mp(
                    coord_list, len(coord_list),
                    rotated_s1, xray, np.array(voxel_size),
                    coefficients, label_list_c, shape,
                    0, 0, 1,args.num_cores)
    return result
if __name__ == '__main__':

    angle_list=np.linspace(start = 0,stop=180,num = 10,endpoint = True)

    mu=0.01 #um-1
    args.voxel_size=np.round(args.voxel_size/args.scale_factor,4)
    voxel_size=[args.voxel_size,args.voxel_size,args.voxel_size] # um
    
    t1=time.time()
    if args.shape=='sphere':
        radius=50  # mur =1
        mur = radius*mu
        filename=f"veri_{args.shape}_{voxel_size[0]}_r_{radius}_{args.sampling_ratio}_mur_{mur}_{args.sampling_method}.json"
        if args.scale_factor!=1:
            filename=f"veri_{args.shape}_{voxel_size[0]}_{args.scale_factor}_r_{radius}_{args.sampling_ratio}_mur_{mur}_{args.sampling_method}.json"
        model=Sphere(radius,voxel_size[0],3)
        if mu*50==0.5:
            reference=sphere_correct_p5
        elif mu*50==1:
            reference=sphere_correct_1
        elif mu*50==1.5:
            reference=sphere_correct_1p5
    elif args.shape=='cylinder':
        radius=50 # mur =1
        length=50
        mur = radius*mu
        filename=f"veri_{args.shape}_{voxel_size[0]}_r_{radius}_l_{length}_{args.sampling_ratio}_mur_{mur}_{args.sampling_method}.json"
        if args.scale_factor!=1:
            filename=f"veri_{args.shape}_{voxel_size[0]}_{args.scale_factor}_r_{radius}_l_{length}_{args.sampling_ratio}_mur_{mur}_{args.sampling_method}.json"
        model=Cylinder(radius,voxel_size[0],3,length)
        if mu*50==0.5:
            reference=cylinder_correct_p5
        elif mu*50==1:
            reference=cylinder_correct_1
        elif mu*50==1.5:
            reference=cylinder_correct_1p5
    
    elif args.shape=='cuboid':
        length=80
        width=100
        height=120
        filename=f"veri_{args.shape}_{voxel_size[0]}_l_{length}_w_{width}_h_{height}_{args.sampling_ratio}_{args.sampling_method}.json"
        if args.scale_factor!=1:
            filename=f"veri_{args.shape}_{voxel_size[0]}_{args.scale_factor}_l_{length}_w_{width}_h_{height}_{args.sampling_ratio}_{args.sampling_method}.json"
        model=cuboid(length, width, height, voxel_size)
        reference=[]
        angle_list/=2
        for theta in angle_list:
                # theta = 180-theta
                if theta == 0:
                    result  = ana_180( mu ,width , height )
                    reference.append(result)
                    continue 
                theta = theta / 180 * np.pi
                
                try :
                    T_l_2 = ana_f_exit_top( mu , theta , width , height )
                except :
                    T_l_2 = ana_f_exit_sides( mu , theta , width , height )
                # if np.isnan(T_l_2):
                #     T_l_2 = 0
                
                reference.append(T_l_2)
    print("reference is ",reference)
    # pdb.set_trace()
    coord_list = generate_sampling(model, cr=3, auto=False, method=args.sampling_method, sampling_ratio=args.sampling_ratio)
    errors=[]
    print("model shape is {}".format(model.shape))
    for i,angle in enumerate(angle_list):
        absorp=veri_ray_tracing(angle,model, coord_list,voxel_size,mu) 
        er=np.abs(reference[i] -absorp)/reference[i] *100
        # if er>1:
        #     pdb.set_trace()
        errors.append([ angle,er, absorp])
        

        with open(os.path.join(args.save_dir,filename), "w") as f1:  # Pickling
            json.dump(errors, f1, indent=2)
    t2=time.time()
    print("time spent is ",t2-t1)
    with open(os.path.join(args.save_dir,'timet_'+filename), "w") as f1:  # Pickling
            json.dump(t2-t1, f1, indent=2)    