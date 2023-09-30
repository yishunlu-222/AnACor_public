import pdb
import json
import time
import argparse
import numpy as np
from utils_rt import *
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=Warning)

parser = argparse.ArgumentParser(description="multiprocessing for batches")
parser.add_argument(
    "--mur",
    type=float,
    default=1,
    help="coordinate setting",

)
parser.add_argument(
    "--sam",
    type=int,
    default=0,
    help="coordinate setting",

)



global args
args = parser.parse_args()
correct_p1=np.array([0.86140064, 0.86140064, 0.8616233 , 0.86206897, 0.86258949,
       0.86311065, 0.86363244, 0.86415486, 0.86445367, 0.8645284 ])
correct_p5=np.array([0.48181161, 0.4829518 , 0.48626307, 0.49164208, 0.49860391,
       0.50645733, 0.51427102, 0.52099614, 0.52565181, 0.52731491])
correct_1 =np.array([0.24250067, 0.24500796, 0.25236593, 0.26373395, 0.27777006,
       0.29295445, 0.30770178, 0.32034854, 0.32913142, 0.33233632])

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



def worker_function(args):
    coord, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients = args
    #coord = crystal_coordinate[index]

    face_1 = which_face(coord, shape, theta_1, phi_1)
    face_2 = which_face(coord, shape, theta, phi)

    path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list)
    path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list)

    numbers_1 = cal_path_plus(path_1, voxel_size)
    numbers_2 = cal_path_plus(path_2, voxel_size)

    absorption = cal_rate((numbers_1 + numbers_2), coefficients)
    return absorption

def sphere_ana_ac_test(mu,angle, radius,voxel_size,sampling):
    # Preparing the arguments for the worker function
    t1=time.time()
    coefficients = [0 , 0 , mu , 0]
    # width = 1  # x-axis
    # height = 1  # y-axis
    #
    # resolution = 100  # must be the factor of 10
    HU = 3
    # radius = 1
    pixel_size=voxel_size[0]
    label_list = Sphere( radius , pixel_size , HU )
    
    shape = label_list.shape
    
    zz , yy , xx = np.where( label_list == 3 )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
    seg = int(np.round(len(crystal_coordinate) / sampling))
   

    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    print("len of coordinate_list is {}".format(len(coordinate_list)))
    absorp = np.empty(len(coordinate_list))
    coord_list=[crystal_coordinate[i] for i in  coordinate_list]
    # coord_list = generate_sampling(label_list, cr=3, dim='z', sampling_size=sampling, auto=False,method='even',sampling_ratio=None)
    del zz, yy, xx
        

    theta , phi = angle / 180 * np.pi , 0 / 180 * np.pi
    theta_1 , phi_1 = 180 / 180 * np.pi , 0 / 180 * np.pi
    args = [
        (coord, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients) 
        for coord in coord_list
    ]

    # Using multiprocessing
    num_cores_to_use = multiprocessing.cpu_count() // 1
    print('number of core is {}'.format(num_cores_to_use))
    pool = multiprocessing.Pool(processes=num_cores_to_use)
    absorptions = pool.map(worker_function, args)
    pool.close()
    pool.join()
    t2  = time.time()
    print(sum(absorptions) / len(absorptions))
    print('angle:{} radius: {} time spent is {}'.format(angle, radius, t2-t1))
    
    return sum(absorptions) / len(absorptions)

# def sphere_ana_ac_test(mu,angle, radius,resolution,sampling=20):
#     # mu = 1
#     t1=time.time()
#     coefficients = [0 , 0 , mu , 0]
#     # width = 1  # x-axis
#     # height = 1  # y-axis
#     #
#     # resolution = 100  # must be the factor of 10
#     HU = 3
#     # radius = 1
#     label_list = Cylinder( radius , resolution , HU )
  
#     shape = label_list.shape
    
#     zz , yy , xx = np.where( label_list == 3 )
#     crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
#     seg = int(np.round(len(crystal_coordinate) / sampling))
 

#     coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
#     absorp = np.empty(len(coordinate_list))
#     del zz, yy, xx
        
  
#     theta , phi = angle / 180 * np.pi , 0 / 180 * np.pi
#     theta_1 , phi_1 = 180 / 180 * np.pi , 0 / 180 * np.pi

#     for k, index in enumerate(coordinate_list):
#         coord = crystal_coordinate[index]

#         face_1 = which_face( coord , shape , theta_1 , phi_1 )

#         face_2 = which_face( coord , shape , theta , phi )

#         path_1 = cal_coord( theta_1 , phi_1 , coord , face_1 , shape , label_list )  # 37
#         path_2 = cal_coord( theta , phi , coord , face_2 , shape , label_list )  # 16
#         numbers_1 = cal_path_plus( path_1 , voxel_size )  # 3.5s
#         numbers_2 = cal_path_plus( path_2 , voxel_size )
        
#         absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )
#         if k % 100 ==0:
#             print('Angle[{}]: [{}]/[{}]'.format(angle,k, len(crystal_coordinate)))
#             print('time spent:{}'.format(time.time()-t1))
#             pdb.set_trace()
#         absorp[k] = absorption

#     return absorp.mean( )


if __name__ == '__main__':
    angle_list=np.linspace(start = 0,stop=180,num = 10,endpoint = True)
  
    t1=time.time()
    sampling=2000
    mu=0.01 #um-1
    if args.sam==1:
      sampling=54000 # 2000
    else:
      sampling=1
      
    for mur in [args.mur]:
     # no unit 
        radius= mur/mu # mm
        voxel_size=[0.3,0.3,0.3] # um
        voxel_size=[0.1,0.1,0.1]  
        #voxel_size=[0.03,0.03,0.03]  
#        if mur>0.3:
#          sampling=2000
        #resolution =int(radius/voxel_size[0])
        if mur==0.1:
            reference=correct_p1
        elif mur==0.5:
            reference=correct_p5
            
        elif mur==1:
            reference=correct_1
        else:
            reference=np.zeros(len(correct_p1))
        errors=[]
        with open("sphere_sample_{}_mur_{}_{}_{}.json".format(sampling,mur,voxel_size[0],mu), "w") as f1:  # Pickling
                json.dump(errors, f1, indent=2)
        for i,angle in enumerate(angle_list):
            absorp=sphere_ana_ac_test(mu,angle, radius,voxel_size,sampling)
            er=np.abs(reference[i] -absorp)/absorp
            errors.append([ angle,er, absorp])

            with open("sphere_sample_{}_mur_{}_{}_{}.json".format(sampling,mur,voxel_size[0],mu), "w") as f1:  # Pickling
                json.dump(errors, f1, indent=2)
        t2=time.time()
        print('the time spent is {}'.format(t2 -t1))

