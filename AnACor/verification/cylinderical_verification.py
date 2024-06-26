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

cylinder_tables_p1=1/np.array([1.1843,	1.1843,	1.1842,	1.1840,	1.1838,	1.1835,	1.1832,	1.1828,	1.1823,	1.1818,	1.1813,	1.1808,	1.1802,	1.1798	,1.1793	,1.1790	,1.1787,	1.1785,	1.1785])
global args
args = parser.parse_args()

def Cylinder ( radius , pixel_size , sphere_value ) :
    # https://stackoverflow.com/questions/64212348/creating-a-sphere-at-center-of-array-without-a-for-loop-with-meshgrid-creates-sh

    num_pix =  int((radius/pixel_size) *3 )
    Radius_sq_pixels = int((radius/pixel_size) ** 2)

    center_pixel = int( num_pix / 2 - 1 )
    new_array = np.zeros( (num_pix , num_pix , num_pix) )

    m , n , r = new_array.shape
    x = np.arange( 0 , m , 1 )
    y = np.arange( 0 , n , 1 )
    #z = np.arange( 0 , r , 1 )
    z = np.arange(center_pixel - 1, center_pixel + 1, 1)
    
    xx , yy , zz = np.meshgrid( x , y , z , indexing = 'ij' , sparse = True )
    xx , yy = np.meshgrid( x , y ,indexing = 'ij' , sparse = True )

    X = (xx - center_pixel)
    Y = (yy - center_pixel)
    Z = (zz - center_pixel)

    mask = ((X ** 2) + (Y ** 2) ) < Radius_sq_pixels  # create sphere mask
    new_array = sphere_value * np.stack([mask for _ in range(len(z))], axis=0)
    new_array = new_array.astype( np.uint16 )  # change datatype
    
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


def worker_function(args):
    index, crystal_coordinate, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients = args
    coord = crystal_coordinate[index]

    face_1 = which_face(coord, shape, theta_1, phi_1)
    face_2 = which_face(coord, shape, theta, phi)

    path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list)
    path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list)

    numbers_1 = cal_path_plus(path_1, voxel_size)
    numbers_2 = cal_path_plus(path_2, voxel_size)

    absorption = cal_rate((numbers_1 + numbers_2), coefficients)
    return absorption

def sphere_ana_ac_test(mu,angle, radius,sampling):
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
    label_list = Cylinder( radius , pixel_size , HU )
  
    shape = label_list.shape
    
    zz , yy , xx = np.where( label_list == 3 )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
    seg = int(np.round(len(crystal_coordinate) / sampling))
 

    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    absorp = np.empty(len(coordinate_list))
    del zz, yy, xx
        
    
    theta , phi = angle / 180 * np.pi , 0 / 180 * np.pi
    theta_1 , phi_1 = 180 / 180 * np.pi , 0 / 180 * np.pi
    args = [
        (index, crystal_coordinate, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients) 
        for index in coordinate_list
    ]

    # Using multiprocessing
    pool = multiprocessing.Pool()
    absorptions = pool.map(worker_function, args)
    pool.close()
    pool.join()
    t2  = time.time()
    print(sum(absorptions) / len(absorptions))
    print('time spent is {}',t2-t1)
   
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
    sampling=2000
    t1=time.time()
    mu=0.01 #um-1
    for mur in [0.1,0.2,0.3,0.5,1,2]:
     # no unit 
        radius= mur/mu # mm
        voxel_size=[0.03,0.03,0.03] # um
#        if mur>0.3:
#          sampling=2000

        errors=[]
        for angle in angle_list:
            er=sphere_ana_ac_test(mu,angle, radius,sampling)
            errors.append([ angle,er])

            with open("cylinder_sample_{}_mur_{}_{}.json".format(sampling,mur,voxel_size[0]), "w") as f1:  # Pickling
                json.dump(errors, f1, indent=2)
        t2=time.time()
        print('the time spent is {}'.format(t2 -t1))

