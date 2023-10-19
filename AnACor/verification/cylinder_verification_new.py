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

cylinder_tables_p1=1/np.array([1.1843,	1.1843,	1.1842,	1.1840,	1.1838,	1.1835,	1.1832,	1.1828,	1.1823,	1.1818,	1.1813,	1.1808,	1.1802,	1.1798	,1.1793	,1.1790	,1.1787,	1.1785,	1.1785])
correct_p1=np.array([0.84438065, 0.84445195, 0.84473729, 0.84516565, 0.84580902, 0.84652501, 0.84731401, 0.84796065, 0.8483923, 0.84853627])
correct_p5=np.array([0.43485824, 0.43618599, 0.44012147, 0.44646843, 0.45462811, 0.46373586, 0.47276853, 0.4805613, 0.48593226, 0.48787627])
correct_1 =np.array([0.19643664, 0.19926273, 0.2074861, 0.22007527, 0.23551023, 0.25211779, 0.26819718, 0.28199199, 0.29158769, 0.29510712])
global args
args = parser.parse_args()

def Plot_Cylinder ( radius , pixel_size , sphere_value=3,length=100 ) :
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
    new_array = new_array.astype( np.uint16 )  # change datatype
    
    import matplotlib.pyplot as plt
    from skimage import measure
    
    fig = plt.figure(figsize = (19,12))
    ax = fig.add_subplot( 1 , 1 , 1 , projection = '3d' )
    
    verts , faces , normals , values = measure.marching_cubes( new_array , 0.5 )
    
    ax.plot_trisurf(
       verts[: , 0] , verts[: , 1] , faces , verts[: , 2] , cmap = 'Spectral' ,
       antialiased = False , linewidth = 0.0 )
    label_fontsize = 80
    
    # Create the top and bottom caps of the cylinder
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x1 = center_pixel + radius * np.outer(np.cos(u), np.sin(v)) / pixel_size
    y1 =  center_pixel +radius * np.outer(np.sin(u), np.sin(v)) / pixel_size

    # z1 = np.full_like(x, center_pixel - int(np.floor(length / 2)))
    # z2 = np.full_like(x, center_pixel + int(np.ceil(length / 2)))
    # z1 = np.full_like(x1,  - int(np.floor(length / 2)) )
    # z2 = np.full_like(x1, length+int(np.ceil(length / 2)))
    z1 = np.full_like(x1,  -int(length*0.02 ))
    z2 = np.full_like(x1,int( length+length*0.02))
    ax.plot_surface(z1, x1, y1, color='k', alpha=0.6)
    ax.plot_surface(z2, x1, y1, color='k', alpha=0.6)


    ax.set_ylabel('X ', fontsize=label_fontsize)
    ax.set_zlabel('Y ', fontsize=label_fontsize)
    ax.set_xlabel('\n Length \n Z', fontsize=label_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.savefig( 'cylinder visualization.png', dpi = 600 )


    
    pdb.set_trace( )
    return new_array

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
    coord, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients = args

    face_1 = which_face(coord, shape, theta_1, phi_1)
    face_2 = which_face(coord, shape, theta, phi)

    path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list)
    path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list)

    numbers_1 = cal_path_plus(path_1, voxel_size)
    numbers_2 = cal_path_plus(path_2, voxel_size)

    absorption = cal_rate((numbers_1 + numbers_2), coefficients)
    return absorption

def sphere_ana_ac_test(mu,angle, radius,length,sampling):
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
    label_list = Cylinder( radius , pixel_size , HU,length )
  
    shape = label_list.shape
    
    zz , yy , xx = np.where( label_list == 3 )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
    seg = int(np.round(len(crystal_coordinate) / sampling))
 

    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    absorp = np.empty(len(coordinate_list))
    del zz, yy, xx
    coord_list=[crystal_coordinate[i] for i in  coordinate_list]
    
    theta , phi = angle / 180 * np.pi , 0 / 180 * np.pi
    theta_1 , phi_1 = 180 / 180 * np.pi , 0 / 180 * np.pi
    args = [
        (coord, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients) 
        for coord in coord_list
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
    sampling=1
    t1=time.time()
    mu=0.01 #um-1
    mu=0.01
    if args.sam==1:
      length=50
      sampling=54000
    else:
      length=1
      sampling=1
    for mur in [args.mur]:
     # no unit 
        radius= mur/mu # mm
        voxel_size=[0.3,0.3,0.3] # um
        voxel_size=[0.03,0.03,0.03]
        voxel_size=[0.1,0.1,0.1] 
        if mur==0.1:
            reference=correct_p1
        elif mur==0.5:
            reference=correct_p5
            
        elif mur==1:
            reference=correct_1
        else:
            reference=np.zeros(len(correct_p1))
        errors=[]
        with open("cylinder_sample_{}_mur_{}_{}_l_{}_mu_{}.json".format(sampling,mur,voxel_size[0],length,mu), "w") as f1:  # Pickling
                json.dump(errors, f1, indent=2)
        for i,angle in enumerate(angle_list):

            absorp=sphere_ana_ac_test(mu,angle, radius,length,sampling)
            er=np.abs(reference[i] -absorp)/absorp
            errors.append([ angle,er, absorp])
            with open("cylinder_sample_{}_mur_{}_{}_l_{}_mu_{}.json".format(sampling,mur,voxel_size[0],length,mu), "w") as f1:  # Pickling
                json.dump(errors, f1, indent=2)
        t2=time.time()
        print('the time spent is {}'.format(t2 -t1))

