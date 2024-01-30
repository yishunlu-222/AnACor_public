import os
import json
# import pickle
# from matplotlib import pyplot as plt
# from multiprocessing import Process
# import multiprocessing
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils import *
from unit_test import *
import gc
import sys
import multiprocessing

try:
    from scipy.interpolate import interp2d,interpn, RectSphereBivariateSpline,SmoothSphereBivariateSpline
    import psutil
    from memory_profiler import profile
except:
    pass
# ===========================================
#        Parse the argument
# ===========================================

parser = argparse.ArgumentParser( description = "multiprocessing for batches" )

parser.add_argument(
    "--low" ,
    type = int ,
    default = 0 ,
    help = "the starting point of the batch" ,
)
parser.add_argument(
    "--up" ,
    type = int ,
    default = -1 ,
    help = "the ending point of the batch" ,
)
parser.add_argument(
    "--store-paths" ,
    type = int ,
    default = 0 ,
    help = "orientation offset" ,
)

parser.add_argument(
    "--offset" ,
    type = float ,
    default = 0 ,
    help = "orientation offset" ,
)

parser.add_argument(
    "--dataset" ,
    type = int ,
    default = 16846 ,
    help = "1 is true, 0 is false" ,
)
parser.add_argument(
    "--modelpath" ,
    type = str ,
    default ='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/16010_tomobar_cropped_f.npy' ,
    help = "full model path" ,
)
parser.add_argument(
    "--save-dir" ,
    type = str ,
    default = './',
    help = "full storing path" ,
)
parser.add_argument(
    "--refl-path" ,
    type = str ,
    default = './16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json',
    help = "full reflection path" ,
)
parser.add_argument(
    "--expt-path" ,
    type = str ,
    default = './16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt16010_0.json',
    help = "full experiment path" ,
)
parser.add_argument(
    "--li" ,
    type = float ,
    default = 0,
    help = "abs of liquor" ,
)
parser.add_argument(
    "--lo" ,
    type = float ,
    default = 0,
    help = "abs of loop" ,
)
parser.add_argument(
    "--cr" ,
    type = float ,
    default = 0,
    help = "abs of crystal" ,
)
parser.add_argument(
    "--bu" ,
    type = float ,
    default = 0,
    help = "abs of other component" ,
)
parser.add_argument(
    "--pixel-size" ,
    type = float ,
    default = 0.3 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--sampling-threshold" ,
    type = float ,
    default = 5000 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--full-iteration" ,
    type = int ,
    default = 0 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--pixel-size-x" ,
    type = float ,
    default = 0.3 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--pixel-size-y" ,
    type = float ,
    default = 0.3 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--pixel-size-z" ,
    type = float ,
    default = 0.3 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--theta-num" ,
    type =int,
    default = 100 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--phi-num" ,
    type =int,
    default = 50 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--gridding-method" ,
    type =int,
    default = 0,
    help = "different gridding methods" ,
)
parser.add_argument(
    "--create-gridding" ,
    type =int,
    default = 0 ,
    help = "pixel size of tomography" ,
)
parser.add_argument(
    "--compare-origin" ,
    type =int,
    default = 0 ,
    help = "pixel size of tomography" ,
)

global args
args = parser.parse_args( )


def ada_sampling ( crystal_coordinate , threshold = 15000 ) :
    num = len( crystal_coordinate )
    sampling = 1
    result = num
    while result > threshold :
        sampling = sampling * 2
        result = num / sampling

    return sampling


def kp_rotation ( axis , theta , raytracing = True ) :
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param axis:
    :param theta:
    :return:
    """

    x , y , z = axis
    c = np.cos( theta )
    s = np.sin( theta )
    first_row = np.array( [c + (x ** 2) * (1 - c) , x * y * (1 - c) - z * s , y * s + x * z * (1 - c)] )
    seconde_row = np.array( [z * s + x * y * (1 - c) , c + (y ** 2) * (1 - c) , -x * s + y * z * (1 - c)] )
    third_row = np.array( [-y * s + x * z * (1 - c) , x * s + y * z * (1 - c) , c + (z ** 2) * (1 - c)] )
    matrix = np.stack( (first_row , seconde_row , third_row) , axis = 0 )
    return matrix

def memorylog():
    process = psutil.Process( )
    mem_info = process.memory_info( )
    print( f"Memory usage: {mem_info.rss / 1024 / 1024} MB" )
    return  mem_info.rss

def gridding_mp():


    # Define the sampling function
    def sample_func ( model , x ) :
        # Sample the model using the x coordinates
        return model[x , : , :]


        # Generate the 100*50 array
        arr = np.random.randint( 0 , 1000 , size = (100 , 50) )

        # Load the 1000*1000*1000 model into memory
        model = np.random.randn( 1000 , 1000 , 1000 )

        # Define the number of processes to use
        num_processes = multiprocessing.cpu_count( )

        # Divide the array into chunks
        chunk_size = arr.shape[0] // num_processes
        chunks = [arr[i :i + chunk_size , :] for i in range( 0 , arr.shape[0] , chunk_size )]

        # Create a process pool
        pool = multiprocessing.Pool( processes = num_processes )

        # Apply the sampling function to each chunk in parallel
        results = [pool.apply_async( sample_func , args = (model , chunk) ) for chunk in chunks]

        # Collect the results
        output = np.vstack( [result.get( ) for result in results] )

        # Terminate the pool
        pool.close( )
        pool.join( )

def create_gridding(label_list,coordinate_list,voxel_size,
                    coefficients,crystal_coordinate,phiphi,thetatheta,
                    num_processes=2,mp=False):
    if args.gridding_method == 3 :
        absorption_map = np.ones( phiphi.shape )*-1
    else :
        absorption_map = np.ones( phiphi.shape + (len( coordinate_list ) ,) )*-1

    if mp is True:
        if args.gridding_method == 3 :

            chunk_size = absorption_map.shape[0] // num_processes
            chunks = [absorption_map[i :i + chunk_size , :] for i in range( 0 , absorption_map.shape[0] , chunk_size )]
            pdb.set_trace()
            pool = multiprocessing.Pool( processes = num_processes )
            results = [pool.apply_async( gridding_2D , args = (chunk,label_list,coordinate_list,voxel_size,
                coefficients,crystal_coordinate,phiphi[j :j + chunk_size , :] ,thetatheta[j :j + chunk_size , :] ) )
                       for j,chunk in enumerate(chunks)]

            ap = np.vstack( [result.get( ) for result in results] )

            pool.close( )
            pool.join( )

        else:
            # need to be finished
            # Divide the array into chunks
            chunk_size = absorption_map.shape[2] // num_processes
            chunks = [(absorption_map[: , : , i :i + chunk_size] ,) for i in range( 0 , absorption_map.shape[2] , chunk_size )]
            pool = multiprocessing.Pool( processes = num_processes )
            results = [pool.starmap_async(gridding_3D(), [(model , (i , j)) for i in range( chunk.shape[0] ) for j in
                                                          range( chunk.shape[1] )] ) for chunk in chunks]


            output = np.concatenate(
                [result.get( ).reshape( chunk.shape[0] , chunk.shape[1] , -1 ) for result , chunk in
                 zip( results , chunks )] , axis = -1 )


            pool.close( )
            pool.join( )
            chunk_size = absorption_map.shape[0] // num_processes
            chunks = [absorption_map[i :i + chunk_size , :] for i in range( 0 , absorption_map.shape[0] , chunk_size )]
            pool = multiprocessing.Pool( processes = num_processes )
            results = [pool.apply_async( gridding_3D , args = (chunk,label_list,coordinate_list,voxel_size,
                coefficients,crystal_coordinate,phiphi,thetatheta) ) for chunk in chunks]
            ap = np.vstack( [result.get( ) for result in results] )
            pool.close( )
            pool.join( )
    else:
        if args.gridding_method == 3:
            ap = gridding_2D( absorption_map , label_list , coordinate_list , voxel_size ,
                         coefficients , crystal_coordinate , phiphi , thetatheta )
        else:
            ap = gridding_3D( absorption_map , label_list , coordinate_list , voxel_size ,
                         coefficients , crystal_coordinate , phiphi , thetatheta )
    np.save( "./gridding/{}_gridding_{}_{}_{}.npy".format( args.dataset,args.theta_num, args.phi_num,args.gridding_method) ,
                    ap)
def coord_transform(theta,phi):
    if theta<0:
        theta+=2 * np.pi
    if phi<0:
        phi+=np.pi
    return theta,phi
def y_concat(angles):
    theta_max , phi_max = angles.shape
    shift_theta = theta_max // 2
    shift_phi = phi_max // 2
    first_half_rows = angles[:shift_theta , :]
    rest_of_rows = angles[shift_theta : , :]
    angles = np.concatenate((rest_of_rows, first_half_rows), axis=0)

    return angles

def x_concat(angles):
    theta_max , phi_max = angles.shape
    shift_theta = theta_max // 2
    shift_phi = phi_max // 2
    first_half_cols = angles[: , :shift_phi]
    rest_of_cols = angles[: , shift_phi :]
    angles = np.concatenate((rest_of_cols, first_half_cols), axis=1)
    return angles

def unit_test_sphere_transform(detector_gridding,thetatheta,phiphi ):
    ap2=spheretransformation(detector_gridding)
    thetatheta_2=y_concat(thetatheta)
    phiphi_2 = x_concat( phiphi )
    thetatheta_2[thetatheta_2 < 0] += 2*np.pi
    phiphi_2[phiphi_2<0]+=np.pi
    df=[]
    for i, phi_row in enumerate(phiphi):
        for j, phi in enumerate(phi_row):
            theta=thetatheta[i][j]
            ap_1=detector_gridding[i][j]
            theta_2 , phi_2 = coord_transform( theta , phi )
            # j_index= np.where(np.abs(phiphi_2-phi_2)<0.01)[1][0]
            # i_index=np.where(np.abs(thetatheta_2- theta_2)<0.01)[0][0]
            j_index= np.argmin(np.abs(phiphi_2-phi_2),axis=1)[0]
            i_index=np.argmin(np.abs(thetatheta_2- theta_2),axis=0)[0]
            ap_2=ap2[i_index][j_index]
            df.append(np.abs(ap_2-ap_1))
    pdb.set_trace()

def spheretransformation(absorption_map):
    if len(absorption_map.shape)==2:
        theta_max,phi_max=absorption_map.shape
        shift_theta = theta_max // 2
        shift_phi = phi_max // 2
        first_half_rows = absorption_map[1:shift_theta , :]
        rest_of_rows = absorption_map[shift_theta: , :]
        absorption_map = np.vstack((rest_of_rows, first_half_rows))
        first_half_cols = absorption_map[: , 1:shift_phi]
        rest_of_cols = absorption_map[: , shift_phi :]
        ap= np.hstack((rest_of_cols, first_half_cols ))

    elif  len(absorption_map.shape)==3:
        theta_max,phi_max,voxels_number=absorption_map.shape
        shift_theta = theta_max // 2
        shift_phi = phi_max // 2
        first_half_rows = absorption_map[:shift_theta , :,:]
        rest_of_rows = absorption_map[shift_theta: , :,:]
        absorption_map= np.concatenate((rest_of_rows, first_half_rows), axis=0)
        first_half_cols = absorption_map[: , :shift_phi,:]
        rest_of_cols = absorption_map[: , shift_phi :,:]
        ap= np.concatenate((rest_of_cols, first_half_cols), axis=1)

    return ap

def thicken_grid(absorption_map, thickness=1):

    if len(absorption_map.shape)==2:
        ap = np.concatenate(  (absorption_map[-thickness:,:] ,absorption_map, absorption_map[:thickness,:]) ,axis=0)
        ap2 = np.concatenate(  (ap[:,-thickness:] ,ap, ap[:,:thickness]) ,axis=1)
    else:
        ap = np.concatenate(  (absorption_map[-thickness:,:,:] ,absorption_map,
                               absorption_map[:thickness,:,:]) ,axis=0)
        ap2 = np.concatenate(  (ap[:,-thickness:,:] ,ap, ap[:,:thickness,:]) ,axis=1)
    return ap2

def gridding_3D(absorption_map,label_list,coordinate_list,voxel_size,
                coefficients,crystal_coordinate,phiphi,thetatheta):
    shape=label_list.shape
    assert args.gridding_method != 3

    t1=time.time()
    for i, phi_row in enumerate(phiphi):
        for j, phi in enumerate(phi_row):
            theta=thetatheta[i][j]
            for k , index in enumerate( coordinate_list ) :
                coord = crystal_coordinate[index]

                face_2 = which_face_2( coord , shape , theta , phi )
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
                numbers_2 = cal_num( path_2 , voxel_size )
                if args.gridding_method == 1:
                    absorption = cal_rate( numbers_2 , coefficients )
                    absorption_map[i][j][k] = absorption
                elif args.gridding_method == 2:
                    absorption = cal_rate( numbers_2 , coefficients , exp = False )
                    absorption_map[i][j][k]=absorption


        t2 = time.time( )
        print( '[{}]/[{}],[{}]/[{}],[{}]/[{}]'.format( i , len( phiphi ) , j , len( phi_row ) , k ,
                                                       len( coordinate_list ) ) )
        print( 'time spent {}'.format( t2 - t1 ) )
    return absorption_map
        # np.save( "./gridding/{}_gridding_{}_{}_{}.npy".format( args.dataset,args.theta_num, args.phi_num,args.gridding_method) ,
        #             absorption_map )

def gridding_2D(absorption_map,label_list,coordinate_list,voxel_size,
                coefficients,crystal_coordinate,phiphi,thetatheta):
    shape=label_list.shape
    assert args.gridding_method == 3
    t1=time.time()

    for i, phi_row in enumerate(phiphi):
        for j, phi in enumerate(phi_row):
            theta=thetatheta[i][j]
            absorp= np.empty( len( coordinate_list ) )
            for k , index in enumerate( coordinate_list ) :
                coord = crystal_coordinate[index]
                face_2 = which_face_2( coord , shape , theta , phi )
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
                numbers_2 = cal_num( path_2 , voxel_size )
                absorption = cal_rate( numbers_2 , coefficients )
                absorp[k] = absorption

            absorption_map[i][j]= absorp.mean()
        t2 = time.time( )
        print( '[{}]/[{}],[{}]/[{}],[{}]/[{}]'.format( i , len( phiphi ) , j , len( phi_row ) , k ,
                                                       len( coordinate_list ) ) )
        print( 'time spent {}'.format( t2 - t1 ) )
    return absorption_map

def gridding(label_list,coordinate_list,voxel_size, coefficients,crystal_coordinate,phiphi,thetatheta):
    shape=label_list.shape

    m1=memorylog()
    if args.gridding_method == 3:
        absorption_map = np.zeros( phiphi.shape)
    else:
        absorption_map=np.zeros(phiphi.shape+(len(coordinate_list ),))
    m2=memorylog()
    print('memory usage on the absorption map is {}'.format((m2-m1)/ 1024 / 1024))
    t1=time.time()
    for i, phi_row in enumerate(phiphi):
        for j, phi in enumerate(phi_row):
            theta=thetatheta[i][j]
            absorp= np.empty( len( coordinate_list ) )
            for k , index in enumerate( coordinate_list ) :
                coord = crystal_coordinate[index]

                face_2 = which_face_2( coord , shape , theta , phi )
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
                numbers_2 = cal_num( path_2 , voxel_size )
                if args.gridding_method == 1:
                    absorption = cal_rate( numbers_2 , coefficients )
                    absorption_map[i][j][k] = absorption
                elif args.gridding_method == 2:
                    absorption = cal_rate( numbers_2 , coefficients , exp = False )
                    absorption_map[i][j][k]=absorption
                elif args.gridding_method == 3:
                    absorption = cal_rate( numbers_2 , coefficients )
                    absorp[k] = absorption
            if args.gridding_method == 3:
                absorption_map[i][j]= absorp.mean()

        t2 = time.time( )
        print( '[{}]/[{}],[{}]/[{}],[{}]/[{}]'.format( i , len( phiphi ) , j , len( phi_row ) , k ,
                                                       len( coordinate_list ) ) )
        print( 'time spent {}'.format( t2 - t1 ) )
        np.save( "./gridding/{}_gridding_{}_{}_{}.npy".format( args.dataset,args.theta_num, args.phi_num,args.gridding_method) ,
                    absorption_map )


def gridding_spherical(label_list,coordinate_list,voxel_size, coefficients,crystal_coordinate,phiphi,thetatheta):
    shape=label_list.shape
    theta_num=args.theta_num
    phi_num=args.phi_num
    theta_list = np.linspace( 0, 2*np.pi , num = theta_num )
    phi_list = np.linspace(-np.pi/2 , np.pi/2 , num = phi_num)
    m1=memorylog()
    if args.gridding_method == 3:
        absorption_map = np.zeros( phiphi.shape)
    else:
        absorption_map=np.zeros(phiphi.shape+(len(coordinate_list ),))
    m2=memorylog()
    print('memory usage on the absorption map is {}'.format((m2-m1)/ 1024 / 1024))
    t1=time.time()
    for i, phi_row in enumerate(phiphi):
        for j, phi in enumerate(phi_row):
            theta=thetatheta[i][j]
            absorp= np.empty( len( coordinate_list ) )
            for k , index in enumerate( coordinate_list ) :
                coord = crystal_coordinate[index]

                face_2 = which_face_2( coord , shape , theta , phi )
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
                numbers_2 = cal_num( path_2 , voxel_size )
                if args.gridding_method == 1:
                    absorption = cal_rate( numbers_2 , coefficients )
                    absorption_map[i][j][k] = absorption
                elif args.gridding_method == 2:
                    absorption = cal_rate( numbers_2 , coefficients , exp = False )
                    absorption_map[i][j][k]=absorption
                elif args.gridding_method == 3:
                    absorption = cal_rate( numbers_2 , coefficients )
                    absorp[k] = absorption
            if args.gridding_method == 3:
                absorption_map[i][j]= absorp.mean()

        t2 = time.time( )
        print( '[{}]/[{}],[{}]/[{}],[{}]/[{}]'.format( i , len( phiphi ) , j , len( phi_row ) , k ,
                                                       len( coordinate_list ) ) )
        print( 'time spent {}'.format( t2 - t1 ) )
        np.save( "./gridding/{}_gridding_{}_{}_{}.npy".format( args.dataset,args.theta_num, args.phi_num,args.gridding_method) ,
                    absorption_map )



def interpolation(theta_interp,phi_interp,theta ,phi,data,kernel_size=4 ):

    theta_index = np.abs( theta - theta_interp ).argmin( )
    phi_index= np.abs( phi - phi_interp ).argmin( )
    # Extract the four nearest data points and their values
    # print(theta_interp*180/np.pi)
    # print( phi_interp * 180 / np.pi )
    #
    increment=kernel_size//2
    if theta_index < 0 or theta_index+increment+1>len(theta) \
        or phi_index < 0 or phi_index+increment+1>len(phi):
        # if interpolate outside, concatenate the sides to beginning
        theta_index+=2
        phi_index+=2
        theta_front=theta[1:increment+1]+2*np.pi
        theta_end = theta[-increment-1:-1] - 2 * np.pi
        theta=np.concatenate((theta_end,theta,theta_front),axis=0)
        phi_front=phi[1:increment+1]+2*np.pi
        phi_end = phi[-increment-1:-1] - 2 * np.pi
        phi=np.concatenate((phi_end,phi,phi_front),axis=0)

        data_front=data[:,1:increment+1]
        data_end = data[: , -increment-1:-1]
        data=np.concatenate((data_end,data,data_front),axis=1)

        data_top=data[1:increment+1,:]
        data_bot = data[ -increment-1:-1,:]
        data=np.concatenate((data_bot ,data,data_top),axis=0)


    indices=[i for i in range(-increment+1,increment+1,1) ]
    theta_values=[ theta[theta_index+i] for i in indices ]
    phi_values = [phi[phi_index+i] for i in indices ]
    data_values=[]
    for i in indices:
        theta_list=[]
        for j in indices:
            theta_list.append(data[int(theta_index+i),int(phi_index+j)])
        data_values.append(theta_list)



    # Define the bilinear interpolation function
    try:
        # scipy version after 1.10.0
        interp_func = RegularGridInterpolator( (theta_values , phi_values) , data , method = 'slinear' )
    except:
        # scipy version before 1.10.0
        interp_func = interp2d( theta_values , phi_values , data_values , kind = 'linear' )
        # inter_func=SmoothSphereBivariateSpline()
        # inter_func=RectSphereBivariateSpline()

    interp_value = interp_func( theta_interp , phi_interp )

    return interp_value

def interpolation_v1 ( theta_interp , phi_interp , theta , phi , data ) :

    theta_index = np.abs( theta - theta_interp ).argmin( )
    phi_index = np.abs( phi - phi_interp ).argmin( )

    # Extract the four nearest data points and their values
    # print(theta_interp*180/np.pi)
    # print( phi_interp * 180 / np.pi )
    try :
        theta_values = [theta[theta_index - 1] , theta[theta_index] , theta[theta_index + 1] ,
                        theta[theta_index + 2]]
        phi_values = [phi[phi_index - 1] , phi[phi_index] , phi[phi_index + 1] , phi[phi_index + 2]]
        data_values = [[data[theta_index - 1 , phi_index - 1] , data[theta_index - 1 , phi_index] ,
                        data[theta_index - 1 , phi_index + 1] , data[theta_index - 1 , phi_index + 2]] ,
                       [data[theta_index , phi_index - 1] , data[theta_index , phi_index] ,
                        data[theta_index , phi_index + 1] , data[theta_index , phi_index + 2]] ,
                       [data[theta_index + 1 , phi_index - 1] , data[theta_index + 1 , phi_index] ,
                        data[theta_index + 1 , phi_index + 1] , data[theta_index + 1 , phi_index + 2]] ,
                       [data[theta_index + 2 , phi_index - 1] , data[theta_index + 2 , phi_index] ,
                        data[theta_index + 2 , phi_index + 1] , data[theta_index + 2 , phi_index + 2]]]
    except :
        try :
            theta_values = [theta[theta_index - 1] , theta[theta_index] , theta[theta_index + 1]]
            phi_values = [phi[phi_index - 1] , phi[phi_index] , phi[phi_index + 1]]
            data_values = [[data[theta_index - 1 , phi_index - 1] , data[theta_index - 1 , phi_index] ,
                            data[theta_index - 1 , phi_index + 1]] ,
                           [data[theta_index , phi_index - 1] , data[theta_index , phi_index] ,
                            data[theta_index , phi_index + 1]] ,
                           [data[theta_index + 1 , phi_index - 1] , data[theta_index + 1 , phi_index] ,
                            data[theta_index + 1 , phi_index + 1]]]
        except :
            return data[theta_index , phi_index]
    # Define the bilinear interpolation function
    interp_func = interp2d( theta_values , phi_values , data_values , kind = 'linear' )

    # Interpolate the data at the desired point
    interp_value = interp_func( theta_interp , phi_interp )

    return interp_value

def ray_tracing(coord,label_list,voxel_size, coefficients,theta , phi,exp=True):
    shape=label_list.shape
    face_2 = which_face_2( coord , shape , theta , phi )
    path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list )
    numbers_2 = cal_num( path_2 , voxel_size )
    absorption = cal_rate_single( numbers_2 , coefficients,exp=exp )

    return absorption

if __name__ == "__main__" :

    """label coordinate loading"""
    rate_list = {'li' : 1 , 'lo' : 2 , 'cr' : 3 , 'bu' : 4}
    path_l = ''
    dataset = args.dataset
    refl_filename = args.refl_path
    # './16846/DataFiles/16846cm31108v1_xdata1_SAD_SWEEP1_p-100_k-10.refl.json'
    expt_filename = args.expt_path  # only contain axes
    save_dir = args.save_dir
    # './save_data/{}_best_km10_{}_{}_okpv_combor'.format(dataset,args.expri,args.angle)
    if os.path.exists( save_dir ) is False :
        os.mkdir( save_dir )
    try:
        os.mkdir(save_dir+'/gridding')
        os.mkdir( save_dir + '/diff' )
    except:
        pass


    theta_num=args.theta_num
    phi_num=args.phi_num
    thickness=1
    theta_list = np.linspace( -np.pi , np.pi , num = theta_num )
    phi_list = np.linspace(-np.pi/2 , np.pi/2 , num = phi_num)
    # theta_list=np.concatenate(( np.array([-np.pi-np.pi*2/theta_num]) ,
    #                             theta_list,np.array([np.pi+np.pi*2/theta_num])   ),axis=0)
    # phi_list=np.concatenate(( np.array([-np.pi/2-np.pi/phi_num]) ,
    #                             phi_list,np.array([np.pi/2+np.pi/phi_num])   ),axis=0)
    # thetatheta , phiphi = np.meshgrid( theta_list , phi_list )
    phiphi ,thetatheta = np.meshgrid(phi_list , theta_list  )
    t1 = time.time( )
    m1 = memorylog()

    label_list = np.load(args.modelpath).astype(np.int8)
    m2 = memorylog()
    print('memory usage on the absorption map is {}'.format((m2 - m1) / 1024 / 1024))
    zz, yy, xx = np.where(label_list == rate_list['cr'])  # this line occupies 1GB, why???
    # crystal_coordinate = zip(zz, yy, xx)  # can be not listise to lower memory usage
    crystal_coordinate = np.stack((zz, yy, xx), axis = 1)
    print('memory usage on the absorption map is {}'.format((m2 - m1) / 1024 / 1024))
    # length_crystal_coordinate=len(list(crystal_coordinate))
    del zz, yy, xx  #
    m3 = memorylog()
    print('memory usage on the absorption map is {}'.format((m3 - m2) / 1024 / 1024))
    gc.collect()
    m2 = memorylog()

    """tomography setup """
    # pixel_size = 0.3e-3  # it means how large for a pixel of tomobar in real life
    pixel_size = args.pixel_size * 1e-3  # it means how large for a pixel of tomobar in real life
    """experiment in lab hyperparameter"""

    """ coefficients are calculated at 4.5kev but the diffraction experiment is in 3.5kev so scaling is needed, but don't know linear is good """
    mu_li = args.li * 1e3  # (unit in mm-1) 16010
    mu_lo = args.lo * 1e3
    mu_cr = args.cr * 1e3
    mu_bu = args.bu * 1e3

    coe = {'mu_li' : mu_li, 'mu_lo' : mu_lo, "mu_cr" : mu_cr}
    if args.full_iteration == 1 :
        full_iter = True
    elif args.full_iteration == 0 :
        full_iter = False

    shape = label_list.shape
    sampling = ada_sampling(crystal_coordinate, threshold = args.sampling_threshold)
    print("the chosen sampling is {}".format(sampling))

    seg = int(np.round(len(crystal_coordinate) / sampling))
    # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel
    coordinate_list = np.linspace(0, len(crystal_coordinate), num = seg, endpoint = False, dtype = int)
    print(" {} voxels are calculated".format(len(coordinate_list)))

    coefficients = mu_li , mu_lo , mu_cr , mu_bu
    voxel_size = [args.pixel_size_z * 1e-3 , args.pixel_size_y * 1e-3 , args.pixel_size_x * 1e-3]
        #gridding( label_list , coordinate_list , voxel_size , coefficients , crystal_coordinate , phiphi , thetatheta )
    if args.create_gridding==1:
        # create_gridding( label_list , coordinate_list , voxel_size ,
        #                  coefficients , crystal_coordinate , phiphi , thetatheta ,
        #                  num_processes = 2 , mp = False )
        gridding(label_list,coordinate_list,voxel_size, coefficients,crystal_coordinate,phiphi,thetatheta)
        sys.exit()

        # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel

    with open( expt_filename ) as f2 :
        axes_data = json.load( f2 )
    with open( refl_filename ) as f1 :
        data = json.load( f1 )
    print( 'The total size of the dataset is {}'.format( len( data ) ) )
    corr = []
    dict_corr = []

    low = args.low
    up = args.up

    if up == -1 :
        select_data = data[low :]
    else :
        select_data = data[low :up]

    del data

    axes = axes_data[0]
    # should be chagned

    kappa_axis = np.array( axes["axes"][1] )
    kappa = axes["angles"][1] / 180 * np.pi
    # kappa_axis=np.array(axes[2])
    # kappa=axes[4][1]/180*np.pi
    kappa_matrix = kp_rotation( kappa_axis , kappa )

    # phi_axis=np.array(axes[1])
    # phi=axes[4][0]/180*np.
    phi_axis = np.array( axes["axes"][0] )
    phi = axes["angles"][0] / 180 * np.pi
    phi_matrix = kp_rotation( phi_axis , phi )
    # https://dials.github.io/documentation/conventions.html#equation-diffractometer

    omega_axis = np.array( axes["axes"][2] )
    F = np.dot( kappa_matrix , phi_matrix )  # phi is the most intrinsic rotation, then kappa

    detector_gridding = np.load( "./gridding/{}_gridding_{}_{}_{}.npy".format( args.dataset  , args.theta_num ,
                                                           args.phi_num,args.gridding_method ) )

        # unit_test_sphere_transform( detector_gridding , thetatheta , phiphi )
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestAddNumbers)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    # ap2 = thicken_grid( detector_gridding )
    ap2= detector_gridding
    if args.gridding_method ==3:
        gridding_list=interp2d( phi_list ,theta_list,  ap2, kind = 'cubic' )
    elif args.gridding_method ==4:
        ap2 = spheretransformation( detector_gridding )
        first_part=phi_list[1:phi_num//2 ]
        after_part=phi_list[phi_num//2: ]
        phi_list_2=np.concatenate((after_part, first_part), axis=0)
        phi_list_2[phi_list_2<0]+=np.pi
        first_part = theta_list[1 :theta_num // 2]
        after_part = theta_list[theta_num // 2 :]
        theta_list_2 = np.concatenate( (after_part , first_part) , axis = 0 )
        theta_list_2[theta_list_2<0]+=2*np.pi
        #gridding_list_2=SmoothSphereBivariateSpline(phiphi_2.ravel(), thetatheta_2.ravel(),ap2.T.ravel())
        gridding_list_2 = RectSphereBivariateSpline(phi_list_2 , theta_list_2 ,ap2.T )
    else:
        gridding_list = [interp2d( phi_list , theta_list , ap2[: , : , k] , kind = 'linear' )
                         for k , index in enumerate( coordinate_list )]

    diff=[]
    corr_inter=[]
    phi_diff_threshold = phi_list[-1]-phi_list[-2]
    theta_diff_threshold = theta_list[-1]-theta_list[-2]
    for i , row in enumerate( select_data ) :

        intensity = float( row['intensity.sum.value'] )
        scattering_vector = literal_eval( row['s1'] )  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']
        lp = row['lp']

        rotation_frame_angle = literal_eval( row['xyzobs.mm.value'] )[2]
        rotation_frame_angle += args.offset / 180 * np.pi

        if rotation_frame_angle < 0 :
            if rotation_frame_angle < 2 * np.pi :
                rotation_frame_angle = 4 * np.pi + rotation_frame_angle
            else :
                rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2 * np.pi :
            if rotation_frame_angle > 4 * np.pi :
                rotation_frame_angle = rotation_frame_angle - 4 * np.pi
            else :
                rotation_frame_angle = rotation_frame_angle - 2 * np.pi

        assert rotation_frame_angle <= 2 * np.pi

        rotation_matrix_frame_omega = kp_rotation( omega_axis , rotation_frame_angle )
        total_rotation_matrix = np.dot( rotation_matrix_frame_omega , F )
        total_rotation_matrix = np.transpose( total_rotation_matrix )

        # xray=-np.array([0,0,1]) # should be chagned
        xray = -np.array( axes_data[1]["direction"] )
        xray = np.dot( total_rotation_matrix , xray )
        rotated_s1 = np.dot( total_rotation_matrix , scattering_vector )

        theta , phi = dials_2_thetaphi_22( rotated_s1 )
        theta_1 , phi_1 = dials_2_thetaphi_22( xray , L1 = True )
        print(theta, phi, theta_1, phi_1)
        # print('id {} with theta {}, phi {}, theta_1 {}, phi_1 {}'.format(id,theta,phi,theta_1,phi_1))
        theta_diff_1=np.abs(theta_1)-np.pi
        theta_diff_2 = np.abs( np.abs( theta ) - np.pi)
        phi_diff_1 = np.abs( np.abs( phi_1 ) - np.pi/2 )
        phi_diff_2 = np.abs( np.abs( phi ) - np.pi/2)

        d= np.empty( len( coordinate_list ) )
        absorp = np.ones( len( coordinate_list ) )
        absorp_inter =  np.ones( len( coordinate_list ) )


        for k , index in enumerate( coordinate_list ) :
            coord = crystal_coordinate[index]
            if args.compare_origin==1:
                face_1 = which_face_2( coord , shape , theta_1 , phi_1 )
                face_2 = which_face_2( coord , shape , theta , phi )
                path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 , shape , label_list)
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list)
                numbers_1 = cal_num( path_1 ,  voxel_size )  # 3.5s
                numbers_2 = cal_num(path_2 , voxel_size )
                absorption = cal_rate( numbers_1+numbers_2 , coefficients )
                absorp[k] = absorption
            else:
                absorption = 1
                if args.gridding_method > 3:
                    break

            if args.gridding_method==1:
                # inter2 = interpolation( theta , phi ,theta_list , phi_list , detector_gridding[:,:,k] )
                # inter1 = interpolation( theta_1 , phi_1 , theta_list , phi_list , detector_gridding[: , : , k] )
                # absorption_inter=inter1*inter2
                if  phi_diff_1 <phi_diff_threshold or theta_diff_1 < theta_diff_threshold:
                    inter1=ray_tracing(coord,label_list,voxel_size, coefficients,theta_1 , phi_1,exp=True)
                else:
                    inter1 = gridding_list[k]( phi_1 , theta_1 )
                if phi_diff_2 <phi_diff_threshold or theta_diff_2 < theta_diff_threshold:
                    inter2 = ray_tracing( coord , label_list , voxel_size , coefficients , theta , phi , exp = True )
                else:
                    inter2 = gridding_list[k]( phi , theta )

                # inter1 = gridding_list[k]( phi_1 , theta_1 )
                #
                # inter2 = gridding_list[k]( phi , theta )
                absorption_inter=inter1*inter2
            elif args.gridding_method==2:
                if  phi_diff_1 <phi_diff_threshold or theta_diff_1 < theta_diff_threshold:
                    inter1=ray_tracing(coord,label_list,voxel_size, coefficients,theta_1 , phi_1,exp=False)
                else:
                    inter1 = gridding_list[k]( phi_1 , theta_1 )

                if phi_diff_2 <phi_diff_threshold or theta_diff_2 < theta_diff_threshold:
                    inter2 = ray_tracing( coord , label_list , voxel_size , coefficients , theta , phi , exp = False )
                else:
                    inter2 = gridding_list[k]( phi , theta )

                # inter1 = gridding_list[k]( phi_1 , theta_1 )
                #
                # inter2 = gridding_list[k]( phi , theta )
                absorption_inter = np.exp( -(inter1 + inter2) )
            else:
                absorption_inter=-2
            absorp_inter[k]=absorption_inter
            if args.compare_origin == 1 :
                d[k]=np.abs((absorption_inter-absorption))/absorption

        if args.gridding_method==3:
            #absorp_inter_mean=interpolation( theta , phi , theta_list , phi_list , detector_gridding[: , : , k] )
            if phi_diff_1 < phi_diff_threshold or theta_diff_1 < theta_diff_threshold:
                absorp = np.ones( len( coordinate_list ) )
                for k , index in enumerate( coordinate_list ) :
                    coord = crystal_coordinate[index]
                    inter1 = ray_tracing( coord , label_list , voxel_size , coefficients , theta_1 , phi_1 , exp = True )
                    absorp[k] = inter1
                absorp_inter1=absorp.mean()
            else :
                absorp_inter1 = gridding_list( phi_1 , theta_1 )[0]

            if phi_diff_2 < phi_diff_threshold or theta_diff_2 < theta_diff_threshold:
                absorp = np.ones( len( coordinate_list ) )
                for k , index in enumerate( coordinate_list ) :
                    coord = crystal_coordinate[index]
                    inter2 = ray_tracing( coord , label_list , voxel_size , coefficients , theta , phi , exp =  True )
                    absorp[k] = inter2
                absorp_inter2=absorp.mean()

            else :
                absorp_inter2=gridding_list(phi,theta)[0]


            # absorp_inter2=gridding_list(phi,theta)[0]
            # absorp_inter1 = gridding_list( phi_1 , theta_1 )[0]
            absorp_inter_mean = absorp_inter1*absorp_inter2
            d_mean=np.abs(absorp.mean( )-absorp_inter_mean)/absorp.mean( )
        elif args.gridding_method==4:
            theta_2 , phi_2 = coord_transform( theta , phi )
            theta_12 , phi_12 = coord_transform( theta_1 , phi_1 )
            absorp_inter22  = gridding_list_2(phi_2,theta_2)[0][0]
            absorp_inter21 = gridding_list_2( phi_12 , theta_12 )[0][0]
            absorp_inter_mean = absorp_inter21*absorp_inter22
            d_mean=np.abs(absorp.mean( )-absorp_inter_mean)/absorp.mean( )
        else:
            absorp_inter_mean=absorp_inter.mean()
            d_mean = d.mean( )
        if args.compare_origin == 1 :
            print( d.mean( ) )
            print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , '
                       'rotation: {:.4f},  absorption: {:.4f},diff:{:.4f}'.format( low + i ,
                                                                                                            low + len(
                                                                                                                select_data ) ,
                                                                                                            theta * 180 / np.pi ,
                                                                                                            phi * 180 / np.pi ,
                                                                                                            rotation_frame_angle * 180 / np.pi ,
                                                                                                            absorp.mean( ),
                                                                                   d.mean( ) ) )
        else:
            print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , '
                       ' absorption: {:.4f}'.format( low + i ,
                                                     low + len(select_data ) ,
                                                     theta * 180 / np.pi ,
                                                     phi * 180 / np.pi ,
                                                     absorp_inter_mean) )
        t2 = time.time( )
        print( 'it spends {}'.format( t2 - t1 ) )
        
        if args.compare_origin == 1 :
            absorp_mean = absorp.mean( )
            diff.append(float(d_mean))
            corr.append( absorp_mean )
            dict_corr.append( {'index' : low + i , 'miller_index' : miller_index ,
                               'intensity' : intensity , 'corr' : absorp.mean( ) , 'lp' : lp} )
        corr_inter.append(absorp_inter_mean)
        if i % 500 == 1 :
            if args.compare_origin==1:
                with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
                    json.dump( corr , fz , indent = 2 )

                with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) ,
                           "w" ) as f1 :  # Pickling
                    json.dump( dict_corr , f1 , indent = 2 )
                with open( os.path.join( save_dir + '/diff' , "{}_refl_diff_{}.json".format( dataset , up ) ) ,
                           "w" ) as fz :  # Pickling
                    json.dump( diff , fz , indent = 2 )
            with open( os.path.join( save_dir+'/gridding' , "{}_refl_inter_{}.json".format( dataset , up ) ) ,
                       "w" ) as fz :  # Pickling
                json.dump( corr_inter , fz , indent = 2 )

    if args.compare_origin==1:
        with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
            json.dump( corr , fz , indent = 2 )

        with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) , "w" ) as f1 :  # Pickling
            json.dump( dict_corr , f1 , indent = 2 )
        with open( os.path.join( save_dir + '/diff' , "{}_refl_diff_{}.json".format( dataset , up ) ) ,
                   "w" ) as fz :  # Pickling
            json.dump( diff , fz , indent = 2 )
    with open( os.path.join( save_dir+'/gridding' , "{}_refl_inter_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
        json.dump( corr_inter , fz , indent = 2 )

    print( 'Finish!!!!' )







