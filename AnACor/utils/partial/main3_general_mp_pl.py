import os
import json
# import pickle
# from matplotlib import pyplot as plt
# from multiprocessing import Process
import multiprocessing as mp
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils import *
from utils_lite import *
from sys import getsizeof
import resource
import  gc
import ctypes as ct
# ===========================================
#        Parse the argument
# ===========================================

def str2bool ( v ) :
    if isinstance( v , bool ) :
        return v
    if v.lower( ) in ('yes' , 'true' , 't' , 'y' , '1') :
        return True
    elif v.lower( ) in ('no' , 'false' , 'f' , 'n' , '0') :
        return False
    else :
        raise argparse.ArgumentTypeError( 'Boolean value expected.' )

parser = argparse.ArgumentParser(description="multiprocessing for batches")

parser.add_argument(
    "--low",
    type=int,
    default=0,
    help="the starting point of the batch",
)
parser.add_argument(
    "--up",
    type=int,
    default=-1,
    help="the ending point of the batch",
)
parser.add_argument(
    "--store-paths",
    type=int,
    default=0,
    help="orientation offset",
)

parser.add_argument(
    "--offset",
    type=float,
    default=0,
    help="orientation offset",
)

parser.add_argument(
    "--dataset",
    type=int,
    default=16846,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--modelpath",
    type=str,
    required = True,
    help="full model path",
)
parser.add_argument(
    "--save-dir",
    type=str,
    required = True,
    help="full storing path",
)
parser.add_argument(
    "--refl-path",
    type=str,
    required = True,
    help="full reflection path",
)
parser.add_argument(
    "--expt-path",
    type=str,
    required = True,
    help="full experiment path",
)
parser.add_argument(
    "--li",
    type=float,
    required = True,
    help="abs of liquor",
)
parser.add_argument(
    "--lo",
    type=float,
    required = True,
    help="abs of loop",
)
parser.add_argument(
    "--cr",
    type=float,
    required = True,
    help="abs of crystal",
)
parser.add_argument(
    "--bu",
    type=float,
    required = True,
    help="abs of other component",
)
parser.add_argument(
    "--sampling-num",
    type=int,
    default=5000,
    help="pixel size of tomography",
)
parser.add_argument(
    "--full-iteration",
    type=int,
    default=0,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-x",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-y",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--pixel-size-z",
    type=float,
    default=0.3,
    help="pixel size of tomography",
)
parser.add_argument(
    "--by-c",
    type=str2bool,
    default=True,
    help="pixel size of tomography",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="pixel size of tomography",
)
parser.add_argument(
    "--slicing",
    type=str,
    default='z',
    help="pixel size of tomography",
)
parser.add_argument(
    "--test-mode",
    type=str2bool,
    default=False,
    help="pixel size of tomography",
)
parser.add_argument(
    "--morecls",
    type=str2bool,
    default=False,
    help="pixel size of tomography",
)
parser.add_argument(
    "--partial-illumination",
    type=str2bool,
    default=True,
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
global args
args = parser.parse_args()

def ada_sampling(crystal_coordinate ,threshold=15000):
    num=len(crystal_coordinate)
    sampling=1
    result = num
    while result >threshold:
        sampling=sampling *2
        result = num/sampling

 
    return sampling
    

def kp_rotation(axis,theta, raytracing=True):
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param axis:
    :param theta:
    :return:
    """

    x,y,z = axis
    c =np.cos(theta)
    s = np.sin(theta)
    first_row = np.array([ c + (x**2)*(1-c), x*y*(1-c) - z*s, y*s + x*z*(1-c)  ])
    seconde_row = np.array([z*s + x*y*(1-c),  c + (y**2)*(1-c) , -x*s + y*z*(1-c) ])
    third_row = np.array([ -y*s + x*z*(1-c), x*s + y*z*(1-c), c + (z**2)*(1-c)  ])
    matrix = np.stack(( first_row, seconde_row, third_row), axis = 0)
    return matrix

def slice_sampling(label_list,dim='z',sampling=5000):
    zz, yy, xx = np.where(label_list == rate_list['cr'])  # this line occupies 1GB, why???
    #crystal_coordinate = zip(zz, yy, xx)  # can be not listise to lower memory usage
    crystal_coordinate = np.stack((zz,yy,xx),axis=1)


    coord_list=[]

    output = []
    output_lengths=[]
    if dim=='z':
        index=0
        zz_u=np.unique(zz)
    elif dim=='y':
        index=1
        zz_u=np.unique(yy)
    elif dim=='x':
        index=2
        zz_u=np.unique(xx)

    # crystal_coordinate = np.sort(crystal_coordinate, axis=index)
    crystal_coordinate= crystal_coordinate[crystal_coordinate[:,index].argsort()]
    for i, z_value in enumerate(zz_u):
        layer=[]
        wherez=np.where(crystal_coordinate[:,index]==z_value)
        for j in wherez[0]:
            assert z_value==crystal_coordinate[j][index]
            layer.append(crystal_coordinate[j])
        output.append(np.array(layer))
        output_lengths.append(len(np.array(layer)))
    output_lengths=np.array(output_lengths)
    sampling_distribution=np.zeros(len(output_lengths))
    for i, lengths in enumerate(output_lengths):
        if sampling/len(output_lengths) <  0.5:
            sorted_indices = np.argsort(output_lengths)[::-1] # descending order
            sampling_distribution[sorted_indices[:sampling]]=1
           
        else:
            sampling_num=np.round(lengths/output_lengths.mean()*sampling/len(output_lengths))
            sampling_distribution[i]=sampling_num
    
    # *sampling/len(output_lengths)
    for i, sampling_num in enumerate(sampling_distribution):
        if sampling_num==0:
            continue
        try:
#            numbers = random.sample(range(0, output_lengths[i]), int(sampling_num))
            #numbers= output_lengths[i]/(int(sampling_num)+1)
            
            numbers=[]
            for k in range(int(sampling_num)):
              numbers.append(int(output_lengths[i]/(int(sampling_num)+1) * (k+1)) )
#            if len(numbers) >2:
#            pdb.set_trace()
        except:
            pdb.set_trace()
        for num in numbers:
            coord_list.append(output[i][num])
    
    return np.array(coord_list)

def line_plane_intersection(P, direction, plane_normal, plane_point):
    denom = np.dot(plane_normal, direction)
    if np.abs(denom) < 1e-6:
        return np.NaN  # The line is parallel to the plane
    t = np.dot(plane_point - P, plane_normal) / denom
    return P + t * direction

def vector_passes_danger_region(P, direction, planes, zmin_danger, ymin_danger, xmin_danger, zmax_danger, ymax_danger, xmax_danger):
    # Determine the boundaries of the danger region

    intersections = np.array([line_plane_intersection(P, direction, plane[0], plane[1]) for plane in planes])
    try:
        valid_intersections = intersections[~np.isnan(intersections).any(axis=1)] 
        
        return  np.any(np.all((valid_intersections >= [zmin_danger, ymin_danger, xmin_danger]) & (valid_intersections <= [zmax_danger, ymax_danger, xmax_danger]), axis=1))
    except:
        print('it has a type error here',P,direction)
        print(intersections.dtype)
        return False

def worker_function(up,dataset,select_data ,label_list,voxel_size,coefficients,F,coord_list,rate_list,by_c=True):
    corr = []
    dict_corr = []
    shape = np.array(label_list.shape)
    if args.morecls:
        bubble_voxels = np.argwhere(label_list == rate_list['bu'])
            
        zmin_danger, ymin_danger, xmin_danger = np.min(bubble_voxels, axis=0)
        zmax_danger, ymax_danger, xmax_danger = np.max(bubble_voxels, axis=0)

        danger_planes=np.array([[[1, 0, 0]  ,[zmin_danger,(ymax_danger+ymin_danger)/2,(xmax_danger+xmin_danger)/2 ]],
                        [[-1, 0, 0] , [zmax_danger,(ymax_danger+ymin_danger)/2,(xmax_danger+xmin_danger)/2 ]],
                        [[0, 1, 0]  , [  (zmax_danger+zmin_danger)/2,ymin_danger,(xmax_danger+xmin_danger)/2 ]], 
                        [[0, -1, 0] , [  (zmax_danger+zmin_danger)/2,ymax_danger,(xmax_danger+xmin_danger)/2 ] ],
                        [[0, 0, 1]  , [  (zmax_danger+zmin_danger)/2,(ymax_danger+ymin_danger)/2,xmin_danger ] ],
                        [[0, 0, -1] , [  (zmax_danger+zmin_danger)/2,(ymax_danger+ymin_danger)/2,xmax_danger ] ]])

    # hull =ConvexHull(bubble_voxels)
    # bubble_surfaces = bubble_voxels[hull.vertices]
    if args.by_c :

        # class Thetaphi( ct.Structure ) :
        #     _fields_ = [("theta" , ct.c_double) ,
        #                 ("phi" , ct.c_double)]

        # class Vector3D( ct.Structure ) :
        #     _fields_ = [("x" , ct.c_int) ,
        #                 ("y" , ct.c_int) ,
        #                 ("z" , ct.c_int)]

        # class Path2( ct.Structure ) :
        #     _fields_ = [("ray" , ct.POINTER( Vector3D )) ,
        #                 ("posi" , ct.POINTER( ct.c_int )) ,
        #                 ("classes" , ct.POINTER( ct.c_char ))]


        # def python_2_c_2d ( arr_2d ) :
        #     labelPtr = ct.POINTER( ct.c_int )
        #     labelPtrPtr = ct.POINTER( labelPtr )
        #     labelPtrMatrix = labelPtr * label_list.shape[0]
        #     array_tuple = ()
        #     # Assign the numpy array to the pointer
        #     for row in arr_2d :
        #         array_tuple = array_tuple + (row.ctypes.data_as( labelPtr ) ,)
        #     arr_2d_ptr = ct.cast( labelPtrMatrix( *(array_tuple) ) , labelPtrPtr )
        #     return arr_2d_ptr

        def python_2_c_3d ( label_list ) :
            # this is a one 1d conversion
            # z, y, x = label_list.shape
            # label_list_ctype = (ct.c_int8 * z * y * x)()
            # for i in range(z):
            #     for j in range(y):
            #         for k in range(x):
            #             label_list_ctype[i][j][k] = ct.c_int8(label_list[i][j][k])
            labelPtr = ct.POINTER( ct.c_int8 )
            labelPtrPtr = ct.POINTER( labelPtr )
            labelPtrPtrPtr = ct.POINTER( labelPtrPtr )
            labelPtrCube = labelPtrPtr * label_list.shape[0]
            labelPtrMatrix = labelPtr * label_list.shape[1]
            matrix_tuple = ()
            for matrix in label_list :
                array_tuple = ()
                for row in matrix :
                    array_tuple = array_tuple + (row.ctypes.data_as( labelPtr ) ,)
                matrix_ptr = ct.cast( labelPtrMatrix( *(array_tuple) ) , labelPtrPtr )
                matrix_tuple = matrix_tuple + (matrix_ptr ,)
            label_list_ptr = ct.cast( labelPtrCube( *(matrix_tuple) ) , labelPtrPtrPtr )
            return label_list_ptr


        dials_lib = ct.CDLL( './ray_tracing.so' )
        # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

        dials_lib.ray_tracing_sampling.restype = ct.c_double
        dials_lib.ray_tracing_sampling.argtypes = [  # crystal_coordinate_shape
            np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # coordinate_list
            ct.c_int ,  # coordinate_list_length
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # rotated_s1
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # xray
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # voxel_size
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # coefficients
            ct.POINTER( ct.POINTER( ct.POINTER( ct.c_int8 ) ) ) ,  # label_list
            np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # shape
            ct.c_int ,  # full_iteration
            ct.c_int  # store_paths
        ]
        label_list_c = python_2_c_3d( label_list )
        # crystal_coordinate_shape = np.array(crystal_coordinate.shape)

    centre_point_on_axis=np.array([args.centre_point_z,
                                   args.centre_point_y, 
                                   args.centre_point_x])
    width =int(args.beam_width/1000/pixel_size/2)
    height=int(args.beam_height/1000/pixel_size/2)
    xray_region=[ centre_point_on_axis[1]-height,centre_point_on_axis[1]+height,centre_point_on_axis[0]-width,centre_point_on_axis[0]+width]  

    # pdb.set_trace()
    for i , row in enumerate( select_data ) :
        # try:
        #     print('up is {} in processor {}'.format( up+i,os.getpid() ))
        # except:
        #     print('up is {} in processor {}'.format( up,os.getpid() ))
        intensity = float( row['intensity.sum.value'] )
        scattering_vector = literal_eval( row['s1'] )  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']

        rotation_frame_angle = literal_eval( row['xyzobs.mm.value'] )[2]
        rotation_frame_angle += args.offset / 180 * np.pi
        rotation_matrix_frame_omega = kp_rotation( omega_axis , rotation_frame_angle )

        total_rotation_matrix = np.dot( rotation_matrix_frame_omega , F )
        total_rotation_matrix = np.transpose( total_rotation_matrix )

        xray = -np.array( axes_data[1]["direction"] )
        xray = np.dot( total_rotation_matrix , xray )
        rotated_s1 = np.dot( total_rotation_matrix , scattering_vector )

        theta , phi = dials_2_thetaphi_11( rotated_s1 )
        theta_1 , phi_1 = dials_2_thetaphi_11( xray , L1 = True )
        
        if args.by_c :
            result = dials_lib.ray_tracing_sampling(
                coord_list , len( coord_list ) ,
                rotated_s1 , xray , voxel_size ,
                coefficients , label_list_c , shape ,
                args.full_iteration , args.store_paths )
            # result = dials_lib.ray_tracing(crystal_coordinate, crystal_coordinate_shape,
            #                     coordinate_list,len(coordinate_list) ,
            #                     rotated_s1, xray, voxel_size,
            #                 coefficients, label_list_c, shape,
            #                 args.full_iteration, args.store_paths)
        else :
            ray_direction = dials_2_numpy_11( rotated_s1 )
            xray_direction = dials_2_numpy_11( xray )
            # absorp = np.empty(len(coordinate_list))
            # for k , index in enumerate( coordinate_list ) :
            #     coord = crystal_coordinate[index]
            absorp = []
            
            for k , coord in enumerate( coord_list ) :
                if args.partial_illumination :
                    pl = partial_illumination_selection(xray_region, total_rotation_matrix, coord, centre_point_on_axis)
                    
                    if pl is False:
                        continue
                    else:
                        pass
                face_1 = cube_face( coord , xray_direction , shape , L1 = True )
                face_2 = cube_face( coord , ray_direction , shape )
                path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 , shape , label_list ) # 37
                numbers_1 = cal_num( path_1 , voxel_size ) 
                # path_1 = cal_coord_mm( theta_1 , phi_1 , coord , face_1 , shape , label_list )
                # numbers_1 = cal_length_mm( path_1 , voxel_size )  # 3.5s
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list )
                numbers_2 = cal_num( path_2 , voxel_size )



                absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )

                absorp.append( absorption )


            result = np.array(absorp).mean( )
            #print( result )
        # print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,
        #                                                                                             low + len(
        #                                                                                                 select_data ) ,
        #                                                                                             theta * 180 / np.pi ,
        #                                                                                             phi * 180 / np.pi ,
        #                                                                                             rotation_frame_angle * 180 / np.pi ,
        #                                                                                             result ) )

        t2 = time.time( )
        corr.append( result )
        # print( 'it spends {}'.format( t2 - t1 ) )
        dict_corr.append( {'index' : low + i , 'miller_index' : miller_index ,
                           'intensity' : intensity , 'corr' : result ,
                           'theta' : theta * 180 / np.pi ,
                           'phi' : phi * 180 / np.pi ,
                           'theta_1' : theta_1 * 180 / np.pi ,
                           'phi_1' : phi_1 * 180 / np.pi , } )
        if i % 1000 == 1 :
            if args.store_paths == 1 :
                np.save( os.path.join( save_dir , "{}_path_lengths_{}.npy".format( dataset , up ) ) , path_length_arr )
            with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
                json.dump( corr , fz , indent = 2 )
            with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) ,
                       "w" ) as f1 :  # Pickling
                json.dump( dict_corr , f1 , indent = 2 )
    if args.store_paths == 1 :
        np.save( os.path.join( save_dir , "{}_path_lengths_{}.npy".format( dataset , up ) ) , path_length_arr )
    with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
        json.dump( corr , fz , indent = 2 )

    with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) , "w" ) as f1 :  # Pickling
        json.dump( dict_corr , f1 , indent = 2 )
    with open( os.path.join( save_dir , "{}_time_{}.json".format( dataset , up ) ) , "w" ) as f1 :  # Pickling
        json.dump( t2 - t1 , f1 , indent = 2 )
    print( '{} ({} ) process is Finish!!!!'.format(os.getpid(),up) )

def test_worker_function(up,dataset,select_data ,label_list,voxel_size,coefficients,F,coord_list,rate_list,by_c=True):
    corr = []
    dict_corr = []
    shape = np.array(label_list.shape)
    if args.morecls:
        bubble_voxels = np.argwhere(label_list == rate_list['bu'])
            
        zmin_danger, ymin_danger, xmin_danger = np.min(bubble_voxels, axis=0)
        zmax_danger, ymax_danger, xmax_danger = np.max(bubble_voxels, axis=0)

        danger_planes=np.array([[[1, 0, 0]  ,[zmin_danger,(ymax_danger+ymin_danger)/2,(xmax_danger+xmin_danger)/2 ]],
                        [[-1, 0, 0] , [zmax_danger,(ymax_danger+ymin_danger)/2,(xmax_danger+xmin_danger)/2 ]],
                        [[0, 1, 0]  , [  (zmax_danger+zmin_danger)/2,ymin_danger,(xmax_danger+xmin_danger)/2 ]], 
                        [[0, -1, 0] , [  (zmax_danger+zmin_danger)/2,ymax_danger,(xmax_danger+xmin_danger)/2 ] ],
                        [[0, 0, 1]  , [  (zmax_danger+zmin_danger)/2,(ymax_danger+ymin_danger)/2,xmin_danger ] ],
                        [[0, 0, -1] , [  (zmax_danger+zmin_danger)/2,(ymax_danger+ymin_danger)/2,xmax_danger ] ]])

    # hull =ConvexHull(bubble_voxels)
    # bubble_surfaces = bubble_voxels[hull.vertices]
    if args.by_c :

        # class Thetaphi( ct.Structure ) :
        #     _fields_ = [("theta" , ct.c_double) ,
        #                 ("phi" , ct.c_double)]

        # class Vector3D( ct.Structure ) :
        #     _fields_ = [("x" , ct.c_int) ,
        #                 ("y" , ct.c_int) ,
        #                 ("z" , ct.c_int)]

        # class Path2( ct.Structure ) :
        #     _fields_ = [("ray" , ct.POINTER( Vector3D )) ,
        #                 ("posi" , ct.POINTER( ct.c_int )) ,
        #                 ("classes" , ct.POINTER( ct.c_char ))]


        # def python_2_c_2d ( arr_2d ) :
        #     labelPtr = ct.POINTER( ct.c_int )
        #     labelPtrPtr = ct.POINTER( labelPtr )
        #     labelPtrMatrix = labelPtr * label_list.shape[0]
        #     array_tuple = ()
        #     # Assign the numpy array to the pointer
        #     for row in arr_2d :
        #         array_tuple = array_tuple + (row.ctypes.data_as( labelPtr ) ,)
        #     arr_2d_ptr = ct.cast( labelPtrMatrix( *(array_tuple) ) , labelPtrPtr )
        #     return arr_2d_ptr

        def python_2_c_3d ( label_list ) :
            # this is a one 1d conversion
            # z, y, x = label_list.shape
            # label_list_ctype = (ct.c_int8 * z * y * x)()
            # for i in range(z):
            #     for j in range(y):
            #         for k in range(x):
            #             label_list_ctype[i][j][k] = ct.c_int8(label_list[i][j][k])
            labelPtr = ct.POINTER( ct.c_int8 )
            labelPtrPtr = ct.POINTER( labelPtr )
            labelPtrPtrPtr = ct.POINTER( labelPtrPtr )
            labelPtrCube = labelPtrPtr * label_list.shape[0]
            labelPtrMatrix = labelPtr * label_list.shape[1]
            matrix_tuple = ()
            for matrix in label_list :
                array_tuple = ()
                for row in matrix :
                    array_tuple = array_tuple + (row.ctypes.data_as( labelPtr ) ,)
                matrix_ptr = ct.cast( labelPtrMatrix( *(array_tuple) ) , labelPtrPtr )
                matrix_tuple = matrix_tuple + (matrix_ptr ,)
            label_list_ptr = ct.cast( labelPtrCube( *(matrix_tuple) ) , labelPtrPtrPtr )
            return label_list_ptr


        dials_lib = ct.CDLL( './ray_tracing.so' )
        # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC

        dials_lib.ray_tracing_sampling.restype = ct.c_double
        dials_lib.ray_tracing_sampling.argtypes = [  # crystal_coordinate_shape
            np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # coordinate_list
            ct.c_int ,  # coordinate_list_length
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # rotated_s1
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # xray
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # voxel_size
            np.ctypeslib.ndpointer( dtype = np.float64 ) ,  # coefficients
            ct.POINTER( ct.POINTER( ct.POINTER( ct.c_int8 ) ) ) ,  # label_list
            np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # shape
            ct.c_int ,  # full_iteration
            ct.c_int  # store_paths
        ]
        label_list_c = python_2_c_3d( label_list )
        # crystal_coordinate_shape = np.array(crystal_coordinate.shape)
    centre_point_on_axis=np.array([529,484,0])
    width =int(0.170/pixel_size/2)
    height=int(0.130/pixel_size/2)
    height =int(0.170/pixel_size/2)
    width=int(0.130/pixel_size/2)
    xray_region=[ centre_point_on_axis[1]-height,centre_point_on_axis[1]+height,centre_point_on_axis[0]-width,centre_point_on_axis[0]+width]  

    # new1=label_list.mean(axis=2)
    # plt.imshow(new1)
    # x1, y1 = xray_region[0],xray_region[2]
    # x2, y2 = xray_region[1],xray_region[3]
    # plt.plot(centre_point_on_axis[1],centre_point_on_axis[0], 'ro-')
    # plt.plot([x1, x2], [y2, y2], 'go-')
    # plt.axvline(x=x1, color='blue', linestyle='--')
    # plt.axvline(x=x2, color='blue', linestyle='--')
    # plt.axhline(y=y1, color='blue', linestyle='--')
    # plt.axhline(y=y2, color='blue', linestyle='--')
    # pdb.set_trace()
    for i , row in enumerate( select_data ) :
        # try:
        #     print('up is {} in processor {}'.format( up+i,os.getpid() ))
        # except:
        #     print('up is {} in processor {}'.format( up,os.getpid() ))
        intensity = float( row['intensity.sum.value'] )
        scattering_vector = literal_eval( row['s1'] )  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']

        rotation_frame_angle = literal_eval( row['xyzobs.mm.value'] )[2]
        rotation_frame_angle += args.offset / 180 * np.pi
        rotation_matrix_frame_omega = kp_rotation( omega_axis , rotation_frame_angle )

        total_rotation_matrix = np.dot( rotation_matrix_frame_omega , F )
        total_rotation_matrix = np.transpose( total_rotation_matrix )

        xray = -np.array( axes_data[1]["direction"] )
        xray = np.dot( total_rotation_matrix , xray )
        rotated_s1 = np.dot( total_rotation_matrix , scattering_vector )

        theta , phi = dials_2_thetaphi_11( rotated_s1 )
        theta_1 , phi_1 = dials_2_thetaphi_11( xray , L1 = True )
        
        if args.by_c :
            result = dials_lib.ray_tracing_sampling(
                coord_list , len( coord_list ) ,
                rotated_s1 , xray , voxel_size ,
                coefficients , label_list_c , shape ,
                args.full_iteration , args.store_paths )
            # result = dials_lib.ray_tracing(crystal_coordinate, crystal_coordinate_shape,
            #                     coordinate_list,len(coordinate_list) ,
            #                     rotated_s1, xray, voxel_size,
            #                 coefficients, label_list_c, shape,
            #                 args.full_iteration, args.store_paths)
        else :
            ray_direction = dials_2_numpy_11( rotated_s1 )
            xray_direction = dials_2_numpy_11( xray )
            # absorp = np.empty(len(coordinate_list))
            # for k , index in enumerate( coordinate_list ) :
            #     coord = crystal_coordinate[index]
            absorp = []
            
            for k , coord in enumerate( coord_list ) :
                if args.partial_illumination :
                    pl = partial_illumination_selection(xray_region, total_rotation_matrix.T, coord, centre_point_on_axis)
                    
                    if pl is False:
                        continue
                    else:
                        pass
                face_1 = cube_face( coord , xray_direction , shape , L1 = True )
                face_2 = cube_face( coord , ray_direction , shape )
                path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 , shape , label_list ) # 37
                numbers_1 = cal_num( path_1 , voxel_size ) 
                path_1 = cal_coord_mm( theta_1 , phi_1 , coord , face_1 , shape , label_list )
                numbers_1 = cal_length_mm( path_1 , voxel_size )  # 3.5s
                path_2 = cal_coord_2( theta , phi , coord , face_2 , shape , label_list )
                numbers_2 = cal_num( path_2 , voxel_size )



                absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )

                absorp.append( absorption )


            result = np.array(absorp).mean( )
            #print( result )
        # print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,
        #                                                                                             low + len(
        #                                                                                                 select_data ) ,
        #                                                                                             theta * 180 / np.pi ,
        #                                                                                             phi * 180 / np.pi ,
        #                                                                                             rotation_frame_angle * 180 / np.pi ,
        #                                                                                             result ) )
        pdb.set_trace()
        t2 = time.time( )
        corr.append( result )
        # print( 'it spends {}'.format( t2 - t1 ) )
        dict_corr.append( {'index' : low + i , 'miller_index' : miller_index ,
                           'intensity' : intensity , 'corr' : result ,
                           'theta' : theta * 180 / np.pi ,
                           'phi' : phi * 180 / np.pi ,
                           'theta_1' : theta_1 * 180 / np.pi ,
                           'phi_1' : phi_1 * 180 / np.pi , } )
        if i % 1000 == 1 :
            if args.store_paths == 1 :
                np.save( os.path.join( save_dir , "{}_path_lengths_{}.npy".format( dataset , up ) ) , path_length_arr )
            with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
                json.dump( corr , fz , indent = 2 )
            with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) ,
                       "w" ) as f1 :  # Pickling
                json.dump( dict_corr , f1 , indent = 2 )
    if args.store_paths == 1 :
        np.save( os.path.join( save_dir , "{}_path_lengths_{}.npy".format( dataset , up ) ) , path_length_arr )
    with open( os.path.join( save_dir , "{}_refl_{}.json".format( dataset , up ) ) , "w" ) as fz :  # Pickling
        json.dump( corr , fz , indent = 2 )

    with open( os.path.join( save_dir , "{}_dict_refl_{}.json".format( dataset , up ) ) , "w" ) as f1 :  # Pickling
        json.dump( dict_corr , f1 , indent = 2 )
    with open( os.path.join( save_dir , "{}_time_{}.json".format( dataset , up ) ) , "w" ) as f1 :  # Pickling
        json.dump( t2 - t1 , f1 , indent = 2 )
    print( '{} ({} ) process is Finish!!!!'.format(os.getpid(),up) )


if __name__ == "__main__":

    """label coordinate loading"""
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
    path_l = ''
    dataset = args.dataset
    label_list = np.load(args.modelpath).astype(np.int8)
    refl_filename = args.refl_path
    expt_filename = args.expt_path   # only contain axes
    save_dir = args.save_dir
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    # zz, yy, xx = np.where(label_list == rate_list['cr'])  
    # crystal_coordinate = np.stack((zz,yy,xx),axis=1)
    # del zz, yy, xx  #
    # gc.collect()
    # sampling = ada_sampling(crystal_coordinate )
    # sampling=2000
    # print("the chosen sampling is {}".format(sampling))

    # seg = int(np.round(len(crystal_coordinate) / sampling))
    # coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    # print(" {} voxels are calculated".format(len(coordinate_list)))

    coord_list = slice_sampling(label_list,dim=args.slicing,sampling=args.sampling_num)
    print(" {} voxels are calculated".format(len(coord_list)))
    """tomography setup """
    pixel_size = args.pixel_size * 1e-3  # it means how large for a pixel of tomobar in real life

    mu_li = args.li*1e3    # (unit in mm-1) 16010
    mu_lo = args.lo*1e3
    mu_cr = args.cr*1e3
    mu_bu=args.bu*1e3
    #
    t1 = time.time()


    with open(expt_filename) as f2:
        axes_data = json.load(f2)
    with open(refl_filename) as f1:
        data = json.load(f1)
    print('The total size of the dataset is {}'.format(len(data)))

    voxel_size=np.array([args.pixel_size_z* 1e-3 ,
                         args.pixel_size_y* 1e-3 ,
                         args.pixel_size_x* 1e-3 ])
    low = args.low
    up = args.up

    if up == -1:
        select_data = data[low:]
    else:
        select_data = data[low:up]

    del data
    coefficients = np.array([mu_li, mu_lo, mu_cr, mu_bu])

    num_workers= args.num_workers
    len_data=len(select_data)
    each_core=int(len_data//num_workers)
    
    axes=axes_data[0]
 # should be chagned

    kappa_axis=np.array(axes["axes"][1])
    kappa = axes["angles"][1]/180*np.pi
    kappa_matrix = kp_rotation(kappa_axis, kappa)

    phi_axis=np.array(axes["axes"][0])
    phi = axes["angles"][0]/180*np.pi
    phi_matrix = kp_rotation(phi_axis, phi)
  #https://dials.github.io/documentation/conventions.html#equation-diffractometer

    omega_axis=np.array(axes["axes"][2])
    F = np.dot(kappa_matrix , phi_matrix )   # phi is the most intrinsic rotation, then kappa






    # Create a list of 48 data copies
    data_copies = [label_list.copy() for _ in range(num_workers)]

    # Create a queue to store the results from each worker process
    result_queue = mp.Queue()

    # Create a list of worker processes
    processes = []
    if args.test_mode:
        test_worker_function('-1',dataset,select_data ,data_copies[0],
                        voxel_size,coefficients,F, 
                        coord_list,rate_list)


    for i in range(num_workers):
        # Create a new process and pass it the data copy and result queue
        if i!=num_workers-1:
            process = mp.Process(target=worker_function, 
                                args=((i+1)*each_core,dataset,select_data[i*each_core:(i+1)*each_core] ,data_copies[i],
                                    voxel_size,coefficients,F, 
                                    coord_list,rate_list))
        else:
            process = mp.Process(target=worker_function, 
                                args=('-1',dataset,select_data[i*each_core:] ,data_copies[i],
                                    voxel_size,coefficients,F, 
                                    coord_list,rate_list))
        processes.append(process)
    # pdb.set_trace()
    # Start all worker processes
    for process in processes:
        process.start()

    # Wait for all worker processes to finish
    for process in processes:
        process.join()

    # # Combine the results from each worker process
    # final_result = 0










