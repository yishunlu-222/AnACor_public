import numpy as np
import pdb
from numba import jit
# from numba import int32, float32
import gc
from ast import literal_eval
import json
import time
import os
import ctypes as ct
import multiprocessing as mp
import psutil


try:
    from AnACor.Core_accelerated import  *
except:
    from Core_accelerated import *
#
# spec = [    ('value', int32),               # a simple scalar field
#     ('array', float32[:]),   ]

global rate_list
rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4,'other':5}


def cal_rate(numbers,coefficients,exp=True ):
    mu_li, mu_lo, mu_cr,mu_bu = coefficients

    if len(numbers)==8:
        li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
    else:
        li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
    if exp:
        abs = np.exp(-((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) )
                    ))
    else:
        abs = ((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) ))
    return  abs


def cal_path2_plus(path_2,voxel_size):

        voxel_length_z=voxel_size[0]
        voxel_length_y = voxel_size[1]
        voxel_length_x = voxel_size[2]
        path_ray = path_2[0]
        posi = path_2[1]
        classes = path_2[2]

        cr_l_2 = 0
        lo_l_2 = 0
        li_l_2 = 0
        bu_l_2 = 0


            # total_length = ( path_ray[-1][1] - path_ray[0][1] )/ (np.sin(np.abs(omega)))
        total_length=np.sqrt(((path_ray[-1][1]  - path_ray[0][1] ) * voxel_length_y ) ** 2 +
                            ((path_ray[-1][0]  - path_ray[0][0] ) * voxel_length_z ) ** 2 +
                            ( (path_ray[-1][2]  - path_ray[0][2] ) * voxel_length_x )** 2)
        for j, trans_index in enumerate(posi):

            if classes[j] == 'cr':
                if j < len(posi) - 1:
                    cr_l_2 += total_length * ( (posi[j+1]-posi[j])/len(path_ray))
                else:
                    cr_l_2 += total_length * ((len(path_ray)- posi[j]) / len(path_ray))
            elif classes[j] == 'li':
                if j < len(posi) - 1:
                    li_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
                else:
                    li_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
            elif classes[j] == 'lo':
                if j < len(posi) - 1:
                    lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
                else:
                    lo_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
            elif classes[j] == 'bu':
                if j < len(posi) - 1:
                    bu_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
                else:
                    bu_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
            else:
                pass
    
        return li_l_2, lo_l_2, cr_l_2,bu_l_2



@jit(nopython=True)
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

def slice_sampling(label_list, dim='z', sampling=5000, auto=True):

    # Find the indices of the non-zero elements directly
    crystal_coordinate = np.argwhere(label_list == rate_list['cr'])
    
    if auto:
        # When sampling ~= N/2000, the results become stable
        sampling = len(crystal_coordinate) // 2000
        print(" The sampling number is {}".format(sampling))
    
    output_lengths = []
    if dim == 'z':
        index = 0

    elif dim == 'y':
        index = 1

    elif dim == 'x':
        index = 2
    zz_u = np.unique(crystal_coordinate[:, index])
    
    # Sort the crystal_coordinate array using the np.argsort() function
    sorted_indices = np.argsort(crystal_coordinate[:, index])
    crystal_coordinate = crystal_coordinate[sorted_indices]
    # total_size=len(crystal_coordinate)

    # Use np.bincount() to count the number of occurrences of each value in the array
    output_lengths = np.bincount(crystal_coordinate[:, index], minlength=len(zz_u))
    zz_u= np.insert(zz_u,0,np.zeros(len(output_lengths)-len(zz_u)))
    # Compute the sampling distribution
    if sampling / len(output_lengths) < 0.5:
        sorted_indices = np.argsort(output_lengths)[::-1] # descending order
        sampling_distribution = np.zeros(len(output_lengths))
        sampling_distribution[sorted_indices[:sampling]] = 1
    else:
        sampling_distribution = np.round(output_lengths / output_lengths.mean() * sampling / len(output_lengths)).astype(int)
    
    coord_list = []

    # Use boolean indexing to filter the output array based on the sampling distribution
    for i, sampling_num in enumerate(sampling_distribution):
        if sampling_num == 0:
            continue
        # output_layer = crystal_coordinate[crystal_coordinate[:, index] == zz_u[i]]
        # Use np.random.choice() to randomly sample elements from the output arrays
        before=output_lengths[:i].sum() 
        after=output_lengths[:i + 1].sum()
        output_layer = crystal_coordinate[before: after]
        numbers=[]
        for k in range(sampling_num):
            
            numbers.append(int(output_lengths[i]/(sampling_num+1) * (k+1)) )

        for num in numbers:
            coord_list.append(output_layer[num])
        # sampled_indices = np.random.choice(range(len(output_layer)), size=int(sampling_num), replace=False)
        # coord_list.extend(output_layer[sampled_indices])
        # pdb.set_trace()

    return np.array(coord_list)

# def slice_sampling_v1(label_list,dim='z',sampling=5000,auto=True):
#     t1=time.time()
#     counter=0
#     zz, yy, xx = np.where(label_list == rate_list['cr'])  # this line occupies 1GB, why???
#     #crystal_coordinate = zip(zz, yy, xx)  # can be not listise to lower memory usage
#     if auto:
#         # when sampling ~= N/2000, the results become stable
#         sampling = len(zz)//2000
#         print(" the sampling number is {}".format(sampling))
#     crystal_coordinate = np.stack((zz,yy,xx),axis=1)


#     coord_list=[]

#     output = []
#     output_lengths=[]
#     if dim=='z':
#         index=0
#         zz_u=np.unique(zz)
#     elif dim=='y':
#         index=1
#         zz_u=np.unique(yy)
#     elif dim=='x':
#         index=2
#         zz_u=np.unique(xx)

#     # crystal_coordinate = np.sort(crystal_coordinate, axis=index)
#     crystal_coordinate= crystal_coordinate[crystal_coordinate[:,index].argsort()]
#     for i, z_value in enumerate(zz_u):
#         counter+=1
#         layer=[]
#         wherez=np.where(crystal_coordinate[:,index]==z_value)
#         for j in wherez[0]:
#             assert z_value==crystal_coordinate[j][index]
#             layer.append(crystal_coordinate[j])
#         output.append(np.array(layer))
#         output_lengths.append(len(np.array(layer)))
#     output_lengths=np.array(output_lengths)
#     sampling_distribution=np.zeros(len(output_lengths))
#     for i, lengths in enumerate(output_lengths):
#         counter+=1
#         if sampling/len(output_lengths) <  0.5:
#             sorted_indices = np.argsort(output_lengths)[::-1] # descending order
#             sampling_distribution[sorted_indices[:sampling]]=1
           
#         else:
#             sampling_num=np.round(lengths/output_lengths.mean()*sampling/len(output_lengths))
#             sampling_distribution[i]=sampling_num
    
#     # *sampling/len(output_lengths)
#     t3=time.time()
#     print("the time for old pre is {}".format(t3-t1))
#     for i, sampling_num in enumerate(sampling_distribution):
#         counter+=1
#         if sampling_num==0:
#             continue

            
#         numbers=[]
#         for k in range(int(sampling_num)):
#             counter+=1
#             numbers.append(int(output_lengths[i]/(int(sampling_num)+1) * (k+1)) )

#         for num in numbers:
#             counter+=1
#             coord_list.append(output[i][num])
#         # pdb.set_trace()
#     # print("\n the number of iteation for slice sampling is {} \n".format(counter))
#     t2=time.time()
#     print("the time for old is {}".format(t2-t1))
#     return np.array(coord_list)


def dials_2_numpy (vector ) :

    numpy_2_dials_1 = np.array( [[1 , 0 , 0] ,
                                    [0 , 0 , 1] ,
                                    [0 , 1 , 0]],dtype=np.float64 )

    back2 = numpy_2_dials_1.dot( vector )

    return back2


def dials_2_thetaphi( rotated_s1 , L1 = False ) :
        """
        dials_2_thetaphi_22
        :param rotated_s1: the ray direction vector in dials coordinate system
        :param L1: if it is the incident path, then the direction is reversed
        :return: the resolved theta, phi in the Raytracing coordinate system
        """
        if L1 is True :
            # L1 is the incident beam and L2 is the diffracted so they are opposite
            rotated_s1 = -rotated_s1

        if rotated_s1[1] == 0 :
            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
            theta = np.arctan( -rotated_s1[2] / (-np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) + 0.001) )
            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
            phi = np.arctan( -rotated_s1[0] / (rotated_s1[1] + 0.001) )
        else :
            if rotated_s1[1] < 0 :
                theta = np.arctan( -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = np.arctan( -rotated_s1[0] / (rotated_s1[1]) )
            else :
                if rotated_s1[2] < 0 :
                    theta = np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)

                else :
                    theta = - np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = - np.arctan( -rotated_s1[0] / (-rotated_s1[1]) )  # tan-1(-z/-x)
        return theta , phi
    



def cube_face (ray_origin , ray_direction , cube_size , L1 = False ) :
        """
        Determine which face of a cube a ray is going out.
        ray casting method
        To find the distance along the vector where the intersection point
        with a plane occurs, you can use the dot product of the vector and
          the plane normal to find the component of the vector that is
          perpendicular to the plane. Then, you can use this perpendicular
          component and the plane equation to solve for the distance along
          the vector to the intersection point.
          t = (plane_distance - np.dot(vector_origin, plane_normal)) /
             np.dot(vector, plane_normal)
        Args:
            ray_origin (tuple): the origin of the ray, as a tuple of (x, y, z) coordinates
            ray_direction (tuple): the direction of the ray, as a unit vector tuple of (x, y, z) coordinates
            cube_center (tuple): the center of the cube, as a tuple of (x, y, z) coordinates
            cube_size (float): the size of the cube, as a scalar value
        /*  'FRONTZY' = 1;
    *   'LEYX' = 2 ;
    *   'RIYX' = 3;
        'TOPZX' = 4;
        'BOTZX' = 5;
        "BACKZY" = 6 ;

        Returns:
            str: the name of the face that the ray intersects with first, or None if the ray doesn't intersect with the cube
        """
        # Determine the minimum and maximum x, y, and z coordinates of the cube

        # min_x = cube_center[0] - cube_size / 2
        # max_x = cube_center[0] + cube_size / 2
        # min_y = cube_center[1] - cube_size / 2
        # max_y = cube_center[1] + cube_size / 2
        # min_z = cube_center[2] - cube_size / 2
        # max_z = cube_center[2] + cube_size / 2
        # if L1 is True:
        #     ray_direction = -ray_direction
        # L1=False

        min_x = 0
        max_x = cube_size[2]
        min_y = 0
        max_y = cube_size[1]
        min_z = 0
        max_z = cube_size[0]
        # Calculate the t values for each face of the cube
        tx_min = (min_x - ray_origin[2]) / ray_direction[2]
        tx_max = (max_x - ray_origin[2]) / ray_direction[2]
        ty_min = (min_y - ray_origin[1]) / ray_direction[1]
        ty_max = (max_y - ray_origin[1]) / ray_direction[1]
        tz_min = (min_z - ray_origin[0]) / ray_direction[0]
        tz_max = (max_z - ray_origin[0]) / ray_direction[0]
        # print("tx min is {}".format(tx_min))
        # print("ty min  is {}".format(ty_min))
        # print("tz min  is {}".format(tz_min))
        # print("tx max  is {}".format(tx_max))
        # print("ty max  is {}".format(ty_max))
        # print("tz max  is {}".format(tz_max))
        # Determine which face is intersected first
        # t_mini = max( tx_min , ty_min , tz_min )
        # t_max = min( tx_max , ty_max , tz_max )
        t_numbers = [tx_min , ty_min , tz_min , tx_max , ty_max , tz_max]
        non_negative_numbers = [num for num in t_numbers if num >= 0]
        # if L1 is True:
        try :
            t_min = min( non_negative_numbers )
        except :
            t_min = max( non_negative_numbers )
            # print( "t_min is max at {}".format( ray_direction ) )
            # print( "t_min is max at {}".format( ray_origin ) )
        # else:
        # try:
        #         t_min=max(t_mini,t_max)
        # except:
        #     pdb.set_trace()

        # print(t_numbers)
        # pdb.set_trace()
        # if t_min > t_max :
        #     # The ray doesn't intersect with the cube
        #     return None
        if t_min == tx_min :
            # The ray intersects with the left face of the cube]
            if L1 is True :
                return "FRONTZY"
            else :
                return "BACKZY"
        elif t_min == tx_max :
            # The ray intersects with the right face of the cube
            if L1 is True :
                return "BACKZY"
            else :
                return "FRONTZY"
        elif t_min == ty_min :
            # The ray intersects with the bottom face of the cube
            if L1 is True :
                return 'BOTZX'
            else :
                return 'TOPZX'
        elif t_min == ty_max :
            # The ray intersects with the top face of the cube
            if L1 is True :
                return 'TOPZX'
            else :
                return 'BOTZX'
        elif t_min == tz_min :
            # The ray intersects with the front face of the cube
            if L1 is True :
                return 'RIYX'
            else :
                return 'LEYX'
        elif t_min == tz_max :
            # The ray intersects with the back face of the cube
            if L1 is True :
                return 'LEYX'
            else :
                return 'RIYX'
        else :
            pass
            # RuntimeError( 'face determination has a problem with direction {}'
            #               'and position {}'.format( ray_direction , ray_origin ) )


class RayTracingBasic(object):
    def __init__(self, args,printing=True):
        self.args=args
        self.printing=printing
        self.t1=time.time()
        try:
            with open(args.expt_path) as f2:
                axes_data = json.load(f2)
            print( "experimental data is loaded... \n" )
            with open(args.refl_path) as f1:
                data = json.load(f1)
            print( "reflection table is loaded... \n" )
            print('The total size of this calculation is {}'.format(len(data )))
        except:
            raise  RuntimeError('no reflections or experimental files detected'
                                'please use --refl-path --expt-path to specify')        

        if args.up == -1:
            select_data = data[args.low:]
        else:
            select_data = data[args.low:args.up]

        del data
        self.reflection_table =select_data 

        self.label_list = np.load(self.args.model_storepath).astype(np.int8)

        print("3D model is loaded... \n")
        mu_cr = self.args.crac*1e3  # (unit in mm-1) 16010
        mu_li = self.args.liac*1e3
        mu_lo = self.args.loac*1e3
        mu_bu = self.args.buac*1e3
        self.coefficients =  np.array([mu_li, mu_lo, mu_cr, mu_bu])
        self.offset = self.args.offset
        self.voxel_size =np.array([self.args.pixel_size_z* 1e-3 ,
                         self.args.pixel_size_y* 1e-3 ,
                         self.args.pixel_size_x* 1e-3 ])
        self.save_dir=self.args.save_dir
        self.dataset=self.args.dataset



        # self.reflections_table = reflections_table
        self.axes_data=axes_data
        self.rate_list = rate_list
        
        # zz , yy , xx = np.where( self.label_list == self.rate_list['cr'] )
        # self.crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
        # self.sampling = self.ada_sampling ( self.crystal_coordinate , threshold = sampling_threshold)
        # seg = int( np.round( len( self.crystal_coordinate ) / self.sampling ) )
        # # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel
        # self.coordinate_list = np.linspace( 0 , len( self.crystal_coordinate ) , num = seg , endpoint = False , dtype = int )
        
        self.coord_list = slice_sampling(self.label_list,dim=self.args.slicing,sampling=self.args.sampling_num,auto=self.args.auto_sampling)
        # old_coord_list=slice_sampling_v1(self.label_list,dim=self.args.slicing,sampling=self.args.sampling_num,auto=self.args.auto_sampling)
        print(" {} voxels are calculated".format(len(self.coord_list)))
        # pdb.set_trace()
        axes=self.axes_data[0]
        kappa_axis=np.array(axes["axes"][1])
        kappa = axes["angles"][1]/180*np.pi
        kappa_matrix = kp_rotation(kappa_axis, kappa)

        phi_axis=np.array(axes["axes"][0])
        phi = axes["angles"][0]/180*np.pi
        phi_matrix = kp_rotation(phi_axis, phi)
    #https://dials.github.io/documentation/conventions.html#equation-diffractometer
        self.xray = -np.array( self.axes_data[1]["direction"] )
        self.omega_axis=np.array(axes["axes"][2])
        self.F = np.dot(kappa_matrix , phi_matrix )   # phi is the most intrinsic rotation, then kappa
        self.rt_lib = ct.CDLL( os.path.join( os.path.dirname( os.path.abspath( __file__ )), './ray_tracing.so' ))
        if self.args.by_c :

            # gcc -shared -o ray_tracing.so ray_tracing.c -fPIC


            self.rt_lib.ray_tracing.restype = ct.c_double
            self.rt_lib.ray_tracing.argtypes = [
                np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # crystal_coordinate
                np.ctypeslib.ndpointer( dtype = np.int64 ) ,  # crystal_coordinate_shape
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
            self.rt_lib.ray_tracing_sampling.restype = ct.c_double
            self.rt_lib.ray_tracing_sampling.argtypes = [  # crystal_coordinate_shape
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

# crystal_coordinate_shape = np.array(crystal_coordinate.shape)



    def mp_run(self,printing=True,test=False):


        num_workers= self.args.num_workers
        len_data=len(self.reflection_table)
        each_core=int(len_data//num_workers)
        data_copies = [self.label_list.copy() for _ in range(num_workers)]
        self.args.t1=self.t1
        if test:
            self.run(0,len(self.reflection_table),
                                self.args,
                                self.reflection_table,
                                self.label_list,self.voxel_size,
                                self.coefficients,self.F,
                                self.omega_axis,self.xray,self.coord_list,
                                self.rt_lib,printing=printing)
            pdb.set_trace()
        # pdb.set_trace()
        # data_copies = [self.label_list.copy() for _ in range(num_workers)]

        # Create a list of worker processes
        processes = []
        for i in range(num_workers):
            # Create a new process and pass it the data copy and result queue
            if i!=num_workers-1:
                process = mp.Process(target=self.run, args=(i*each_core,(i+1)*each_core,
                                            self.args,
                                            self.reflection_table[i*each_core:(i+1)*each_core],
                                            data_copies[i],self.voxel_size,
                                            self.coefficients,self.F,
                                            self.omega_axis,self.xray,self.coord_list,
                                            self.rt_lib,
                                            printing))
                # process = mp.Process(target=self.test_mp, 
                #                     args=(data_copies[i],self.coord_list,
                #                           self.reflection_table[i*each_core:(i+1)*each_core]))
            else:
                process = mp.Process(target=self.run, 
                                    args=(i*each_core,'-1',
                                            self.args,
                                            self.reflection_table[i*each_core:],
                                            data_copies[i],self.voxel_size,
                                            self.coefficients,self.F,
                                            self.omega_axis,self.xray,self.coord_list,
                                            self.rt_lib,
                                            printing))
                # process = mp.Process(target=self.test_mp, 
                #                     args=(data_copies[i],self.coord_list,
                #                     self.reflection_table[i*each_core:]))
            processes.append(process)
        del self.label_list
        del self.reflection_table
        gc.collect()
        # Start all worker processes
        for process in processes:
            
            process.start()
            # time.sleep(20)
        # Wait for all worker processes to finish
        for process in processes:
            process.join()

    @staticmethod
    def test_mp(label_list,cord_list,select_data):
        print(label_list.shape)
        label_list[0][0][0]=1
        print(label_list[0][0][0])
        print(cord_list[0][0][0])
        print(select_data[0])

    @staticmethod
    def run(low,up,args,selected_data ,label_list,voxel_size,coefficients,F,omega_axis,xray,coord_list,rt_lib,printing=False):
        # if mp:
        #     label_list = self.label_list.copy()
        # else:
        #     label_list = self.label_list
        if args.by_c :
            label_list_c = python_2_c_3d( label_list )

        
        # if up == -1:
        #     selected_data = self.reflection_table[low:]
        # else:
        #     selected_data = self.reflection_table[low:up]

        corr = []
        dict_corr = []
        shape = np.array(label_list.shape)
        for i , row in enumerate( selected_data ) :
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

            
            xray = np.dot( total_rotation_matrix , xray )
            rotated_s1 = np.dot( total_rotation_matrix , scattering_vector )

            theta , phi = dials_2_thetaphi( rotated_s1 )
            theta_1 , phi_1 = dials_2_thetaphi( xray , L1 = True )
            # Get the current process
            process = psutil.Process()

            # Get the current memory usage in bytes
            memory_usage = process.memory_info().rss

            print(f"Current memory usage: {memory_usage} bytes")
            if args.by_c :
                result = rt_lib.ray_tracing_sampling(
                    coord_list , len( coord_list ) ,
                    rotated_s1 , xray , voxel_size ,
                    coefficients , label_list_c , shape ,
                    args.full_iteration , args.store_paths )
                # result = dials_lib.ray_tracing(crystal_coordinate, crystal_coordinate_shape,
                #                     coordinate_list,len(coordinate_list) ,
                #                     rotated_s1, xray, voxel_size,
                #                 coefficients, label_list_c, shape,
                #                 args.full_iteration, args.store_paths)
            else:
                ray_direction = dials_2_numpy( rotated_s1 )
                xray_direction = dials_2_numpy( xray )
                # absorp = np.empty(len(coordinate_list))
                # for k , index in enumerate( coordinate_list ) :
                #     coord = crystal_coordinate[index]
                absorp = np.empty( len( coord_list ) )
                for k , coord in enumerate( coord_list ) :
                    # face_1 = which_face_2(coord, shape, theta_1, phi_1)
                    # face_2 = which_face_2(coord, shape, theta, phi)
                    face_1 = cube_face( coord , xray_direction , shape , L1 = True )
                    face_2 = cube_face( coord , ray_direction , shape )
                    path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 , shape ,label_list )  # 37
                    path_2 = cal_coord_2( theta , phi , coord , face_2 , shape ,label_list )  # 16

                    numbers_1 = cal_path2_plus( path_1 , voxel_size )  # 3.5s
                    numbers_2 = cal_path2_plus( path_2 , voxel_size )  # 3.5s
                    if args.store_paths == 1 :
                        if k == 0 :
                            path_length_arr_single = np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )
                        else :

                            path_length_arr_single = np.concatenate(
                                (
                                path_length_arr_single , np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )) ,
                                axis = 0 )
                    absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )

                    absorp[k] = absorption

                if args.store_paths == 1 :
                    if i == 0 :
                        path_length_arr = np.expand_dims( path_length_arr_single , axis = 0 )
                    else :
                        path_length_arr = np.concatenate(
                            (path_length_arr , np.expand_dims( path_length_arr_single , axis = 0 )) , axis = 0 )
                result = absorp.mean( )
                #print( result )
            t2 = time.time( )
            corr.append( result )
            if printing:
                print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,
                                                                                                        low + len(
                                                                                                            selected_data ) ,
                                                                                                        theta * 180 / np.pi ,
                                                                                                        phi * 180 / np.pi ,
                                                                                                        rotation_frame_angle * 180 / np.pi ,
                                                                                                        result ) )
            # pdb.set_trace()

                print( 'process {} it spends {}'.format( os.getpid(),t2 -  args.t1 ) )
            dict_corr.append( {'index' : low + i , 'miller_index' : miller_index ,
                            'intensity' : intensity , 'corr' : result ,
                            'theta' : theta * 180 / np.pi ,
                            'phi' : phi * 180 / np.pi ,
                            'theta_1' : theta_1 * 180 / np.pi ,
                            'phi_1' : phi_1 * 180 / np.pi , } )
            if i % 1000 == 1 :
                if args.store_paths == 1 :
                    np.save( os.path.join(  args.save_dir , "{}_path_lengths_{}.npy".format(  args.dataset , up ) ) , path_length_arr )
                with open( os.path.join(  args.save_dir , "{}_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as fz :  # Pickling
                    json.dump( corr , fz , indent = 2 )
                with open( os.path.join(  args.save_dir , "{}_dict_refl_{}.json".format(  args.dataset , up ) ) ,
                        "w" ) as f1 :  # Pickling
                    json.dump( dict_corr , f1 , indent = 2 )
        if args.store_paths == 1 :
            np.save( os.path.join(  args.save_dir , "{}_path_lengths_{}.npy".format(  args.dataset , up ) ) , path_length_arr )
        with open( os.path.join(  args.save_dir , "{}_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as fz :  # Pickling
            json.dump( corr , fz , indent = 2 )

        with open( os.path.join(  args.save_dir , "{}_dict_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as f1 :  # Pickling
            json.dump( dict_corr , f1 , indent = 2 )
        with open( os.path.join( args.save_dir , "{}_time_{}.json".format(  args.dataset , up ) ) , "w" ) as f1 :  # Pickling
            json.dump( t2 -  args.t1 , f1 , indent = 2 )
        print( '{} ({} ) process is Finish!!!!'.format(os.getpid(),up) )



    @staticmethod
    def run(low,up,args,selected_data ,label_list,voxel_size,coefficients,F,omega_axis,xray,coord_list,rt_lib,printing=False):
        # if mp:
        #     label_list = self.label_list.copy()
        # else:
        #     label_list = self.label_list
        if args.by_c :
            label_list_c = python_2_c_3d( label_list )

        
        # if up == -1:
        #     selected_data = self.reflection_table[low:]
        # else:
        #     selected_data = self.reflection_table[low:up]

        corr = []
        dict_corr = []
        shape = np.array(label_list.shape)
        for i , row in enumerate( selected_data ) :
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

            
            xray = np.dot( total_rotation_matrix , xray )
            rotated_s1 = np.dot( total_rotation_matrix , scattering_vector )

            theta , phi = dials_2_thetaphi( rotated_s1 )
            theta_1 , phi_1 = dials_2_thetaphi( xray , L1 = True )

            if args.by_c :
                result = rt_lib.ray_tracing_sampling(
                    coord_list , len( coord_list ) ,
                    rotated_s1 , xray , voxel_size ,
                    coefficients , label_list_c , shape ,
                    args.full_iteration , args.store_paths )
                # result = dials_lib.ray_tracing(crystal_coordinate, crystal_coordinate_shape,
                #                     coordinate_list,len(coordinate_list) ,
                #                     rotated_s1, xray, voxel_size,
                #                 coefficients, label_list_c, shape,
                #                 args.full_iteration, args.store_paths)
            else:
                ray_direction = dials_2_numpy( rotated_s1 )
                xray_direction = dials_2_numpy( xray )
                # absorp = np.empty(len(coordinate_list))
                # for k , index in enumerate( coordinate_list ) :
                #     coord = crystal_coordinate[index]
                absorp = np.empty( len( coord_list ) )
                for k , coord in enumerate( coord_list ) :
                    # face_1 = which_face_2(coord, shape, theta_1, phi_1)
                    # face_2 = which_face_2(coord, shape, theta, phi)
                    face_1 = cube_face( coord , xray_direction , shape , L1 = True )
                    face_2 = cube_face( coord , ray_direction , shape )
                    path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 , shape ,label_list )  # 37
                    path_2 = cal_coord_2( theta , phi , coord , face_2 , shape ,label_list )  # 16

                    numbers_1 = cal_path2_plus( path_1 , voxel_size )  # 3.5s
                    numbers_2 = cal_path2_plus( path_2 , voxel_size )  # 3.5s
                    if args.store_paths == 1 :
                        if k == 0 :
                            path_length_arr_single = np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )
                        else :

                            path_length_arr_single = np.concatenate(
                                (
                                path_length_arr_single , np.expand_dims( np.array( (numbers_1 + numbers_2) ) , axis = 0 )) ,
                                axis = 0 )
                    absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )

                    absorp[k] = absorption

                if args.store_paths == 1 :
                    if i == 0 :
                        path_length_arr = np.expand_dims( path_length_arr_single , axis = 0 )
                    else :
                        path_length_arr = np.concatenate(
                            (path_length_arr , np.expand_dims( path_length_arr_single , axis = 0 )) , axis = 0 )
                result = absorp.mean( )
                #print( result )
            t2 = time.time( )
            corr.append( result )
            if printing:
                print( '[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format( low + i ,
                                                                                                        low + len(
                                                                                                            selected_data ) ,
                                                                                                        theta * 180 / np.pi ,
                                                                                                        phi * 180 / np.pi ,
                                                                                                        rotation_frame_angle * 180 / np.pi ,
                                                                                                        result ) )
            # pdb.set_trace()

                print( 'process {} it spends {}'.format( os.getpid(),t2 -  args.t1 ) )
            dict_corr.append( {'index' : low + i , 'miller_index' : miller_index ,
                            'intensity' : intensity , 'corr' : result ,
                            'theta' : theta * 180 / np.pi ,
                            'phi' : phi * 180 / np.pi ,
                            'theta_1' : theta_1 * 180 / np.pi ,
                            'phi_1' : phi_1 * 180 / np.pi , } )
            if i % 1000 == 1 :
                if args.store_paths == 1 :
                    np.save( os.path.join(  args.save_dir , "{}_path_lengths_{}.npy".format(  args.dataset , up ) ) , path_length_arr )
                with open( os.path.join(  args.save_dir , "{}_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as fz :  # Pickling
                    json.dump( corr , fz , indent = 2 )
                with open( os.path.join(  args.save_dir , "{}_dict_refl_{}.json".format(  args.dataset , up ) ) ,
                        "w" ) as f1 :  # Pickling
                    json.dump( dict_corr , f1 , indent = 2 )
        if args.store_paths == 1 :
            np.save( os.path.join(  args.save_dir , "{}_path_lengths_{}.npy".format(  args.dataset , up ) ) , path_length_arr )
        with open( os.path.join(  args.save_dir , "{}_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as fz :  # Pickling
            json.dump( corr , fz , indent = 2 )

        with open( os.path.join(  args.save_dir , "{}_dict_refl_{}.json".format(  args.dataset , up ) ) , "w" ) as f1 :  # Pickling
            json.dump( dict_corr , f1 , indent = 2 )
        with open( os.path.join( args.save_dir , "{}_time_{}.json".format(  args.dataset , up ) ) , "w" ) as f1 :  # Pickling
            json.dump( t2 -  args.t1 , f1 , indent = 2 )
        print( '{} ({} ) process is Finish!!!!'.format(os.getpid(),up) )



    # @staticmethod
    # def dials_2_numpy (vector ) :

    #     numpy_2_dials_1 = np.array( [[1 , 0 , 0] ,
    #                                  [0 , 0 , 1] ,
    #                                  [0 , 1 , 0]] )

    #     back2 = numpy_2_dials_1.dot( vector )

    #     return back2

    def ada_sampling ( self,crystal_coordinate , threshold = 10000 ) :

        num = len( crystal_coordinate )
        sampling = 1
        result = num
        while result > threshold :
            sampling = sampling * 2
            result = num / sampling

        return sampling

    @staticmethod
    def dials_2_thetaphi( rotated_s1 , L1 = False ) :
        """
        dials_2_thetaphi_22
        :param rotated_s1: the ray direction vector in dials coordinate system
        :param L1: if it is the incident path, then the direction is reversed
        :return: the resolved theta, phi in the Raytracing coordinate system
        """
        if L1 is True :
            # L1 is the incident beam and L2 is the diffracted so they are opposite
            rotated_s1 = -rotated_s1

        if rotated_s1[1] == 0 :
            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
            theta = np.arctan( -rotated_s1[2] / (-np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) + 0.001) )
            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
            phi = np.arctan( -rotated_s1[0] / (rotated_s1[1] + 0.001) )
        else :
            if rotated_s1[1] < 0 :
                theta = np.arctan( -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = np.arctan( -rotated_s1[0] / (rotated_s1[1]) )
            else :
                if rotated_s1[2] < 0 :
                    theta = np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)

                else :
                    theta = - np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = - np.arctan( -rotated_s1[0] / (-rotated_s1[1]) )  # tan-1(-z/-x)
        return theta , phi
    
    @staticmethod
    def which_face(coord , theta , phi,shape ) :
        """
        the face of the 3D model that the incident or diffracted passing through
        :param coord:   the point which was calculated the ray length
        :param shape:  shape of the tomography matrix
        :param theta: calculated theta angle to the point on the detector, positive means rotate clockwisely, vice versa
        :param phi: calculated phi angle to the point on the detector,positive means rotate clockwisely
        :return:  which face of the ray to exit, that represents the which (x,y,z) increment is 1

        top front left is the origin, not bottom front left

        """
        """ 
         the detector and the x-ray anti-clockwise rotation is positive  
        """
        # assert theta <= np.pi, phi <= np.pi/2
        z_max , y_max , x_max = shape
        x_max -= 1
        y_max -= 1
        z_max -= 1
        z , y , x = coord

        if np.abs( theta ) < np.pi / 2 :

            theta_up = np.arctan( (y - 0) / (x - 0 + 0.001) )
            theta_down = -np.arctan( (y_max - y) / (x - 0 + 0.001) )  # negative
            phi_right = np.arctan( (z_max - z) / (x - 0 + 0.001) )
            phi_left = -np.arctan( (z - 0) / (x - 0 + 0.001) )  # negative
            omega = np.arctan( np.tan( theta ) * np.cos( phi ) )

            if omega > theta_up :
                # at this case, theta is positive,
                # normally the most cases for theta > theta_up, the ray passes the top ZX plane
                # if the phis are smaller than both edge limits
                # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
                side = (y - 0) * np.sin( abs( phi ) ) / np.tan(
                    theta )  # the length of rotation is the projected length on x
                if side > (z - 0) and phi < phi_left :
                    face = 'LEYX'
                elif side > (z_max - z) and phi > phi_right :
                    face = 'RIYX'
                else :
                    face = 'TOPZX'

            elif omega < theta_down :
                side = (y_max - y) * np.sin( abs( phi ) ) / np.tan( -theta )
                if side > (z - 0) and phi < phi_left :
                    face = 'LEYX'
                elif side > (z_max - z) and phi > phi_right :
                    face = 'RIYX'
                else :
                    face = 'BOTZX'

            elif phi > phi_right :
                # when the code goes to this line, it means the theta is within the limits
                face = 'RIYX'
            elif phi < phi_left :
                face = 'LEYX'

            else :
                # ray passes through the back plane
                face = "BACKZY"

        else :
            # theta is larger than 90 degree or smaller than -90
            theta_up = np.arctan( (y - 0) / (x_max - x + 0.001) )
            theta_down = np.arctan( (y_max - y) / (x_max - x + 0.001) )  # negative
            phi_left = np.arctan( (z_max - z) / (x_max - x + 0.001) )  # it is the reverse of the top phi_left
            phi_right = -np.arctan( (z - 0) / (x_max - x + 0.001) )  # negative
            #
            #
            if (np.pi - theta) > theta_up and theta > 0 :
                # at this case, theta is positive,
                # normally the most cases for theta > theta_up, the ray passes the top ZX plane
                # if the phis are smaller than both edge limits
                # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
                side = (y - 0) * np.sin( abs( phi ) ) / np.abs( np.tan( theta ) )
                if side > (z - 0) and -phi < phi_right :
                    face = 'LEYX'
                elif side > (z_max - z) and -phi > phi_left :
                    face = 'RIYX'
                else :
                    face = 'TOPZX'
            #
            elif theta > theta_down - np.pi and theta <= 0 :
                side = (y_max - y) * np.sin( abs( phi ) ) / np.abs( np.tan( -theta ) )
                if side > (z - 0) and -phi < phi_right :
                    face = 'LEYX'
                elif side > (z_max - z) and -phi > phi_left :
                    face = 'RIYX'
                else :
                    face = 'BOTZX'

            elif -phi < phi_right :
                # when the code goes to this line, it means the theta is within the limits
                face = 'LEYX'
            elif -phi > phi_left :
                face = 'RIYX'

            else :
                # ray passes through the back plane
                face = 'FRONTZY'
        # pdb.set_trace()
        return face

    # @staticmethod
    # def cube_face (ray_origin , ray_direction , cube_size , L1 = False ) :
    #     """
    #     Determine which face of a cube a ray is going out.
    #     ray casting method
    #     To find the distance along the vector where the intersection point
    #     with a plane occurs, you can use the dot product of the vector and
    #       the plane normal to find the component of the vector that is
    #       perpendicular to the plane. Then, you can use this perpendicular
    #       component and the plane equation to solve for the distance along
    #       the vector to the intersection point.
    #       t = (plane_distance - np.dot(vector_origin, plane_normal)) /
    #          np.dot(vector, plane_normal)
    #     Args:
    #         ray_origin (tuple): the origin of the ray, as a tuple of (x, y, z) coordinates
    #         ray_direction (tuple): the direction of the ray, as a unit vector tuple of (x, y, z) coordinates
    #         cube_center (tuple): the center of the cube, as a tuple of (x, y, z) coordinates
    #         cube_size (float): the size of the cube, as a scalar value
    #     /*  'FRONTZY' = 1;
    # *   'LEYX' = 2 ;
    # *   'RIYX' = 3;
    #     'TOPZX' = 4;
    #     'BOTZX' = 5;
    #     "BACKZY" = 6 ;

    #     Returns:
    #         str: the name of the face that the ray intersects with first, or None if the ray doesn't intersect with the cube
    #     """
    #     # Determine the minimum and maximum x, y, and z coordinates of the cube

    #     # min_x = cube_center[0] - cube_size / 2
    #     # max_x = cube_center[0] + cube_size / 2
    #     # min_y = cube_center[1] - cube_size / 2
    #     # max_y = cube_center[1] + cube_size / 2
    #     # min_z = cube_center[2] - cube_size / 2
    #     # max_z = cube_center[2] + cube_size / 2
    #     # if L1 is True:
    #     #     ray_direction = -ray_direction
    #     # L1=False

    #     min_x = 0
    #     max_x = cube_size[2]
    #     min_y = 0
    #     max_y = cube_size[1]
    #     min_z = 0
    #     max_z = cube_size[0]
    #     # Calculate the t values for each face of the cube
    #     tx_min = (min_x - ray_origin[2]) / ray_direction[2]
    #     tx_max = (max_x - ray_origin[2]) / ray_direction[2]
    #     ty_min = (min_y - ray_origin[1]) / ray_direction[1]
    #     ty_max = (max_y - ray_origin[1]) / ray_direction[1]
    #     tz_min = (min_z - ray_origin[0]) / ray_direction[0]
    #     tz_max = (max_z - ray_origin[0]) / ray_direction[0]
    #     # print("tx min is {}".format(tx_min))
    #     # print("ty min  is {}".format(ty_min))
    #     # print("tz min  is {}".format(tz_min))
    #     # print("tx max  is {}".format(tx_max))
    #     # print("ty max  is {}".format(ty_max))
    #     # print("tz max  is {}".format(tz_max))
    #     # Determine which face is intersected first
    #     # t_mini = max( tx_min , ty_min , tz_min )
    #     # t_max = min( tx_max , ty_max , tz_max )
    #     t_numbers = [tx_min , ty_min , tz_min , tx_max , ty_max , tz_max]
    #     non_negative_numbers = [num for num in t_numbers if num >= 0]
    #     # if L1 is True:
    #     try :
    #         t_min = min( non_negative_numbers )
    #     except :
    #         t_min = max( non_negative_numbers )
    #         print( "t_min is max at {}".format( ray_direction ) )
    #         print( "t_min is max at {}".format( ray_origin ) )
    #     # else:
    #     # try:
    #     #         t_min=max(t_mini,t_max)
    #     # except:
    #     #     pdb.set_trace()

    #     # print(t_numbers)
    #     # pdb.set_trace()
    #     # if t_min > t_max :
    #     #     # The ray doesn't intersect with the cube
    #     #     return None
    #     if t_min == tx_min :
    #         # The ray intersects with the left face of the cube]
    #         if L1 is True :
    #             return "FRONTZY"
    #         else :
    #             return "BACKZY"
    #     elif t_min == tx_max :
    #         # The ray intersects with the right face of the cube
    #         if L1 is True :
    #             return "BACKZY"
    #         else :
    #             return "FRONTZY"
    #     elif t_min == ty_min :
    #         # The ray intersects with the bottom face of the cube
    #         if L1 is True :
    #             return 'BOTZX'
    #         else :
    #             return 'TOPZX'
    #     elif t_min == ty_max :
    #         # The ray intersects with the top face of the cube
    #         if L1 is True :
    #             return 'TOPZX'
    #         else :
    #             return 'BOTZX'
    #     elif t_min == tz_min :
    #         # The ray intersects with the front face of the cube
    #         if L1 is True :
    #             return 'RIYX'
    #         else :
    #             return 'LEYX'
    #     elif t_min == tz_max :
    #         # The ray intersects with the back face of the cube
    #         if L1 is True :
    #             return 'LEYX'
    #         else :
    #             return 'RIYX'
    #     else :
    #         RuntimeError( 'face determination has a problem with direction {}'
    #                       'and position {}'.format( ray_direction , ray_origin ) )

    @staticmethod
    def cal_rate(numbers,coefficients,exp=True ):
        mu_li, mu_lo, mu_cr,mu_bu = coefficients

        if len(numbers)==8:
            li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        else:
            li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
            li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
        if exp:
            abs = np.exp(-((mu_li * (li_l_1  + li_l_2) +
                        mu_lo * (lo_l_1  + lo_l_2) +
                        mu_cr * (cr_l_1 + cr_l_2) +
                            mu_bu * (bu_l_1+ bu_l_2) )
                        ))
        else:
            abs = ((mu_li * (li_l_1  + li_l_2) +
                        mu_lo * (lo_l_1  + lo_l_2) +
                        mu_cr * (cr_l_1 + cr_l_2) +
                            mu_bu * (bu_l_1+ bu_l_2) ))
        return  abs
    
    @staticmethod
    def cal_rate_single(numbers,coefficients,exp=True ):
        mu_li, mu_lo, mu_cr,mu_bu = coefficients
        assert len(numbers)==4
        li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers

        if exp:
            abs = np.exp(-((mu_li * ( li_l_2) +
                        mu_lo * ( lo_l_2) +
                        mu_cr * (cr_l_2) +
                            mu_bu * ( bu_l_2) )
                        ))
        else:
            abs = ((mu_li * (li_l_2) +
                        mu_lo * (lo_l_2) +
                        mu_cr * ( cr_l_2) +
                            mu_bu * (bu_l_2) ))
        return  abs



class RayTracingBisect(RayTracingBasic):
    def __init__(self, reflections_table,label_list,coefficients,
                 sampling=2000 ,offset=0,pixel_size = 0.3e-3,store_path=False ):
        # super(RayTracingCore,self).__init__()
        self.reflections = reflections_table
        self.label_list = label_list
        self.coefficients = coefficients
        self.offset = offset
        self.sampling = sampling
        self.pixel_size =pixel_size
        self.store_path=store_path
        # self.save_dir=save_dir
        # self.dataset=dataset
        # self.low=low
        # self.up=up
        self.rate_list = {'li' : 1 , 'lo' : 2 , 'cr' : 3 , 'bu' : 4}
        zz , yy , xx = np.where( self.label_list == self.rate_list['cr'] )
        self.crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
        seg = int( np.round( len( self.crystal_coordinate ) / self.sampling ) )
        # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel
        self.coordinate_list = np.linspace( 0 , len( self.crystal_coordinate ) , num = seg , endpoint = False , dtype = int )

    def run( self,xray , rotated_s1  ):

        theta , phi = self.dials_2_thetaphi( rotated_s1 )
        theta_1 , phi_1 = self.dials_2_thetaphi( xray , L1 = True )
        absorp = np.empty( len( self.coordinate_list ) )
        for k , index in enumerate( self.coordinate_list ) :
            coord = self.crystal_coordinate[index]
            # face_2 = which_face_2(coord, shape, theta, phi)  # 1s

            # face_1 = which_face_1_anti(coord, shape, rotation_frame_angle)  # 0.83
            face_1 = self.which_face( coord , theta_1 , phi_1 )
            face_2 = self.which_face( coord , theta , phi )

            path_1 = cal_coord_2( theta_1 , phi_1 , coord , face_1 ,self.label_list.shape,self.label_list)  # 37
            #            face_2 = which_face_matrix(coord,rotated_s1,shape)
            #            face_1 = which_face_matrix(coord,xray,shape,exit=False)
            #            path_1 = cal_coord_1_anti(rotation_frame_angle, coord, face_1, shape, label_list)
            path_2 = cal_coord_2( theta , phi , coord ,face_2,self.label_list.shape,self.label_list)  # 16
            numbers = self.cal_num( path_1 , path_2 )  # 3.5s
            if self.store_path:
                if k == 0 :
                    path_length_arr_single = np.expand_dims( np.array( numbers ) , axis = 0 )
                else :

                    path_length_arr_single = np.concatenate(
                        (path_length_arr_single , np.expand_dims( np.array( numbers ) , axis = 0 )) , axis = 0 )
            absorption = self.cal_rate( numbers , self.coefficients , self.pixel_size )
            absorp[k] = absorption

        if self.store_path :
            return  absorp.mean( ), path_length_arr_single
        else:
            return absorp.mean( )


        #            path_12=iterative_bisection(theta_1,phi_1,coord,face_1,label_list)
        #            path_22=iterative_bisection(theta, phi,coord,face_2,label_list)
        #            numbers_2 = cal_num22(coord,path_12,path_22,theta,rotation_frame_angle)
        #            absorption = cal_rate(numbers_2, coefficients, pixel_size)
        #            absorp[k] = absorption





    def dials_2_thetaphi(self, rotated_s1 , L1 = False ) :
        """
        dials_2_thetaphi_22
        :param rotated_s1: the ray direction vector in dials coordinate system
        :param L1: if it is the incident path, then the direction is reversed
        :return: the resolved theta, phi in the Raytracing coordinate system
        """
        if L1 is True :
            # L1 is the incident beam and L2 is the diffracted so they are opposite
            rotated_s1 = -rotated_s1

        if rotated_s1[1] == 0 :
            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
            theta = np.arctan( -rotated_s1[2] / (-np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) + 0.001) )
            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
            phi = np.arctan( -rotated_s1[0] / (rotated_s1[1] + 0.001) )
        else :
            if rotated_s1[1] < 0 :
                theta = np.arctan( -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = np.arctan( -rotated_s1[0] / (rotated_s1[1]) )
            else :
                if rotated_s1[2] < 0 :
                    theta = np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)

                else :
                    theta = - np.pi - np.arctan(
                        -rotated_s1[2] / np.sqrt( rotated_s1[0] ** 2 + rotated_s1[1] ** 2 ) )  # tan-1(y/-x)
                phi = - np.arctan( -rotated_s1[0] / (-rotated_s1[1]) )  # tan-1(-z/-x)
        return theta , phi

    def which_face(self, coord , theta , phi ) :
        """
        the face of the 3D model that the incident or diffracted passing through
        :param coord:   the point which was calculated the ray length
        :param shape:  shape of the tomography matrix
        :param theta: calculated theta angle to the point on the detector, positive means rotate clockwisely, vice versa
        :param phi: calculated phi angle to the point on the detector,positive means rotate clockwisely
        :return:  which face of the ray to exit, that represents the which (x,y,z) increment is 1

        top front left is the origin, not bottom front left

        """
        """ 
         the detector and the x-ray anti-clockwise rotation is positive  
        """
        # assert theta <= np.pi, phi <= np.pi/2
        z_max , y_max , x_max = self.label_list.shape
        x_max -= 1
        y_max -= 1
        z_max -= 1
        z , y , x = coord

        if np.abs( theta ) < np.pi / 2 :

            theta_up = np.arctan( (y - 0) / (x - 0 + 0.001) )
            theta_down = -np.arctan( (y_max - y) / (x - 0 + 0.001) )  # negative
            phi_right = np.arctan( (z_max - z) / (x - 0 + 0.001) )
            phi_left = -np.arctan( (z - 0) / (x - 0 + 0.001) )  # negative
            omega = np.arctan( np.tan( theta ) * np.cos( phi ) )

            if omega > theta_up :
                # at this case, theta is positive,
                # normally the most cases for theta > theta_up, the ray passes the top ZX plane
                # if the phis are smaller than both edge limits
                # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
                side = (y - 0) * np.sin( abs( phi ) ) / np.tan(
                    theta )  # the length of rotation is the projected length on x
                if side > (z - 0) and phi < phi_left :
                    face = 'LEYX'
                elif side > (z_max - z) and phi > phi_right :
                    face = 'RIYX'
                else :
                    face = 'TOPZX'

            elif omega < theta_down :
                side = (y_max - y) * np.sin( abs( phi ) ) / np.tan( -theta )
                if side > (z - 0) and phi < phi_left :
                    face = 'LEYX'
                elif side > (z_max - z) and phi > phi_right :
                    face = 'RIYX'
                else :
                    face = 'BOTZX'

            elif phi > phi_right :
                # when the code goes to this line, it means the theta is within the limits
                face = 'RIYX'
            elif phi < phi_left :
                face = 'LEYX'

            else :
                # ray passes through the back plane
                face = "BACKZY"

        else :
            # theta is larger than 90 degree or smaller than -90
            theta_up = np.arctan( (y - 0) / (x_max - x + 0.001) )
            theta_down = np.arctan( (y_max - y) / (x_max - x + 0.001) )  # negative
            phi_left = np.arctan( (z_max - z) / (x_max - x + 0.001) )  # it is the reverse of the top phi_left
            phi_right = -np.arctan( (z - 0) / (x_max - x + 0.001) )  # negative
            #
            #
            if (np.pi - theta) > theta_up and theta > 0 :
                # at this case, theta is positive,
                # normally the most cases for theta > theta_up, the ray passes the top ZX plane
                # if the phis are smaller than both edge limits
                # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
                side = (y - 0) * np.sin( abs( phi ) ) / np.abs( np.tan( theta ) )
                if side > (z - 0) and -phi < phi_right :
                    face = 'LEYX'
                elif side > (z_max - z) and -phi > phi_left :
                    face = 'RIYX'
                else :
                    face = 'TOPZX'
            #
            elif theta > theta_down - np.pi and theta <= 0 :
                side = (y_max - y) * np.sin( abs( phi ) ) / np.abs( np.tan( -theta ) )
                if side > (z - 0) and -phi < phi_right :
                    face = 'LEYX'
                elif side > (z_max - z) and -phi > phi_left :
                    face = 'RIYX'
                else :
                    face = 'BOTZX'

            elif -phi < phi_right :
                # when the code goes to this line, it means the theta is within the limits
                face = 'LEYX'
            elif -phi > phi_left :
                face = 'RIYX'

            else :
                # ray passes through the back plane
                face = 'FRONTZY'
        # pdb.set_trace()
        return face



    def cal_path2_plus ( self,path_2 ) :
        path_ray = path_2[0]
        posi = path_2[1]
        classes = path_2[2]

        cr_l_2 = 0
        lo_l_2 = 0
        li_l_2 = 0
        bu_l_2 = 0

        # total_length = ( path_ray[-1][1] - path_ray[0][1] )/ (np.sin(np.abs(omega)))
        total_length = np.sqrt( (path_ray[-1][1] - path_ray[0][1]) ** 2 +
                                (path_ray[-1][0] - path_ray[0][0]) ** 2 +
                                (path_ray[-1][2] - path_ray[0][2]) ** 2 )
        for j , trans_index in enumerate( posi ) :

            if classes[j] == 'cr' :
                if j < len( posi ) - 1 :
                    cr_l_2 += total_length * ((posi[j + 1] - posi[j]) / len( path_ray ))
                else :
                    cr_l_2 += total_length * ((len( path_ray ) - posi[j]) / len( path_ray ))
            elif classes[j] == 'li' :
                if j < len( posi ) - 1 :
                    li_l_2 += total_length * ((posi[j + 1] - posi[j]) / len( path_ray ))
                else :
                    li_l_2 += total_length * ((len( path_ray ) - posi[j]) / len( path_ray ))
            elif classes[j] == 'lo' :
                if j < len( posi ) - 1 :
                    lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / len( path_ray ))
                else :
                    lo_l_2 += total_length * ((len( path_ray ) - posi[j]) / len( path_ray ))
            elif classes[j] == 'bu' :
                if j < len( posi ) - 1 :
                    bu_l_2 += total_length * ((posi[j + 1] - posi[j]) / len( path_ray ))
                else :
                    bu_l_2 += total_length * ((len( path_ray ) - posi[j]) / len( path_ray ))
            else :
                pass

        return li_l_2 , lo_l_2 , cr_l_2 , bu_l_2
    def cal_num (self,  path_1 , path_2  ) :

        li_l_2 , lo_l_2 , cr_l_2 , bu_l_2 = self.cal_path2_plus( path_2  )
        if path_1 is not None :
            li_l_1 , lo_l_1 , cr_l_1 , bu_l_1 = self.cal_path2_plus( path_1 )
            return li_l_1 , lo_l_1 , cr_l_1 , bu_l_1 , li_l_2 , lo_l_2 , cr_l_2 , bu_l_2
        else :
            return li_l_2 , lo_l_2 , cr_l_2 , bu_l_2