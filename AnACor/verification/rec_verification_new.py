import numpy as np
from utils_rt import *
from analytical import ana_f_exit_sides , ana_f_exit_top
import pdb
import json
import time
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def worker_function(args):
    index, crystal_coordinate, shape, theta_1, phi_1, theta, phi, label_list, voxel_size, coefficients = args
    coord = crystal_coordinate[index]
    print(coord)
    face_1 = which_face(coord, shape, theta_1, phi_1)
    face_2 = which_face(coord, shape, theta, phi)

    path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list)
    path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list)

    numbers_1 = cal_path_plus(path_1, voxel_size)
    numbers_2 = cal_path_plus(path_2, voxel_size)

    absorption = cal_rate((numbers_1 + numbers_2), coefficients)
    return absorption

def rect_ana_ac_test(mu,angle, width,height,sampling):
    t1=time.time()
    coefficients = [0,0,mu,0]
    t_theta = angle / 180 * np.pi
    # resolution = 100  # must be the factor of 10
  
    try :
        T_l_2 = ana_f_exit_top( mu , t_theta , width , height )
    except :
        T_l_2 = ana_f_exit_sides( mu , t_theta , width , height )
    print(T_l_2)
    if np.isnan(T_l_2):
        return 0
    
    # mesh = np.zeros( (1 , int( height * resolution * 1.2 ) , int( width * resolution * 1.2 )) )
    #
    # mesh[: , int( height * resolution * 0.1 ) : int( height * resolution * 0.9 ) ,
    # int( width * resolution * 0.1 ) : int( width * resolution * 0.9 )] = 10.3875031836219992
    label_list = np.ones( (1 , int( height/ voxel_size[0]) , int( width/voxel_size[0]  )) ).astype(np.int8)   #0.3875031836219992
    shape=label_list.shape
    zz , yy , xx = np.where( label_list == 1 )
    crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
    seg = int(np.round(len(crystal_coordinate) / sampling))
    absorp = np.empty( len( crystal_coordinate  ) )
    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    del zz , yy , xx
    theta , phi =  angle / 180 * np.pi, 0/ 180 * np.pi
    theta_1 , phi_1 = 180 / 180 * np.pi, 0/ 180 * np.pi
    
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
   
    return np.abs(sum(absorptions) / len(absorptions)- T_l_2)/T_l_2



# def rect_ana_ac_test(mu,angle, width,height,resolution):
#     t1=time.time()
#     coefficients = [0,0,mu,0]
#     t_theta = angle / 180 * np.pi
#     # resolution = 100  # must be the factor of 10
#     pixel_size= min( width , height ) / resolution
#     try :
#         T_l_2 = ana_f_exit_top( mu , t_theta , width , height )
#     except :
#         T_l_2 = ana_f_exit_sides( mu , t_theta , width , height )
#     print( T_l_2 )
    
#     # mesh = np.zeros( (1 , int( height * resolution * 1.2 ) , int( width * resolution * 1.2 )) )
#     #
#     # mesh[: , int( height * resolution * 0.1 ) : int( height * resolution * 0.9 ) ,
#     # int( width * resolution * 0.1 ) : int( width * resolution * 0.9 )] = 10.3875031836219992
#     label_list = np.ones( (1 , int( height * resolution  ) , int( width * resolution  )) ).astype(np.int8)   #0.3875031836219992
#     shape=label_list.shape
#     zz , yy , xx = np.where( label_list == 1 )
#     crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
#     absorp = np.empty( len( crystal_coordinate  ) )

#     del zz , yy , xx
#     theta , phi =  angle / 180 * np.pi, 0/ 180 * np.pi
#     theta_1 , phi_1 = 180 / 180 * np.pi, 0/ 180 * np.pi

#     for k , index in enumerate( crystal_coordinate ) :
#         coord = index

#         face_1 = which_face( coord , shape , theta_1 , phi_1 )
#         face_2 = which_face( coord , shape , theta , phi )

#         path_1 = cal_coord( theta_1 , phi_1 , coord , face_1 , shape , label_list )  # 37
#         path_2 = cal_coord( theta , phi , coord , face_2 , shape , label_list )  # 16
#         numbers_1 = cal_path_plus( path_1 , voxel_size )  # 3.5s
#         numbers_2 = cal_path_plus( path_2 , voxel_size )
        
#         absorption = cal_rate( (numbers_1 + numbers_2) , coefficients )
#         if k % 10000 ==0:
#             print('[{}]/[{}]'.format(k, len(crystal_coordinate)))
#             print('absorption rate:{}'.format(absorption))
#             print('time spent:{}'.format(time.time()-t1))
#         absorp[k] = absorption

#         # if error_theta * 180 / np.pi >2:
#         #     pdb.set_trace()
#     print( absorp.mean( ) )
#     return np.abs(absorp.mean( ) - T_l_2)/T_l_2

if __name__ == '__main__':

    angle_list=np.linspace(start = 0,stop=90,num = 10,endpoint = True)

    mu=0.01 #um-1

    sampling=1
    for mur in [(150,150),(300,150),(150,300)]:
        width=mur[0]
        height=mur[1]
        # radius= mur/mu # mm
        voxel_size=[0.3,0.3,0.3] # um
        #resolution=int(min( width , height ) / voxel_size[0])
        errors=[]
        for angle in angle_list:
          er=rect_ana_ac_test(mu,angle, width,height,sampling) *100
          errors.append([ angle,er])

          with open("rect_sample_w_{}_h_{}_{}.json".format(width,height,voxel_size[0]), "w") as f1:  # Pickling
              json.dump(errors, f1, indent=2)
        

    # mu=1
    # width = 1
    # height = 1
    # resolution = 500
    # errors=[]
    # for angle in angle_list:
    #     er=rect_ana_ac_test(mu,angle, width,height,resolution) *100
    #     errors.append([ angle,er])

    # with open("rectangular sample 1 w1_h1.json", "w") as f1:  # Pickling
    #     json.dump(errors, f1, indent=2)
    # pdb.set_trace()