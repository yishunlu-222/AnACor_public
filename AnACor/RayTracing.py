import numpy as np
import pdb
from numba import jit
# from numba import int32, float32
# from ast import literal_eval
# import json
# import time
# import os
try:
    from AnACor.Core_accelerated import  *
except:
    from Core_accelerated import *
#
# spec = [    ('value', int32),               # a simple scalar field
#     ('array', float32[:]),   ]


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

class RayTracingBasic(object):
    def __init__(self, reflections_table,label_list,coefficients,
                 sampling_threshold=5000 ,offset=0,pixel_size = 0.3e-3,store_path=False ):
        # super(RayTracingCore,self).__init__()
        self.reflections = reflections_table
        self.label_list = label_list
        self.coefficients = coefficients
        self.offset = offset
        self.pixel_size =pixel_size
        self.store_path=store_path
        # self.save_dir=save_dir
        # self.dataset=dataset
        # self.low=low
        # self.up=up
        self.rate_list = {'li' : 1 , 'lo' : 2 , 'cr' : 3 , 'bu' : 4}
        zz , yy , xx = np.where( self.label_list == self.rate_list['cr'] )
        self.crystal_coordinate = np.stack( (zz , yy , xx) , axis = 1 )
        self.sampling = self.ada_sampling ( self.crystal_coordinate , threshold = sampling_threshold)
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

    def ada_sampling ( self,crystal_coordinate , threshold = 10000 ) :

        num = len( crystal_coordinate )
        sampling = 1
        result = num
        while result > threshold :
            sampling = sampling * 2
            result = num / sampling

        return sampling

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

    def cal_rate(self,numbers,coefficients , pixel_size):
        """
        the calculation normally minus 0.5 for regularization and to represent the ray starting
        from the centre of the voxel
        :param coefficients:
        :param pixel_size:
        :return:
        """
        mu_li, mu_lo, mu_cr,mu_bu = coefficients
        if len(numbers)==8:
            li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        else:
            li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
            li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
        abs1 = np.exp(-((mu_li * (li_l_1 - 0.5 + li_l_2) +
                     mu_lo * (lo_l_1 - 0.5 + lo_l_2) +
                     mu_cr * (cr_l_1- 0.5 + cr_l_2) +
                         mu_bu * (bu_l_1- 0.5 + bu_l_2) ) * pixel_size
                    ))

        return  abs1

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

    def cal_rate(self,numbers,coefficients , pixel_size):
        """
        the calculation normally minus 0.5 for regularization and to represent the ray starting
        from the centre of the voxel
        :param coefficients:
        :param pixel_size:
        :return:
        """
        mu_li, mu_lo, mu_cr,mu_bu = coefficients
        if len(numbers)==8:
            li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        else:
            li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
            li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
        abs1 = np.exp(-((mu_li * (li_l_1 - 0.5 + li_l_2) +
                     mu_lo * (lo_l_1 - 0.5 + lo_l_2) +
                     mu_cr * (cr_l_1- 0.5 + cr_l_2) +
                         mu_bu * (bu_l_1- 0.5 + bu_l_2) ) * pixel_size
                    ))

        return  abs1

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