import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
import json
from skimage.exposure import histogram
from skimage.filters import *
import skimage
import os
from scipy.ndimage import shift
from scipy.stats import kurtosis , skew , norm , sem
import skimage.io as io
from scipy.signal import find_peaks
import re
from skimage.metrics import *
import time
import logging

rate_list = {'li' : 1 , 'lo' : 2 , 'cr' : 3 , 'bu' : 4}

np.seterr( divide = 'ignore' , invalid = 'ignore' )



class AbsorptionCoefficient( object ) :
    def __init__ ( self , tomo_img_path , ModelFilename , auto_orientation , auto_viewing , logger , coe_li = 0 , coe_lo = 0 ,
                   coe_cr = 0 , coe_bu = 0 , base = 'li' , crop = None ,pad=None, thresholding = 'mean' , angle = 0 , save_dir = "./" ,
                   pixel_size = 0.3 ,
                   kernel_square = (15 , 15) , full = False , offset = 0 , v_flip = False , h_flip = False ,
                   ModelRotate = -90 , flat_fielded = None , *args ) :
        """

        :param tomo_img_path:
        :param ModelFilename:
        :param angle:
        :param save_dir:
        :param pixel_size:
        :param kernel_square:
        :param full: full image of the class instead of the image thas is not blocked by the other classes
        """

        self.base = base
        self.variable_coe = {
            'li' : coe_li ,
            'lo' :  coe_lo ,
            'cr' : coe_cr ,
            'bu' : coe_bu
        }

        self.logger = logger
        self.save_dir = save_dir
        self.thresholding = thresholding
        self.crop = crop
        self.pad=pad
        self.full = full
        self.angle = angle
        self.ModelFilename = ModelFilename
        self.tomo_img_path = tomo_img_path
        self.kernel_square = kernel_square
        self.kernel = np.ones( self.kernel_square , np.uint8 )
        self.offset = offset
        self.ModelRotate = ModelRotate
        # the flipping of flat-field-corrected image
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.auto_orientation = auto_orientation
        self.auto_viewing = auto_viewing
        # current the first image is where the gonionmeter is 0
        if self.auto_orientation or self.auto_viewing is True :
            self.cal_orientation_auto( )

        # if  self.auto_viewing is True:
        #     self.cal_viewing_auto()

        self.differet_orientation( self.angle , flat_fielded = flat_fielded )
        self.pixel_size = pixel_size
        self.upper_lim_li = 0.1
        self.lower_lim_li = 0
        self.upper_lim_lo = 0.1
        self.lower_lim_lo = 0
        self.upper_lim_cr = 0.1
        self.lower_lim_cr = 0
        self.upper_lim_bu = 0.1
        self.lower_lim_bu = 0
        # import plotly.graph_objects as go
        # Z , Y , X = np.arange(self.img_list.shape[0]),np.arange(self.img_list.shape[1]),np.arange(self.img_list.shape[2])
        # values = self.img_list
        # pdb.set_trace()
        # fig = go.Figure( data = go.Volume(
        #     x = X.flatten( ) ,
        #     y = Y.flatten( ) ,
        #     z = Z.flatten( ) ,
        #     value = values.flatten( ) ,
        #     isomin = 0.1 ,
        #     isomax = 0.8 ,
        #     opacity = 0.1 ,  # needs to be small to see through all surfaces
        #     surface_count = 17 ,  # needs to be a large number for good volume rendering
        # ) )
        # fig.show( )

    def pre_process ( self ) :
        if os.path.exists( self.save_dir ) is False :

            try :
                os.makedirs( self.save_dir )
            except :
                os.mkdir( self.save_dir )
        new = self.img_list.mean( axis = 1 )
        # img2 and mask2 are the images from 3D model
        self.img2 = 255 - \
                    cv2.normalize( new , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        self.mask2 = self.mask_generation( self.img2 , thresh = 255 )

        # img is the raw tomo image to extract intensities
        # img1 and mask1 are the raw tomo images after thresholding

        self.img1 = cv2.normalize(
            self.img , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        # plot_different_thresholding(img1)
        # imagemask_overlapping(img1,mask2,title='Null')

        # try_all_threshold(self.img1)
        # plt.show()

        if self.thresholding == 'triangle' :
            thresh = threshold_triangle( self.img1 )
        elif self.thresholding == 'li' :
            thresh = threshold_li( self.img1 )
        elif self.thresholding == 'mean' :
            thresh = threshold_mean( self.img1 )
        elif self.thresholding == 'minimum' :
            thresh = threshold_minimum( self.img1 )
        elif self.thresholding == 'otsu' :
            thresh = threshold_otsu( self.img1 )
        elif self.thresholding == 'yen' :
            thresh = threshold_yen( self.img1 )
        elif self.thresholding == 'isodata' :
            thresh = threshold_isodata( self.img1 )

        self.mask1 = self.mask_generation( self.img1 , thresh = thresh )
        """ image processing: extract the overall boundary of the raw image and the projection of the 3D model
        """

        # mask1 = fill_the_labels( mask1 )
        self.mask2 , self.yxshift = self.skimage_translation_matching( self.mask1 ,
                                                                       self.mask2 )  # move mask2 to get mask1

        # in the projection of 3D tomography, the image axes are z of 3D (y in image) and x of 3D (x in image)
        # imagemask_overlapping( img1 , mask2 )
        # these absorption coefficients may be iteratively updated
        # self.coe_lo = 0
        # self.coe_bu = 0
        # self.coe_cr = 0
        # self.coe_base = 0

        # the final absorption coefficients after converge
        self.coe_lo_final = 0
        self.coe_bu_final = 0
        self.coe_cr_final = 0
        self.coe_li_final = 0

        self.lo_multi_flag = 0
        self.li_multi_flag = 0
        self.cr_multi_flag = 0
        self.bu_multi_flag = 0
        """ extract region of interest from the 3D model
        """

        self.li_region = self.region_of_interest_only( target = 1 )
        self.lo_region = self.region_interaction_of_two_classes( target = 2 )
        self.cr_region = self.region_interaction_of_two_classes( target = 3 )
        self.bu_region = self.region_interaction_of_two_classes( target = 4 )


        self.imagemask_overlapping( self.img1 , np.roll( self.li_region , self.yxshift ) ,
                                    title = '{}_region_of_interest_overall'.format( 'li' ) )
        self.imagemask_overlapping( self.img1 , np.roll( self.lo_region , self.yxshift ) ,
                                    title = '{}_region_of_interest_overall'.format( 'lo' ) )
        self.imagemask_overlapping( self.img1 , np.roll( self.cr_region , self.yxshift ) ,
                                    title = '{}_region_of_interest_overall'.format( 'cr' ) )
        self.imagemask_overlapping( self.img1 , np.roll( self.bu_region , self.yxshift ) ,
                                    title = '{}_region_of_interest_overall'.format( 'bu' ) )

        if len( np.unique( self.lo_region ) ) == 1 or self.full is True :
            self.lo_region = self.region_of_interest( target = 2 )
            self.lo_multi_flag = 1
            """then i can set a flag to use multi calculate, set a class for loop 
            """
        if len( np.unique( self.cr_region ) ) == 1 or self.full is True :
            self.cr_region = self.region_of_interest( target = 3 )
            self.cr_multi_flag = 1
        if len( np.unique( self.bu_region ) ) == 1 or self.full is True :
            self.bu_region = self.region_of_interest( target = 4 )
            self.bu_multi_flag = 1

        self.bu_region_back = self.bu_region.copy( )
        self.li_region_back = self.li_region.copy( )
        self.cr_region_back = self.cr_region.copy( )
        self.lo_region_back = self.lo_region.copy( )

    def determine_single_class ( self , cls , percent , y_list , x_list , region_back , target , single = True ,
                                 determine_peaks = False , lower_peak = True ) :

        if single :
            roi_cls , output_y_list , output_x_list = self.calculate_area_coe_single( y_list ,
                                                                                      x_list , cls = cls ,
                                                                                      percentile = percent ,
                                                                                      ranktype = "absolute" )
        else :

            roi_cls , output_y_list , output_x_list = self.calculate_area_coe_multi( y_list ,
                                                                                     x_list , percentile = percent ,
                                                                                     cls = cls ,
                                                                                     ranktype = "proportion" )
        # coe_cls= roi_cls.mean( )
        if percent == 1 :
            if cls == 'li' :
                self.upper_lim_li = np.max( roi_cls ) * 1.2
                self.lower_lim_li = np.min( roi_cls ) * 0.8
            elif cls == 'lo' :
                self.upper_lim_lo = np.max( roi_cls ) * 1.2
                self.lower_lim_lo = np.min( roi_cls ) * 0.8
            elif cls == 'cr' :
                self.upper_lim_cr = np.max( roi_cls ) * 1.2
                self.lower_lim_cr = np.min( roi_cls ) * 0.8
            elif cls == 'bu' :
                self.upper_lim_bu = np.max( roi_cls ) * 1.2
                self.lower_lim_bu = np.min( roi_cls ) * 0.8

        if determine_peaks :

            bins , edges , patches = self.histogram_in_area(
                roi_cls , percent , cls = cls , number_bins = 50 , xlim = None ,
                determine_peaks = determine_peaks )  # [0.01,0.03] )
            try :
                peaks = find_peaks( bins , prominence = 100 )
                peak_1 = edges[peaks[0][0]]
                peak_2 = edges[peaks[0][1]]
                peak_mean = (peak_1 + peak_2) / 2
                # output_y_list=output_y_list[np.where(roi_cls>peak_mean)[0]]
                # output_x_list = output_x_list[np.where( roi_cls > peak_mean )[0]]
                # roi_cls=roi_cls[np.where(roi_cls>peak_mean)[0]]

                if lower_peak is True :
                    output_y_list = output_y_list[np.where(
                        roi_cls < peak_mean )[0]]
                    output_x_list = output_x_list[np.where(
                        roi_cls < peak_mean )[0]]
                    roi_cls = roi_cls[np.where( roi_cls < peak_mean )[0]]
                else :
                    output_y_list = output_y_list[np.where(
                        roi_cls > peak_mean )[0]]
                    output_x_list = output_x_list[np.where(
                        roi_cls > peak_mean )[0]]
                    roi_cls = roi_cls[np.where( roi_cls > peak_mean )[0]]
            except :
                pass
        else :
            self.histogram_in_area(
                roi_cls , percent , cls = cls , number_bins = 50 , xlim = None )
            #

        self.logger.info( 'mean of {} is {}'.format( cls , roi_cls.mean( ) ) )
        self.logger.info(
            'standard deviation of {} is {}'.format( cls , np.std( roi_cls ) ) )
        print( 'mean of {} is {}'.format( cls , roi_cls.mean( ) ) )
        print( 'standard deviation of {} is {}'.format( cls , np.std( roi_cls ) ) )
        crd = np.stack( (output_y_list , output_x_list) , axis = 1 )

        # region_back[region_back == target] = 0
        region_background = np.zeros( region_back.shape )
        for c in crd :
            region_background[tuple( c )] = 1
        # region_back[region_back == 10] = 1
        plt.clf( )
        # if cls=='lo':
        #     pdb.set_trace()
        # if percent==0.5 and cls=='lo':
        #     pdb.set_trace()
        self.imagemask_overlapping( self.img1 , np.roll( region_background , self.yxshift ) ,
                                    title = 'area_of_{}_with_acceptance_percentage_of_{}'.format( cls , percent ) )

        # plt.imshow( region + region_back )
        # plt.title( ' area of {} with acceptance percentage of {} '.format( cls , percent ) )
        return roi_cls

    def imagemask_overlapping ( self , img1 , mask , title = 'Null' ) :
        try :
            label = np.unique( mask )[1]
        except :
            label = 10
        img1 = skimage.color.gray2rgb( img1 )

        maskrgb = skimage.color.gray2rgb( mask )
        maskrgb[mask == label] = np.array( [255 , 255 , 0] )
        # overaly = np.ubyte( img1 + 0.1 * maskrgb )
        overaly = img1 + 0.3 * maskrgb
        overaly[overaly > 255] = 255
        overaly = np.ubyte( overaly )
        plt.imshow( overaly )
        plt.title( title )
        plt.savefig( '{}/{}.png'.format( self.save_dir , title ) )
        plt.clf( )
        # plt.show( )

    def imagemask_overlapping_tri ( self , img1 , mask , mask2 , title = 'Null' ) :
        try :
            label = np.unique( mask )[1]
        except :
            label = 10
        try :
            label2 = np.unique( mask )[1]
        except :
            label2 = 20
        img1 = skimage.color.gray2rgb( img1 )

        maskrgb = skimage.color.gray2rgb( mask )
        maskrgb[mask == label] = np.array( [255 , 255 , 0] )

        maskrgb2 = skimage.color.gray2rgb( mask2 )
        maskrgb2[mask2 == label2] = np.array( [0 , 255 , 255] )
        # overaly = np.ubyte( img1 + 0.1 * maskrgb )
        overaly = img1 + 0.3 * maskrgb
        overaly[overaly > 255] = 255

        overaly = overaly + 0.3 * maskrgb2
        overaly[overaly > 255] = 255
        overaly = np.ubyte( overaly )
        plt.imshow( overaly )
        plt.title( title )
        plt.savefig( '{}/{}.png'.format( self.save_dir , title.split( '\n' )[0] ) )
        plt.clf( )

    def region_interaction_of_two_classes ( self , target , base = 1 ) :
        """
        the interactive region between the base class ( the most area ,usually the liquor )
         and the target class
        :param img_list:
        :param target: {'va':0,'li':1,'lo':2,'cr':3,'bu':4}
        :return:
        """
        inter = self.img_list.copy( )
        inter[inter == base] = 0
        number = np.count_nonzero( inter , axis = 1 )

        region = inter.sum( axis = 1 )
        final_map = region / number

        final_map[np.isnan( final_map )] = 0
        final_map[final_map != target] = 0

        return final_map

    def region_of_interest ( self , target ) :
        """
        find the region of interest where it overlaps with other area
        :param img_list:
        :param target: {'va':0,'li':1,'lo':2,'cr':3,'bu':4}
        :return:
        """
        inter = self.img_list.copy( )
        inter[inter != target] = 0
        number = np.count_nonzero( inter , axis = 1 )

        region = inter.sum( axis = 1 )
        final_map = region / number

        final_map[np.isnan( final_map )] = 0
        final_map[final_map != target] = 0

        return final_map

    def region_of_interest_only ( self , target = 1 ) :
        """
        the region where the xray only passes through it, mostly is liquor ,or could be crystal or loop
        :param liquor:
        :return:
        """
        new = self.img_list.sum( axis = 1 )
        number = np.count_nonzero( self.img_list , axis = 1 )
        final_map = new / number
        final_map[np.isnan( final_map )] = 0

        final_map[final_map != target] = 0
        return final_map

    def region_of_interest_overlap ( self , target ) :
        """
        same function with region of interest but slower speed
        :param img_list:
        :param target: {'va':0,'li':1,'lo':2,'cr':3,'bu':4}
        :return:
        """
        zz , yy , xx = self.img_list.shape
        target_position = np.where( self.img_list == target )
        abc = np.vstack( (target_position[0] , target_position[2]) ).T
        unq , count = np.unique( abc , axis = 0 , return_counts = True )
        final_map = np.zeros( (zz , xx) )
        area = [tuple( i ) for i in unq]
        for i in area :
            final_map[i] = target
        return final_map

    def erosion ( self , region , kernel , iterations = 3 , vert_lim = 0.1 , hori_lim = 0.1 ) :
        cls = np.unique( region )[1]
        y_max , x_max = region.shape

        vert_lim_top = int( y_max * vert_lim )
        vert_lim_bot = int( y_max * (1 - vert_lim) )
        hori_lim_left = int( x_max * hori_lim )
        hori_lim_right = int( x_max * (1 - hori_lim) )

        region[region > 0] = 255
        img_erosion = cv2.erode( region , kernel , iterations = iterations )
        img_erosion[img_erosion == 255] = cls
        # only focus on the central area of interest
        img_erosion[:vert_lim_top , :] = 0
        img_erosion[vert_lim_bot : , :] = 0
        img_erosion[: , :hori_lim_left] = 0
        img_erosion[: , hori_lim_right :] = 0

        y_list , x_list = np.nonzero( img_erosion )
        return y_list , x_list , img_erosion

    @staticmethod
    def rotate_image ( image , angle ) :
        image_center = tuple( np.array( image.shape[1 : :-1] ) / 2 )
        rot_mat = cv2.getRotationMatrix2D( image_center , angle , 1.0 )

        result = cv2.warpAffine(
            image , rot_mat , (image.shape[1] , image.shape[0]) , flags = cv2.INTER_NEAREST )
        return result

    def different_orientation_v1 ( self , angle , filename ) :
        self.img_list = np.load( self.ModelFilename )
        new = np.zeros( self.img_list.shape )
        if angle != 0 :
            for i , slice in enumerate( self.img_list ) :
                result = self.rotate_image( slice , angle )
                new[i] = result

        del self.img_list
        self.img_list = new

        file = os.path.join( self.tomo_img_path ,
                             "p_{}.tif".format( str( filename ).zfill( 5 ) ) )

        self.img = cv2.imread( file , 2 )

    def thresholding_method ( self ) :
        if self.thresholding == 'triangle' :
            thresh_method = threshold_triangle
        elif self.thresholding == 'li' :
            thresh_method = threshold_li
        elif self.thresholding == 'mean' :
            thresh_method = threshold_mean
        elif self.thresholding == 'minimum' :
            thresh_method = threshold_minimum
        elif self.thresholding == 'otsu' :
            thresh_method = threshold_otsu
        elif self.thresholding == 'yen' :
            thresh_method = threshold_yen
        elif self.thresholding == 'isodata' :
            thresh_method = threshold_isodata

        return thresh_method

    def cal_orientation_auto ( self ) :
        def extract_number ( s ) :
            # Helper function to extract the number from a string using regular expressions
            match = re.findall( r'\d+' , s )[-1]

            if match :
                return match
            else :
                return None

        # thresh_method = self.thresholding_method( )
        # prefix = re.findall( r'\d+' , os.listdir( self.tomo_img_path )[0] )[-1]
        #
        # prefix = os.listdir( self.tomo_img_path )[0].replace( prefix , "candidate" )
        # afterfix = len( os.listdir( self.tomo_img_path ) ) // 180
        imgfile_list = os.listdir( self.tomo_img_path )
        sorted_imgfile_list = sorted( imgfile_list , key = extract_number )
        self.img_list = np.load( self.ModelFilename )
        new1 = self.img_list.mean( axis = 1 )
        img_label = 255 - \
                    cv2.normalize( new1 , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        mask_label = self.mask_generation( img_label , thresh = 255 )

        def calculate_difference ( start , end , sorted_imgfile_list , step , mask_label , plot = False ) :
            increment = len( sorted_imgfile_list ) / 180
            difference = []
            contents = []
            proportion_cut = 0.1
            y_max_mask_label = mask_label.shape[0]
            mask_label[:int( y_max_mask_label * proportion_cut ) , :] = 0
            mask_label[int( y_max_mask_label * (1 - proportion_cut) ) : , :] = 0

            # for i,angle in enumerate(np.linspace( start , end , num = num , dtype = int )) :
            for i , angle in enumerate( np.arange( start , end , step = step , dtype = int ) ) :
                # fileindex = str( int( angle * afterfix ) ).zfill( 5 )
                # filename = prefix.replace( 'candidate' , fileindex )

                try :
                    if angle == 180 :
                        filename = sorted_imgfile_list[int(
                            increment * angle ) - 1]
                    else :
                        filename = sorted_imgfile_list[int( increment * angle )]
                except :
                    pdb.set_trace( )
                # print( filename )

                file = os.path.join( self.tomo_img_path , filename )

                candidate_img = cv2.normalize( cv2.imread( file , 2 ) , None , 0 , 255 , cv2.NORM_MINMAX ).astype(
                    'uint8' )
                if self.v_flip :
                    candidate_img = cv2.flip( candidate_img , 0 )
                thresh = self.thresholding_method( )( candidate_img )
                candidate_mask = self.mask_generation(
                    candidate_img , thresh = thresh )
                candidate_mask , mask_label = self.padding(
                    candidate_mask , mask_label )
                contents.append( len( candidate_mask[candidate_mask > 0] ) )
                candidate_mask[:int( y_max_mask_label * proportion_cut ) , :] = 0
                candidate_mask[int( y_max_mask_label *
                                    (1 - proportion_cut) ) : , :] = 0

                # shift the mask_label to match candidate_mask to plot it
                shifted_mask , xyshift = self.skimage_translation_matching(
                    candidate_mask , mask_label )
                # if plot:
                #     self.imagemask_overlapping_tri( candidate_img ,candidate_mask, shifted_mask,"threshold of angle of {} degree".format(angle))
                difference.append( np.abs( candidate_mask - shifted_mask ).mean( ) )

            return difference , contents

        angle_start = 0
        angle_end = 190
        # number = int( (angle_end - angle_start) / 10 + 1 )
        step_1 = 10
        difference_1 , contents_1 = np.array(
            calculate_difference( sorted_imgfile_list = sorted_imgfile_list , start = angle_start , end = angle_end ,
                                  step = step_1 , mask_label = mask_label ) )
        ori_peak = np.where( difference_1 == np.min( difference_1 ) )[0][0]

        if self.auto_orientation :

            if ori_peak != 0 and ori_peak != 18 :
                angle_start = (ori_peak - 1) * 10
                angle_end = (ori_peak + 1) * 10
            elif ori_peak == 0 :
                angle_start = ori_peak * 10
                angle_end = (ori_peak + 1) * 10
            elif ori_peak == 18 :
                angle_start = (ori_peak - 1) * 10
                angle_end = ori_peak * 10
            else :
                RuntimeError( "inappropriate peak_1" )
            # print("the zone is at around {}".format(peak*10))
            step_2 = 1
            # print("the zone is at around {}".format(peak*10))

            difference_2 , contents_2 = np.array(
                calculate_difference( sorted_imgfile_list = sorted_imgfile_list , start = angle_start ,
                                      end = angle_end , step = step_2 , mask_label = mask_label , plot = False ) )
            ori_peak2 = np.where( difference_2 == np.min( difference_2 ) )[0][0]
            self.offset = -(angle_start + ori_peak2)
            self.logger.info( "the zone is at  {} which is the offset".format(
                angle_start + ori_peak2 ) )
            print( "the zone is at  {} which is the offset".format(
                angle_start + ori_peak2 ) )

        if self.auto_viewing :
            view_peak = np.where( contents_1 == np.max( contents_1 ) )[0][0]

            if view_peak != 0 and view_peak != 18 :
                angle_start = (view_peak - 1) * 10
                angle_end = (view_peak + 1) * 10
            elif view_peak == 0 :
                angle_start = view_peak * 10
                angle_end = (view_peak + 1) * 10
            elif view_peak == 18 :
                angle_start = (view_peak - 1) * 10
                angle_end = view_peak * 10
            else :
                RuntimeError( "inappropriate peak_1" )
            # print("the zone is at around {}".format(peak*10))
            step_2 = 1

            # print("the biggest region is at around {}".format(peak*10))
            difference_3 , contents_3 = np.array(
                calculate_difference( sorted_imgfile_list = sorted_imgfile_list , start = angle_start ,
                                      end = angle_end , step = step_2 , mask_label = mask_label , plot = False ) )
            view_peak_2 = np.where( contents_3 == np.max( contents_3 ) )[0][0]
            self.logger.info( "The estimated angle where"
                              " the sample perfectly perpendicular to the screen is {} degree".format(
                angle_start + view_peak_2 ) )
            print( "The estimated angle where"
                   " the sample perfectly perpendicular to the screen is {} degree".format(
                angle_start + view_peak_2 ) )
            self.angle = (angle_start + view_peak_2)

    def differet_orientation ( self , angle , flat_fielded = None ) :

        angle_inv = -(angle + self.offset)
        self.img_list = np.load( self.ModelFilename )
        # if self.ModelRotate< 0:
        #     self.ModelRotate = 360+self.ModelRotate
        # counter = self.ModelRotate //90

        # self.img_list = np.rot90(self.img_list,k=counter,axes=(1,2))  # rotate clockwisely along 0 axis, if axes(2,1), anti-clockwise

        if angle_inv != 0 :
            for i , slice in enumerate( self.img_list ) :
                result = self.rotate_image( slice , angle_inv )
                self.img_list[i] = result
        if flat_fielded is not None and flat_fielded.isspace( ) is not True \
                and flat_fielded != '' :
            file = os.path.join( self.tomo_img_path , flat_fielded )
        else :
            afterfix = len( os.listdir( self.tomo_img_path ) ) / 180

            fileindex = int( angle * afterfix )
            if fileindex == 0 :
                pass
            else:
                fileindex -= 1
            print("the fileindex is {}".format(fileindex))
            for f in os.listdir( self.tomo_img_path ) :
                index = int( re.findall( r'\d+' , f )[-1] )
                if index == fileindex :
                    filename = f
                    break
            file = os.path.join( self.tomo_img_path , filename )

        self.logger.info(
            "the examined flat-fielded corrected image is {}".format( os.path.basename( file ) ) )
        print(
            "the examined flat-fielded corrected image is {}".format( os.path.basename( file ) ) )
        new1 = self.img_list.mean( axis = 1 )
        self.img = cv2.imread( file , 2 )

        if self.v_flip :
            # 1 : flip over horizontal ; 0 : flip over vertical
            self.img = cv2.flip( self.img , 0 )
        if self.h_flip :
            # 1 : flip over horizontal ; 0 : flip over vertical
            self.img = cv2.flip( self.img , 1 )
        if self.img.shape[0] != new1.shape[0] :
            if self.img.shape[0] > new1.shape[0]:
                self.img , new1 = self.cropping( self.img , new1 , crop = self.crop )
            else:
                self.img , new1 = self.padding( self.img , new1 , pad = self.pad )
         

        candidate_img = cv2.normalize( self.img , None , 0 , 255 , cv2.NORM_MINMAX ).astype(
            'uint8' )
        thresh = self.thresholding_method( )( candidate_img )
        candidate_mask = self.mask_generation( candidate_img , thresh = thresh )
        img_label = 255 - \
                    cv2.normalize( new1 , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        mask_label = self.mask_generation( img_label , thresh = 255 )
        shifted_mask , xyshift = self.skimage_translation_matching(
            candidate_mask , mask_label )
     
        self.imagemask_overlapping( candidate_img , shifted_mask ,
                                    "Overlap_threshold_of_angle_{}_yellow_is_the_projection_of_3d_model".format( angle ) )
        self.imagemask_overlapping_tri( candidate_img , candidate_mask , shifted_mask ,
                                        "Debug_threshold_of_angle_of_{}_degree\n"
                                        "yellow is the thresholding of flat-field \n"
                                        "blue is the projection of 3D model \n"
                                        "green is where they overlap".format( angle ) )

    def cropping ( self , img , label_img , crop = None ) :

        if self.crop is not None :
            top , bot , left , right = self.crop
            img_y , img_x = img.shape
            label_img_y , label_img_x = label_img.shape
            if img_x > label_img_x :
                img = self.img[top :bot-1, left :right]
            else:
                label_img= label_img[top :bot, left :right]

        else :
            if label_img.shape != img.shape :
                img_y , img_x = img.shape
                label_img_y , label_img_x = label_img.shape
                if img_x > label_img_x :
                    diff = img_x-label_img_x
                    img = img[: , int(diff/2):int(img_x-diff/2)]
                else :
                    diff = label_img_x - img_x
                    label_img = label_img[: , int(diff/2):int(label_img_x-diff/2)]

                if img_y > label_img_y :
                    diff = img_y - label_img_y
                    img = img[int(diff/2):int(img_y-diff/2) , :]
                else :
                    diff = label_img_y - img_y
                    label_img = label_img[int(diff/2):int(label_img_y-diff/2), :]

        return img , label_img

    def padding ( self , img , label_img,pad = None) :
        # if two images have different sizes, pad the smaller one
        if pad is None :
            pad = img[0][0]
        img_y , img_x = img.shape
        label_img_y , label_img_x = label_img.shape
        
        if img_x > label_img_x :
            pad_width = img_x - label_img_x
            padding = np.ones( (label_img_y , pad_width) ) *pad
            label_img = np.concatenate( (label_img , padding) , axis = 1 )
        else :
            pad_width = label_img_x - img_x
            padding =  np.ones( (img_y , pad_width) ) *pad
            img = np.concatenate( (img , padding) , axis = 1 )

        if img_y > label_img_y :
            pad_height = img_y - label_img_y
            padding =  np.ones( (pad_height , img_x) ) *pad
            label_img = np.concatenate( (label_img , padding) , axis = 0 )
        else :
            pad_height = label_img_y - img_y
            padding =  np.ones( (pad_height , label_img_x) ) *pad
            img = np.concatenate( (img , padding) , axis = 0 )

        return img , label_img

    def fill_the_labels ( self , mask , fill_value = 1 , difference_threshold = 10 ) :
        yy , xx = mask.shape

        for i in range( yy - 1 ) :
            row = mask[i , :]
            non_zero = np.nonzero( row )[0]
            if len( non_zero ) > 1 :
                valve = 'left'
                for j in range( len( non_zero ) ) :

                    if j == len( non_zero ) - 1 :
                        break
                    if non_zero[j + 1] - non_zero[j] < difference_threshold :
                        # if the gap is small, then fill the gap
                        mask[i][non_zero[j] :non_zero[j + 1]] = fill_value
        for i in range( xx - 1 ) :

            col = mask[: , i]
            non_zero = np.nonzero( col )[0]
            if len( non_zero ) > 1 :
                valve = 'left'
                for j in range( len( non_zero ) ) :

                    if j == len( non_zero ) - 1 :
                        break
                    if non_zero[j + 1] - non_zero[j] < difference_threshold :
                        # if the gap is small, then fill the gap
                        # for k in np.arange(non_zero[j],non_zero[j + 1],step=1,dtype=int):
                        #     mask[k][i] = fill_value
                        mask[non_zero[j] :non_zero[j + 1] , i] = fill_value
                        # needs to be changed to a better way to achieve
        return mask

    def calculate_ray_lengths ( self , rayoi ) :
        classes = np.unique( rayoi )
        lengths = {'va' : 0 , 'li' : 0 , 'lo' : 0 , 'cr' : 0 , 'bu' : 0}
        for cls in classes :
            if cls == 0 :
                lengths['va'] = np.count_nonzero( rayoi == 0 )
            elif cls == 3 :
                # crystal
                lengths['cr'] = np.count_nonzero( rayoi == 3 )
            elif cls == 2 :
                # loop
                lengths['lo'] = np.count_nonzero( rayoi == 2 )
            elif cls == 1 :
                # liquor
                lengths['li'] = np.count_nonzero( rayoi == 1 )
            elif cls == 4 :
                # bubble
                lengths['bu'] = np.count_nonzero( rayoi == 4 )
            else :
                RuntimeError(
                    "unexpected pixel value: {} is found in the 3D model".format( cls ) )
        return lengths

    def rank_list_generator ( self , lengths , cls , ranktype = "proportion" , Denominator = "li" ) :
        if ranktype == "proportion" :
            targetLength = lengths[cls]
            demon = lengths[Denominator]
            if demon != 0 :
                return targetLength / demon
            else :
                return 1
        elif ranktype == "absolute" :
            return lengths[cls]

    def calculate_lin_abscoe_li ( self , lengths , intensity , cls = "li" , accept_threshold = 0 ) :
        li = lengths[cls]
        if li < accept_threshold :
            coe_li = 0
        else :
            coe_li = -np.log( intensity ) / (li * self.pixel_size)
        return coe_li

    def calculate_area_coe_single ( self , y_list , x_list , percentile = 1 , cls = 'li' , ranktype = "absolute" ) :
        # roi_x_min,roi_x_max,roi_y_min,roi_y_max):

        # roi = np.zeros( (roi_y_max - roi_y_min , roi_x_max - roi_x_min) )  # region of interest of only liquor
        # yy , xx = np.meshgrid( np.arange( roi_x_min , roi_x_max ) , np.arange( roi_y_min , roi_y_max ) )
        # roi   = np.zeros(len(y_list))
        roi = []
        output_y_list = []
        output_x_list = []
        # as we need to fix mask2 and extract intensities from mask1
        zshift , xshift = self.yxshift
        proportion_list = []
        for i , y in enumerate( y_list ) :
            x = x_list[i]
            rayoi = self.img_list[y , : , x]
            # intensity=img[y+ int( zshift )  , x+ int( xshift )]
            lengths = self.calculate_ray_lengths( rayoi )

            proportion_lengths = self.rank_list_generator(
                lengths , cls , ranktype = ranktype )

            proportion_list.append( [i , proportion_lengths , lengths] )

        def sorting ( x ) :
            return x[1]

        proportion_list.sort( key = sorting , reverse = True )

        prop_list = proportion_list[:int( len( proportion_list ) * percentile )]
        
        for j , thingoi in enumerate( prop_list ) :
            index = thingoi[0]
            lengths = thingoi[2]
            x = x_list[index]
            y = y_list[index]

            intensity = self.img[y + int( zshift ) , x + int( xshift )]
            coe = self.calculate_lin_abscoe_li( lengths , intensity , cls = cls )

            # print( '[{}]/[{}] calculating [{}][{}] absorption coefficient'.format( index , len( prop_list ) , y , x ) )
            output_x_list.append( x )
            output_y_list.append( y )
            roi.append( coe )

        return np.array( roi ) , np.array( output_y_list ) , np.array( output_x_list )

    def calculate_ray_abscoe_multi ( self , lengths , intensity , cls , ) :
        li = lengths['li']
        lo = lengths['lo']
        cr = lengths['cr']
        bu = lengths['bu']
        coe_li = self.variable_coe['li']
        coe_lo = self.variable_coe['lo']
        coe_cr = self.variable_coe['cr']
        coe_bu = self.variable_coe['bu']
        # len_target is the needed absorption coefficient
        if cls == 'cr' :
            len_target = lengths[cls]
            cr = 0
        elif cls == 'lo' :
            len_target = lengths[cls]
            lo = 0
        elif cls == 'bu' :
            len_target = lengths[cls]
            bu = 0
        elif cls == 'li' :
            len_target = lengths[cls]
            li = 0
        else :
            raise RuntimeError( 'unregonized class label' )

        coe_other = -((coe_li * li + coe_lo * lo + coe_cr * cr + coe_bu * bu) * self.pixel_size + np.log(
            intensity )) / (
                            len_target * self.pixel_size)
        # if coe_other<0:
        #     pdb.set_trace()
        return coe_other

    def calculate_area_coe_multi ( self , y_list , x_list , percentile = 1 , cls = 'li' , ranktype = "proportion" ) :
        # roi_x_min,roi_x_max,roi_y_min,roi_y_max):

        # roi = np.zeros( (roi_y_max - roi_y_min , roi_x_max - roi_x_min) )  # region of interest of only liquor
        # yy , xx = np.meshgrid( np.arange( roi_x_min , roi_x_max ) , np.arange( roi_y_min , roi_y_max ) )
        # roi   = np.zeros(len(y_list))
        roi = []
        output_y_list = []
        output_x_list = []
        # as we need to fix mask2 and extract intensities from mask1
        zshift , xshift = self.yxshift
        proportion_list = []
        for i , y in enumerate( y_list ) :
            x = x_list[i]
            rayoi = self.img_list[y , : , x]
            lengths = self.calculate_ray_lengths( rayoi )
            proportion_lengths = self.rank_list_generator(
                lengths , cls , ranktype = ranktype , Denominator = "li" )
            proportion_list.append( [i , proportion_lengths , lengths] )

        """sorting the path lengths, and calculate the top percentile % to determine absorption coefficient
        """

        def sorting ( x ) :
            return x[1]

        proportion_list.sort( key = sorting , reverse = True )
        prop_list = proportion_list[:int( len( proportion_list ) * percentile )]

        for j , thingoi in enumerate( prop_list ) :
            index = thingoi[0]
            lengths = thingoi[2]
            x = x_list[index]
            y = y_list[index]
            intensity = self.img[y + int( zshift ) , x + int( xshift )]
            coe = self.calculate_ray_abscoe_multi( lengths , intensity , cls = cls )

            # print( '[{}]/[{}] calculating [{}][{}] absorption coefficient'.format( index , len( prop_list ) , y , x ) )
            output_x_list.append( x )
            output_y_list.append( y )
            roi.append( coe )

        return np.array( roi ) , np.array( output_y_list ) , np.array( output_x_list )

    def background_generation ( self , img1 , thresh ) :
        output = np.ones( img1.shape )
        mask = img1 > thresh
        mask = output * mask
        return mask

    def mask_generation ( self , img1 , thresh ) :
        output = np.ones( img1.shape )
        mask = img1 < thresh
        mask = output * mask
        return mask

    def histogram_in_area ( self , roi , percent = 1 , cls = 'liquor' , number_bins = 100 , xlim = None ,
                            determine_peaks = False ) :
        plt.figure( figsize = (19 , 12) )
        try :
            bins , edges , patches = plt.hist( roi , number_bins , facecolor = 'g' )
        except :
            pdb.set_trace( )
        # ppf = norm.ppf( 0.975 , loc = roi.mean( ) , scale = np.std( roi ) )
        roi_std = np.std( roi )
        if cls == 'li' :
            upper_lim = self.upper_lim_li
            lower_lim = self.lower_lim_li
        elif cls == 'lo' :
            upper_lim = self.upper_lim_lo
            lower_lim = self.lower_lim_lo
        elif cls == 'cr' :
            upper_lim = self.upper_lim_cr
            lower_lim = self.lower_lim_cr
        elif cls == 'bu' :
            upper_lim = self.upper_lim_bu
            lower_lim = self.lower_lim_bu
        intervals_1sigma = [roi.mean( ) + roi_std , roi.mean( ) - roi_std]
        intervals_2sigma = [roi.mean( ) + 2 * roi_std , roi.mean( ) - 2 * roi_std]
        intervals_3sigma = [roi.mean( ) + 3 * roi_std , roi.mean( ) - 3 * roi_std]
        try :
            peaks = find_peaks( bins , prominence = 100 )
            peak_1 = edges[peaks[0][0]]
            peak_2 = edges[peaks[0][1]]
            thre = (peak_1 + peak_2) / 2
            roi_cls_low = roi[np.where( roi < thre )[0]]
            roi_cls_high = roi[np.where( roi > thre )[0]]
            plt.axvline( thre , color = 'k' , label = 'threshold between two peaks' )
            plt.axvline( roi_cls_low.mean( ) , color = 'r' , linestyle = 'dashed' ,
                         label = 'mean of the lower peak: {}'.format( np.round( roi_cls_low.mean( ) , 5 ) ) )
            plt.axvline( roi_cls_high.mean( ) , color = 'r' , linestyle = 'dashed' ,
                         label = 'mean of the higher peak: {}'.format( np.round( roi_cls_high.mean( ) , 5 ) ) )

        except :
            pass
        self.logger.info( ' skewness is {}'.format( skew( roi ) ) )
        self.logger.info( ' kurtosis is {}'.format( kurtosis( roi ) ) )
        print( ' skewness is {}'.format( skew( roi ) ) )
        print( ' kurtosis is {}'.format( kurtosis( roi ) ) )
        # https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf
        # https://stackoverflow.com/questions/57829702/how-to-i-fill-the-central-95-confidence-interval-of-a-matplotlib-histogram
        # plt.axvline( x = norm.interval( alpha = 0.8 , loc = np.mean( roi ) , scale = sem( roi ) )[0],color = 'k',linestyle='dashed' )
        # plt.axvline( x = norm.interval( alpha = 0.8 , loc = np.mean( roi ) , scale = sem( roi ) )[1],color = 'k',linestyle='dashed' )
        plt.axvline( intervals_1sigma[0] , color = 'c' ,
                     linestyle = 'dashed' , label = ' 1 sigma' )
        plt.axvline( intervals_1sigma[1] , color = 'c' ,
                     linestyle = 'dashed' , label = ' 1 sigma' , linewidth = 2 )
        plt.axvline( intervals_2sigma[0] , color = 'm' ,
                     linestyle = 'dashed' , label = ' 2 sigma' , linewidth = 2 )
        plt.axvline( intervals_2sigma[1] , color = 'm' ,
                     linestyle = 'dashed' , label = ' 2 sigma' , linewidth = 2 )
        plt.axvline( intervals_3sigma[0] , color = 'y' ,
                     linestyle = 'dashed' , label = ' 3 sigma' , linewidth = 2 )
        plt.axvline( intervals_3sigma[1] , color = 'y' ,
                     linestyle = 'dashed' , label = ' 3 sigma' , linewidth = 2 )
        plt.axvline( x = roi.mean( ) , color = 'r' ,
                     label = 'mean: {}'.format( round( roi.mean( ) , 5 ) ) )
        plt.axvline( x = np.median( roi ) , color = 'b' ,
                     label = 'median: {}'.format( round( np.median( roi ) , 5 ) ) )

        plt.xlim( (lower_lim , upper_lim) )
        plt.xlabel( 'absorption coefficients' , fontsize = 18 )
        plt.ylabel( 'Counts' , fontsize = 18 )
        plt.xticks( fontsize = 18 )
        if xlim is not None :
            plt.xlim( xlim )
        # plt.xticks(np.round(np.linspace(min(edges), max(edges), int(number_bins/2)),4), fontsize = 18)
        # plt.xticks( np.round( np.linspace( 0.015 , 0.025 , number_bins ) , 4 ) , fontsize = 18 )
        plt.yticks( fontsize = 18 )
        plt.title( ' Histogram of {} with acceptance percentage of {} \n'
                   'std is {} \n'
                   ' skewness is {} \n'
                   ' kurtosis is {} '.format( cls , percent , round( np.std( roi ) , 5 ) ,
                                              round( skew( roi ) , 5 ) , round( kurtosis( roi ) , 5 ) ) ,
                   fontsize = 18 )

        handles , labels = plt.gca( ).get_legend_handles_labels( )
        by_label = dict( zip( labels , handles ) )
        plt.legend( by_label.values( ) , by_label.keys( ) , fontsize = 18 )
        plt.savefig(
            '{}/Hist_{}_with_acceptance_percentage_of_{}.png'.format( self.save_dir , cls , percent ) )
        # plt.show()
        if determine_peaks :
            return bins , edges , patches

    def skimage_translation_matching ( self , img , mask1 ) :
        # https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html
        # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156

        from skimage.registration import phase_cross_correlation
        from skimage.registration._phase_cross_correlation import _upsampled_dft
        from scipy.ndimage import fourier_shift
        xyshift , error , diffphase = phase_cross_correlation( img , mask1 )
        mask1 = shift( mask1 , xyshift , mode = 'nearest' ).astype( 'int' )
        return mask1 , xyshift.astype( int )


class RunAbsorptionCoefficient( AbsorptionCoefficient ) :
    def __init__ ( self , tomo_img_path , ModelFilename , auto_viewing , auto_orientation , logger ,coe_li = 0 , coe_lo = 0 ,
                   coe_cr = 0 , coe_bu = 0 , base = 'li' ,
                   angle = 0 , save_dir = './' , pixel_size = 0.3 ,
                   kernel_square = (15 , 15) , full = False , offset = 0 , v_flip = False , h_flip = False ,
                   ModelRotate = -90 , crop = None , thresholding = 'triangle' , flat_fielded = None) :
        super( ).__init__( tomo_img_path , ModelFilename , angle = angle , logger = logger , coe_li = coe_li , coe_lo = coe_lo ,
                           coe_cr = coe_cr , coe_bu = coe_bu , base = base ,
                           auto_viewing = auto_viewing , auto_orientation = auto_orientation ,
                           save_dir = save_dir , pixel_size = pixel_size ,
                           kernel_square = kernel_square , full = full ,
                           offset = offset , v_flip = v_flip , h_flip = h_flip ,
                           ModelRotate = ModelRotate , crop = crop , thresholding = thresholding ,
                           flat_fielded = flat_fielded )

    def run ( self , singlecls = False ) :
        # new = self.img_list.mean( axis = 1 )
        # img2 = 255 - cv2.normalize( new , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        # mask2 = self.mask_generation( img2 , thresh = 255 )
        # img1 = cv2.normalize( self.img , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
        #
        # """ image processing: extract the overall boundary of the raw image and the projection of the 3D model
        # """
        # # plot_different_thresholding(img1)
        # # imagemask_overlapping(img1,mask2,title='Null')
        # thresh = threshold_triangle( img1 )
        # mask1 = mask_generation( img1 , thresh = thresh )
        # # mask1 = fill_the_labels( mask1 )
        # mask2 , yxshift = skimage_translation_matching( mask1 , mask2 )  # move mask2 to get mask1
        # zshift,xshift = yxshift # in the projection of 3D tomography, the image axes are z of 3D (y in image) and x of 3D (x in image)
        # #imagemask_overlapping( img1 , mask2 )

        self.pre_process( )
        """ extract inner area of the region of interest to mitiagte edge effect
        (this is can be improved by morthogonal erosion)
        """
        number_cls = np.unique( self.img_list ).shape[0]
        try :
            y_list_li , x_list_li , self.li_region_back = self.erosion(
                self.li_region_back , self.kernel )
            self.imagemask_overlapping( self.img1 , np.roll( self.li_region_back , self.yxshift ) ,
                                        title = '{}_region_of_interest_overall_after_erosion'.format( 'li' ) )
        except :
            y_list_li , x_list_li , self.li_region_back = None , None , None

        try :

            y_list_cr , x_list_cr , self.cr_region_back = self.erosion(
                self.cr_region_back , self.kernel )
            self.imagemask_overlapping( self.img1 , np.roll( self.cr_region_back , self.yxshift ) ,
                                        title = '{}_region_of_interest_overall_after_erosion'.format( 'cr' ) )
        except :
            y_list_cr , x_list_cr , self.cr_region_back = None , None , None

        try :
            y_list_lo , x_list_lo , self.lo_region_back = self.erosion(
                self.lo_region_back , self.kernel )
            self.imagemask_overlapping( self.img1 , np.roll( self.lo_region_back , self.yxshift ) ,
                                        title = '{}_region_of_interest_overall_after_erosion'.format( 'lo' ) )
        except :
            y_list_lo , x_list_lo , self.lo_region_back = None , None , None

        if number_cls == 5 :
            y_list_bu , x_list_bu , self.bu_region_back = self.erosion(
                self.bu_region_back , self.kernel )
            self.imagemask_overlapping( self.img1 , np.roll( self.bu_region_back , self.yxshift ) ,
                                        title = '{}_region_of_interest_overall_after_erosion'.format( 'bu' ) )

        # coe_li = 0.01891746
        # coe_lo = 0.0162

        percent = 1
        # 1 has to be the first to constraint the range of histogram
        order = [1 , 0.75 , 0.5 , 0.25]
        liac_mean_list = []
        loac_mean_list = []
        crac_mean_list = []
        buac_mean_list = []
        liac_median_list = []
        loac_median_list = []
        crac_median_list = []
        buac_median_list = []
        variable_dict_x = {
            'li' : x_list_li ,
            'lo' : x_list_lo ,
            'cr' : x_list_cr ,

        }

        variable_dict_y = {
            'li' : y_list_li ,
            'lo' : y_list_lo ,
            'cr' : y_list_cr ,
        }
        variable_mean_dict_list = {
            'li' : liac_mean_list ,
            'lo' : loac_mean_list ,
            'cr' : crac_mean_list ,
        }
        variable_median_dict_list = {
            'li' : liac_median_list ,
            'lo' : loac_median_list ,
            'cr' : crac_median_list ,
        }
        variable_region_back = {
            'li' : self.li_region_back ,
            'lo' : self.lo_region_back ,
            'cr' : self.cr_region_back ,
        }
        variable_flags = {
            'li' : 0 ,
            'lo' : 0 ,
            'cr' : 0 ,
        }
        # flags=0 means this class will be calculated
        # flags=1 means this class will not be calculated

        main_classes=[]
        if number_cls == 5:
            variable_dict_x['bu'] = x_list_bu
            variable_dict_y['bu'] = y_list_bu
            variable_mean_dict_list['bu'] = buac_mean_list
            variable_median_dict_list['bu'] = buac_median_list
            variable_region_back['bu'] = self.bu_region_back
            
            variable_flags['bu'] = 0


        for key in variable_flags :
            if self.variable_coe[key] !=0 :
                variable_flags[key] = 1
            else:
                main_classes.append(key)

        name_list=main_classes.copy()
        try:
            main_classes.remove( self.base )
        except:
            pass
        
        if singlecls :

            roi_othercls = self.determine_single_class( cls = other_cls , percent = percent ,
                                                        y_list = variable_dict_y[other_cls] ,
                                                        x_list = variable_dict_x[other_cls] ,
                                                        region_back = variable_region_back[other_cls] ,
                                                        target = rate_list[
                                                            other_cls] , single = False )
        else :
            for percent in order :
                """
                calculate the absorption coefficient of liquor regions and plot the historgrams and the covering area on the raw image
                """
                if variable_flags[self.base] == 0 :
                    roi_base = self.determine_single_class( cls = self.base , percent = percent ,
                                                            y_list = variable_dict_y[self.base] ,
                                                            x_list = variable_dict_x[self.base] ,
                                                            region_back = variable_region_back[self.base] ,
                                                            target = rate_list[
                                                                self.base] , single = True ,
                                                            determine_peaks = False , lower_peak = False )

                    # self.coe_li = roi_base.mean()
                    self.variable_coe[self.base] = roi_base.mean( )
                    variable_mean_dict_list[self.base].append( roi_base.mean( ) )
                    variable_median_dict_list[self.base].append( np.median( roi_base ) )
                else:
                    variable_mean_dict_list[self.base].append( self.variable_coe[self.base]  )
                    variable_median_dict_list[self.base].append( self.variable_coe[self.base]  )
                for other_cls in main_classes :
                    # try:
                    if variable_flags[other_cls] == 0 :
                        if variable_dict_x[other_cls] is not None :

                            roi_othercls = self.determine_single_class( cls = other_cls , percent = percent ,
                                                                        y_list = variable_dict_y[other_cls] ,
                                                                        x_list = variable_dict_x[other_cls] ,
                                                                        region_back = variable_region_back[other_cls] ,
                                                                        target = rate_list[
                                                                            other_cls] , single = False )
                            # coe_othercls = roi_othercls.mean()
                            self.variable_coe[other_cls] = roi_othercls.mean( )
                            variable_mean_dict_list[other_cls].append( roi_othercls.mean( ) )
                            variable_median_dict_list[other_cls].append( np.median( roi_othercls ) )
                            # pdb.set_trace()
                        else :
                            self.variable_coe[other_cls] = 0
                            variable_mean_dict_list[other_cls].append(0 )
                            variable_median_dict_list[other_cls].append( 0 )
                    else:
                        variable_mean_dict_list[other_cls].append( self.variable_coe[other_cls] )
                        variable_median_dict_list[other_cls].append( self.variable_coe[other_cls] )
                        # except:
                        #     coe_othercls

                """set the classes except liquor and crystal to be 0 to determine the area
                (note: but now the loop absorption coefficients are assumed to be measured by Ramona)"""

                # if number_cls == 5 :
                #     roi_bu = self.determine_single_class( cls = 'bu' , percent = percent , y_list = y_list_bu ,
                #                                           x_list = x_list_bu ,
                #                                           region_back = self.bu_region_back , target = 4 ,
                #                                           single = False )

                #     self.variable_coe['bu'] = roi_bu.mean( )
                #     variable_dict_list['bu'].append( self.variable_coe['bu'] )

            # for percent in order :
            #     """
            #      calculate the absorption coefficient of liquor regions and plot the historgrams and the covering area on the raw image
            #     """

            #     mean_std = []
            #     roi_li = self.determine_single_class( cls = base , percent = percent , y_list = variable_dict_y[base] ,
            #                                           x_list = variable_dict_x[base] ,
            #                                           region_back = self.li_region_back , target = rate_list[base] , single = True ,
            #                                           determine_peaks = True , lower_peak = False )

            #     self.coe_li = roi_li.mean( )
            #     for other_cls in classes:
            #     """determine loop absorption coefficient """
            #     roi_lo = self.determine_single_class( cls = 'lo' , percent = percent , y_list = variable_dict_y[other_cls] ,
            #                                           x_list = variable_dict_x[other_cls] ,
            #                                           region_back = self.lo_region_back , target = 2 , single = False )
            #     self.coe_lo = roi_lo.mean( )

            #     """set the classes except liquor and crystal to be 0 to determine the area
            #     (note: but now the loop absorption coefficients are assumed to be measured by Ramona)"""

            #     roi_cr = self.determine_single_class( cls = 'cr' , percent = percent , y_list = y_list_cr ,
            #                                           x_list = x_list_cr ,
            #                                           region_back = self.cr_region_back , target = 3 , single = False )
            #     self.coe_cr = roi_cr.mean( )

            #     """determine bubble absorption coefficient """
            #     """set the classes except liquor and crystal to be 0 to determine the area"""
            #     if number_cls >3:

            #             roi_bu = self.determine_single_class( cls = 'bu' , percent = percent , y_list = y_list_bu ,
            #                                                 x_list = x_list_bu ,
            #                                                 region_back = self.bu_region_back , target = 4 , single = False )
            #             self.coe_bu = roi_bu.mean( )
            #             buac_list.append( self.coe_bu)

            #     liac_list.append( self.coe_li)
            #     loac_list.append( self.coe_lo)
            #     crac_list.append( self.coe_cr)

            # print("the offset is {}".format(self.offset))
            # print( "the rotation angle is {}".format( self.angle) )
            self.logger.info( "The starting omega angle of "
                              "tomography experiment is chosen as {} degree".format( self.offset ) )

            self.logger.info( "The angle where"
                              " the sample perfectly perpendicular to the screen is chosen as {} degree".format(
                self.angle ) )
            print( "The starting omega angle of "
                   "tomography experiment is chosen as {} degree".format( self.offset ) )

            print( "The angle where"
                   " the sample perfectly perpendicular to the screen is chosen as {} degree".format( self.angle ) )

            output = [order]
            output.append(name_list )
            output.append( liac_mean_list )
            output.append( loac_mean_list )
            output.append(  crac_mean_list )
            if number_cls == 5 :
                output.append( buac_mean_list )
            # pdb.set_trace()
            with open( os.path.join( self.save_dir , "mean_coefficients_with_percentage.json" ) , 'w' ) as f1 :
                json.dump( output , f1 , indent = 2 )

            output = [order]
            output.append(name_list )
            output.append( liac_median_list )
            output.append( loac_median_list)
            output.append( crac_median_list)
            if number_cls == 5 :
                output.append( buac_median_list )
            # pdb.set_trace()
            with open( os.path.join( self.save_dir , "median_coefficients_with_percentage.json" ) , 'w' ) as f1 :
                json.dump( output , f1 , indent = 2 )

def tablization ( dataset , centre = 0 , save_dir = './' ) :
    import pandas as pd

    def cal_percentage ( df , centre ) :
        def per_list ( column ) :
            percentage_list = []

            for num in column :
                percentage = (column[centre] - num) / column[centre] * 100
                percentage_list.append( percentage )
            return percentage_list

        column = per_list( df['liquor'] )
        df['liquor_percentage'] = column
        column = per_list( df['loop'] )
        df['loop_percentage'] = column
        column = per_list( df['crystal'] )
        df['crystal_percentage'] = column

        try :
            column = per_list( df['bubble'] )
            df['bubble_percentage'] = column
        except :
            pass

        return df

    dataset = str( dataset )
    first_one = True
    for root , folders , files in os.walk( save_dir ) :
        for folder in folders :
            if dataset in folder :
                angle = re.findall( r'\d+' , folder )[-1]
                angle = int( angle )
                for file in os.listdir( os.path.join( root , folder ) ) :
                    if 'json' in file :
                        with open( os.path.join( save_dir , folder , file ) , 'r' ) as f1 :
                            data = json.load( f1 )
                        list_nanme = [str( i ) for i in data[0]]
                        df = pd.DataFrame( data[1 :] , columns = list_nanme )
                        if first_one :
                            df_100 = pd.DataFrame( columns = [angle] )
                            df_75 = pd.DataFrame( columns = [angle] )
                            df_50 = pd.DataFrame( columns = [angle] )
                            df_25 = pd.DataFrame( columns = [angle] )
                            first_one = False
                        else :
                            pass
                        df_100[angle] = df['1']
                        df_75[angle] = df['0.75']
                        df_50[angle] = df['0.5']
                        df_25[angle] = df['0.25']

            else :
                continue

    if np.isnan( df_25.tail( 1 ).values[0][0] ) :
        df_25.drop( index = df_25.index[-1] , axis = 0 , inplace = True )
        df_75.drop( index = df_75.index[-1] , axis = 0 , inplace = True )
        df_50.drop( index = df_50.index[-1] , axis = 0 , inplace = True )
        df_100.drop( index = df_100.index[-1] , axis = 0 , inplace = True )
        df_25.index = ["liquor" , "loop" , "crystal"]
        df_50.index = ["liquor" , "loop" , "crystal"]
        df_75.index = ["liquor" , "loop" , "crystal"]
        df_100.index = ["liquor" , "loop" , "crystal"]

    else :
        df_25.index = ["liquor" , "loop" , "crystal" , "bubble"]
        df_50.index = ["liquor" , "loop" , "crystal" , "bubble"]
        df_75.index = ["liquor" , "loop" , "crystal" , "bubble"]
        df_100.index = ["liquor" , "loop" , "crystal" , "bubble"]

    df_25 = df_25.reindex( sorted( df_25.columns ) , axis = 1 ).transpose( )
    df_50 = df_50.reindex( sorted( df_50.columns ) , axis = 1 ).transpose( )
    df_75 = df_75.reindex( sorted( df_75.columns ) , axis = 1 ).transpose( )
    df_100 = df_100.reindex( sorted( df_100.columns ) , axis = 1 ).transpose( )

    df_25 = cal_percentage( df_25 , centre )
    df_50 = cal_percentage( df_50 , centre )
    df_75 = cal_percentage( df_75 , centre )
    df_100 = cal_percentage( df_100 , centre )

    df_25.to_csv( os.path.join(
        save_dir , '{}_25%_coefficients.csv'.format( dataset ) ) )
    df_50.to_csv( os.path.join(
        save_dir , '{}_50%_coefficients.csv'.format( dataset ) ) )
    df_75.to_csv( os.path.join(
        save_dir , '{}_75%_coefficients.csv'.format( dataset ) ) )
    df_100.to_csv( os.path.join(
        save_dir , '{}_100%_coefficients.csv'.format( dataset ) ) )


if __name__ == '__main__' :
    path = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16846segmentation_labels'
    filename = '{}_tomobar_cropped.npy'.format( '16846' )

    save_npy( path , filename = filename , label = True , reverse = False )
    img_list = np.load( filename )
    pdb.set_trace( )
