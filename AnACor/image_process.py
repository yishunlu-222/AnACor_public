import skimage.io as io
from matplotlib import pyplot as plt
import os
import pdb
import cv2
import numpy as np
import skimage.transform as trans
from tqdm import tqdm
import re
from ast import literal_eval
import re

class ImagePreprocess(object):
    def __init__(self,root,prefix,save_dir):
        self.root=root
        self.prefix=prefix
        self.save_dir=save_dir
        # red
        lower_red = np.array( [0 , 43 , 46] )
        upper_red = np.array( [10 , 255 , 255] )
        # green
        lower_green = np.array( [35 , 43 , 46] )
        upper_green = np.array( [77 , 255 , 255] )
        # blue
        lower_blue = np.array( [100 , 43 , 46] )
        upper_blue = np.array( [124 , 255 , 255] )
        # cyan
        lower_cyan = np.array( [78 , 43 , 46] )
        upper_cyan = np.array( [99 , 255 , 255] )
        # liquor, the background hsv is the lower
        lower_liquor = np.array( [89 , 30 , 203] )
        upper_liquor = np.array( [91 , 32 , 205] )

        lower_yellow = np.array( [26 , 43 , 46] )
        upper_yellow = np.array( [34 , 255 , 255] )
        self.col_range=[[lower_red,upper_red,],[lower_liquor,upper_liquor],
                        [lower_green,upper_green],[lower_yellow,upper_yellow]]
    def change_names ( self,root , prefix = '13295_tomobar' ) :
        for img_list in os.listdir( root ) :
            if 'tif' in img_list :
                old = os.path.join( root , img_list )
                new = os.path.join( root , '{}_{}.tiff'.format( prefix , int(
                    re.findall( r'_\d+' , os.path.basename( img_list ) )[-1][1 :] ) ) )

                os.rename( old , new )
                print( new )

    def mask_genertator(self,o_root,m_root,col_range,
                        names=['crystal_mask','liquor_mask','loop_mask','bu_mask']):
        # generate mask
        # col_range =
        for i,name in enumerate(names):
            low=col_range[i][0]
            upper=col_range[i][1]
            baseroot= os.path.join(os.path.dirname(o_root),name)
            try:
                os.mkdir(baseroot)
                # so it will be like /root/crystal_mask
            except:
                pass
            for img_list in os.listdir(m_root):
                path = os.path.join(m_root,img_list)
                img = cv2.imread(path)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # get mask
                mask = cv2.inRange(hsv, low, upper)
                if  'liquor' in name:
                    mask = 255- mask
                img_list = img_list.replace('.tiff','')
                index=img_list.split('_')[-1]
                img_list = img_list.split('_')
                del img_list[-1]
                prefix = '_'.join(img_list)
                # pdb.set_trace()
                filename= '{}_{}_{}.tiff'.format(prefix,name,index)
                filepath=os.path.join(baseroot,filename)
                print(filepath)
                cv2.imwrite(filepath, mask)

class Image2Model(object):
    def __init__(self,segpath,modelpath,h_flip=False,v_flip=False,reverse=False,label=True,crop=False):
        self.path=segpath
        self.h_flip=h_flip
        self.v_flip = v_flip
        self.filename=modelpath
        self.reverse=reverse
        self.label=label
        self.crop=crop
        self.dataset= os.path.basename(self.filename).split("_")[0]
        try:
            self.change_names(prefix = "{}_tomobar".format(self.dataset))
        except:
            pass
        which_pic = str(int(len(os.listdir(self.path))/2))
        if which_pic in self.dataset:
            which_pic= str(int(which_pic)+1)

        for img_list in os.listdir(self. path) :
            if which_pic in img_list and 'tif' in img_list :
                        self.exam_image=os.path.join(self.path,img_list)
    def run( self ):
        print( "\n"
               "Please type down the rgb colours of the classes later after viewing the image\n"
               ""
               "in this software, \n"
               "crystal has pixel value of [3],\n"
               "background has pixel value of [0]\n"
               "liquor has pixel value of [1]\n"
               "loop has pixel value of[2]\n"
               "the other classes (e.g. bubble)has pixel value of [4] or above \n"
               "Please wait for the image of raw segmentation \n")
        try:
            self.find_color( self.exam_image )
        except:
            raise  RuntimeError("there is problem on reading the segmentation images, please check")

        correct_pixel="no"
        while correct_pixel != 'y' and correct_pixel != 'yes':
            input_air = input( "Please type down the rgb colours of the background class and press Enter \n"
                  "e.g. background=[ 0, 0, 0] or [0] (with closed square brackets)\n")
            input_cr = input( "Please type down the rgb colours of the crystal class and press Enter \n"
                  "e.g. crystal=[ 0, 0, 255] or [1] (with closed square brackets)\n")
            input_li = input( "Please type down the rgb colours of the liquor class and press Enter \n"
                  "e.g. liquor=[ 0, 255, ] or [3] (with closed square brackets)\n")
            input_lo = input( "Please type down the rgb colours of the loop class and press Enter \n"
                  "e.g. loop=[ 255, 0, 255] or [2] (with closed square brackets) \n")
            input_bu = input( "Please type down the rgb colours of the bubble class and press Enter \n"
                  "e.g. bubble=[ 255, 255, 255] or [4] if there is none enter [255] (with closed square brackets) \n")
            try:
                print("what you enter are \n" \
                      " background pixel values {} \n"
                      " crystal pixel values {} \n"
                      "liquor pixel values {} \n" 
                      "loop pixel values {} \n"
                      "bubble pixel values {} \n"
                     " Please check the values above are correct. \n".format(literal_eval(input_air),literal_eval(input_cr),
                                                         literal_eval(input_li),literal_eval(input_lo),literal_eval(input_bu))   )

                self.find_color( self.exam_image )

            except:
                RuntimeError("You have to assign the values to the classes and with closed square brackets  \n")
            correct_pixel=input("Please check the values above are correct. \n"
                                " If yes, enter y or yes to continue. If not, enter no to reenter the values \n")
        try:
            if input_cr :
                self.COLOR = {0 : literal_eval( input_air ) , 1 : literal_eval( input_li ) ,
                              2 : literal_eval( input_lo ) , 3 : literal_eval( input_cr ) ,
                              4 : literal_eval( input_bu )}
            else :
                self.COLOR = {0 : [0 , 0 , 0] , 1 : [0 , 0 , 255] , 2 : [0 , 255 , 0] , 3 : [255 , 0 , 0] ,
                              4 : [255 , 255 , 0]}
        except:
            RuntimeError( "You also have to enter the closed square brackets  \n" )

        img=io.imread(self.exam_image)
        if len(self.COLOR[0])==1:
            img=self.mask2mask(img,COLOR =self.COLOR)
        elif len(self.COLOR[0])==3:
            img=self.rgb2mask(img,COLOR =self.COLOR)
        else:
            RuntimeError( "The size of pixel value you entered is wrong. It has to be size of 1 (grey-scale) or 3 (rgb) \n" )

        print( "\n"
               "in this software, \n"
               "crystal has pixel value of [3],\n"
               "background has pixel value of [0]\n"
               "liquor has pixel value of [1]\n"
               "loop has pixel value of[2]\n"
               "the other classes (e.g. bubble)has pixel value of [4] or above \n"
               "Please check that the pixel values of the classes are definitely correct !!! \n")
        plt.title( "\n"
               "in this software, \n"
               "crystal has pixel value of [3],\n"
               "background has pixel value of [0]\n"
               "liquor has pixel value of [1]\n"
               "loop has pixel value of[2]\n"
               "the other classes (e.g. bubble)has pixel value of [4] or above \n"
               "Please check that the pixel values of the classes are definitely correct !!! \n "
                "If they are not correct, please rerun the program and enter again \n")
        # plt.figure(figsize = (19,12))
        plt.imshow(img)
        plt.show()
        hflip = input( "Does it need to be horizontally flipped? \n"
                "If yes, Press y or yes. Otherwise press any other keys \n" )
        if 'y' in hflip or 'yes' in hflip:
            self.h_flip=True
        else :
            self.h_flip = False

        vflip = input( "Does it need to be vertically flipped? \n"
                "If yes, Press y or yes. Otherwise press any other keys \n" )
        if 'y' in hflip or 'yes' in  vflip:
            self.v_flip=True
        else :
            self.v_flip = False

        order = input( "Does the stacking order need to be reversed? \n"
                "If yes, Press y or yes. Otherwise press any other keys \n" )
        if 'y' in hflip or 'yes'  in order:
            self.reverse=True
        else :
            self.reverse = False


        self.save_npy( label = True , crop = False )
        print("finish the 3D model generation")

        return self.filename
    def find_color (self, path) :
        # which_pic = str(int(len(os.listdir(m_root))/2))
        # if which_pic in self.dataset:
        #     which_pic= str(int(which_pic)+1)
        # for img_list in os.listdir( m_root ) :
        #     if which_pic in img_list and 'tif' in img_list :
        #         path = os.path.join( m_root , img_list )
        #         # img = cv2.imread( path )
        #         # hsv = cv2.cvtColor( img , cv2.COLOR_BGR2HSV )
        #         # fig = plt.figure( )
        hsv = io.imread(path)
        plt.imshow( hsv )
        plt.show( )



    def change_names ( self, prefix ) :
        print("\n Filenames are changing for data standardization...\n")
        with tqdm( total = len(os.listdir( self.path ) ) ) as pbar :
            for img_list in os.listdir( self.path ) :
                if 'tif' in img_list :
                    old = os.path.join(self.path , img_list )
                    # img_list=os.path.splitext(img_list)[0]
                    # abc = img_list.split('_')[-1][1:]

                    new = os.path.join( self.path, '{}_{}.tiff'.format( prefix , int(re.findall( r'\d+.' , os.path.basename( img_list ) )[-1][:-1] ) ) )

                    os.rename( old , new )
                    # except:
                    #     import shutil
                    #     shutil.move( old , new  )
                    pbar.update(1)
        print("\n All Filenames have been changed \n")
    def rgb2mask ( self,rgb , COLOR = None ) :
        """

        :param bgr: input mask to be converted to rgb image
        :param COLOR: 1:liquor,blue; 2: loop, green ; 3: crystal, red
        :return: rgb image
        """
        if COLOR is None :
            COLOR = self.COLOR
        mask = np.zeros( (rgb.shape[0] , rgb.shape[1]) )

        for k , v in COLOR.items( ) :
            mask[np.all( rgb == v , axis = 2 )] = k

        return mask

    def mask2mask( self,inputmask,COLOR = None):
        if COLOR is None :
            COLOR = self.COLOR
        mask = np.zeros( inputmask.shape)
        for k , v in COLOR.items( ) :
            mask[inputmask == v]=k

        return mask

    def mask2rgb ( self,mask , COLOR = None ) :
        """

        :param mask: input mask to be converted to rgb image
        :param COLOR: 1:liquor,blue; 2: loop, green ; 3: crystal, red
        :return: rgb image
        """
        if COLOR is None :
            COLOR = self.COLOR

        rgb = np.zeros( mask.shape + (3 ,) , dtype = np.uint8 )

        for i in np.unique( mask ) :
            rgb[mask == i] = COLOR[i]

        return rgb

    def save_npy ( self,label = True , crop = False ) :
        """

        :param path: path should directed to image path
        :param filename:
        :param label:
        :param crop:  #[y1:y2,x1:x2]
        :return:
        """
        na = []
        for root , dir , files in os.walk( self.path ) :
            for file in files :
                if 'tif' in file :
                    na.append( os.path.join( root , file ) )

        def take_num ( ele ) :
            return int( re.findall( r'\d+' , ele )[-1] )

        # sort the list according to the last index of the filename
        na.sort( key = take_num , reverse = self.reverse )

        if self.v_flip :
            prefix = os.path.basename( self.filename ).split( '.' )[0]
            dir = os.path.dirname( self.filename )
            prefix = prefix + "_vf.npy"
            self.filename = os.path.join( dir , prefix )

        if self.h_flip :
            prefix = os.path.basename( self.filename ).split( '.' )[0]
            dir = os.path.dirname( self.filename )
            prefix = prefix + "_hf.npy"
            self.filename = os.path.join( dir , prefix )

        if self.reverse :
            prefix = os.path.basename( self.filename ).split( '.' )[0]
            dir = os.path.dirname( self.filename )
            prefix = prefix + "_r.npy"
            self.filename = os.path.join( dir , prefix )
        print( "\n 3D model is generating...\n" 
               "storing in {} \n".format(self.filename) )

        with tqdm( total = len(na) ) as pbar :
            for i , file in enumerate( na ) :

                if i == 0 :
                    file = os.path.join( self.path , file )

                    img = io.imread( file )
                    if self.h_flip:
                        img = cv2.flip(img,1)

                    if self.v_flip :
                        img = cv2.flip( img , 0 )
                    if crop :
                        img = img[crop[0] :crop[1] , crop[2] :crop[3]]  # [y1:y2,x1:x2]
                    if label :
                        if self.COLOR[0]==3:
                            img = self.rgb2mask( img )
                        else:
                            img = self.mask2mask( img )
                        img = img.astype( np.int8 )
                    img = np.expand_dims( img , axis = 0 )
                    stack = img
                    # pdb.set_trace()
                else :

                    # index = file.split('.')[0][-4:].lstrip('0')
                    # index = file.split( '.' )[0].split( '_' )[-1]
                    # index = re.findall( r'\d+' , file )[-1]
                    # assert i == int(index)
                    file = os.path.join( self.path , file )
                    img = io.imread( file )
                    if self.h_flip:
                        img = cv2.flip(img,1)

                    if self.v_flip :
                        img = cv2.flip( img , 0 )

                    if crop :
                        img = img[crop[0] :crop[1] , crop[2] :crop[3]]  # [y1:y2,x1:x2]
                    if label :
                        if self.COLOR[0]==3:
                            img = self.rgb2mask( img )
                        else:
                            img = self.mask2mask( img )

                        img = img.astype( np.int8 )
                    img = np.expand_dims( img , axis = 0 )
                    stack = np.concatenate( (stack , img) , axis = 0 )
                    # print( '{} is attached'.format( index ) )
                pbar.update(1)


        if label :
            stack_int = stack.astype( np.int8 )
            np.save( self.filename , stack_int )
        else :
            np.save( self.filename , stack )


    def projection(self,object,angle):
        # based on the skimage.transform.rotate(2D) and it rotates about y axis
        # r_matrix=np.array([[cos(angle),0,sin(angle)],
        #                    [0, 1, 0],
        #                    [-sin(angle),0,cos(angle)]])
        if object.dtype == 'uint8':
            object=object.astype(np.float64)/255
        Z,Y,X=object.shape
        for i in range(Y):
            object[:,i,:]=trans.rotate(object[:,i,:],angle,mode='constant', cval=0,)

        proj =np.max(object,axis=0)

        return proj







