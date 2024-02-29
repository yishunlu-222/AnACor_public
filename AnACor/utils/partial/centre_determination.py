import numpy as np
import tomopy
import os
import skimage.io as io
import pdb
import re
import matplotlib.pyplot as plt
import cv2

def normalize_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image to the range 0-255
    normalized_image_array = ((image_array - image_array.min()) * (255 / (image_array.max() - image_array.min()))).astype(np.uint8)

    return normalized_image_array


def find_center():
    # Let's assume you've read your 1800 images into a 3D numpy array called 'projections'.
    # 'projections' should have a shape like (1800, num_pixels_y, num_pixels_x)
    # where num_pixels_y and num_pixels_x are the number of pixels along the vertical and horizontal directions of your images.

    # Make an array of the angles at which each projection was taken.
    # In your case, you have 1800 projections over 180 degrees, so each projection is 0.1 degree apart.
    na = []
    project=[]
    pth='/data/dataset/flat_field/16010_full_scale/TiffSaver_1/'
    for root , dir , files in os.walk(pth ) :
        for file in files :
            if 'tif' in file :
                na.append( os.path.join( root , file ) )

    def take_num ( ele ) :
        return int( re.findall( r'\d+' , ele )[-1] )

    # sort the list according to the last index of the filename
    na.sort( key = take_num , reverse = False)
    increment=2
    total=100
    centers=[]
    proj1=cv2.imread(na[0] , 2 )
    proj1[proj1<0]=0
    proj1[proj1>1]=1
    # proj1= cv2.normalize(proj1, None , 0 , 255 , cv2.NORM_MINMAX ).astype(
    #         'uint8' )
    proj1=normalize_image(proj1)
    proj2=io.imread(na[-1])
    sc = tomopy.find_center_pc(proj1, proj2, tol=0.5, rotc_guess=None) #=603
    plt.imshow(proj1)

    plt.axvline(x=sc, color='r')
    plt.text(sc*1.05, 100, 'x = {}'.format(sc), color='r', fontsize=24)
    plt.show()
    for i in range(total):
        proj1=io.imread(na[i])
        proj2=io.imread(na[-1-i])
        sc = tomopy.find_center_pc(proj1, proj2, tol=0.5, rotc_guess=None)
        pdb.set_trace()
        centers.append(sc)
    projections=np.array(project)
    angles = np.linspace(0, np.pi, int(1800/increment)+1)
    pdb.set_trace()


    # Use tomopy's built-in functions to find the rotation center.
    rot_center = tomopy.find_center(projections, angles, init=None, ind=0, tol=0.5)

    print(f"The rotation center is located at pixel column {rot_center}.")


def experiment_setup():
    origin=[186.091,-8.63189,-258.838]
    fast_axis=np.array([-0.999997,-0.0022969,-0.0011057])
    slow_axis=np.array([-0.00233309,0.999422,0.0339201])
    pixel_size=[0.172,0.172]
    panel_pixel=[1080.22,52.70]
    origin+panel_pixel[0]*pixel_size[0]*fast_axis+panel_pixel[1]*pixel_size[1]*slow_axis
    Rotation=[1,0,0]

if __name__ == '__main__':
    find_center()
    # experiment_setup()