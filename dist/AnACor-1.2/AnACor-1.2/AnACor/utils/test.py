import skimage.io as io 
from skimage.filters import threshold_mean
import pdb
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mask_generation (  img1 , thresh ) :
        output = np.ones( img1.shape )
        mask = img1 < thresh
        mask = output * mask
        return mask

img = io.imread('D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/img_20072_rot_00765.tiff')
width_low, width_high, height_low, height_high = [20,:,:,:]
img = img[:,20:]
plt.imshow(img)
plt.show()
# img= cv2.normalize(
#             img , None , 0 , 255 , cv2.NORM_MINMAX ).astype( 'uint8' )
thre= threshold_mean(img)
plt.imshow(img)
plt.show()
img =mask_generation(img,thre)
pdb.set_trace()
