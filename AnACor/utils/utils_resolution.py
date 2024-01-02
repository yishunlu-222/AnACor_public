import numpy as np
import pdb
import argparse
import cv2
# from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import timeit
import os
import json
from scipy.ndimage import zoom

"""
extrapolation:
https://analyticsindiamag.com/an-illustrative-guide-to-extrapolation-in-machine-learning/
"""

def model3D_resize(model,factors,save_dir=None):
    t1=timeit.default_timer()
    z,y,x=model.shape
    opencv=True
    desired_x=int(x * factors[2])
    desired_y = int( y * factors[1] )
    desired_z = int( z * factors[0] )
    if opencv is not True:
        new_model_xz = zoom(model, factors, order=0)

    else:

        new_model_xy=np.zeros((z,desired_y,desired_x))
        new_model_xz = np.zeros( (desired_z , desired_y , desired_x) )
        
        for i,img in enumerate(model):
            new_img = cv2.resize( img , (desired_x,desired_y) , interpolation = cv2.INTER_NEAREST )
            # new_img = cv2.resize( img , (desired_x,desired_y) , interpolation =cv2.INTER_CUBIC)
            new_model_xy[i]=new_img

        for j in range(desired_y):
            slice =  new_model_xy[:,j,:]
            final_img = cv2.resize( slice, (desired_x,desired_z) , interpolation = cv2. INTER_NEAREST )
            # final_img = cv2.resize( slice, (desired_x,desired_z) , interpolation = cv2.INTER_CUBIC )

            new_model_xz[:,j,:] = final_img
    
    t2=timeit.default_timer()
    print("time for resizing is {}".format(t2-t1))
    # try:
    #     difference = np.unique( new_model ) -np.unique(model)
    #     if np.count_nonzero(difference) >0:
    #         new_model = np.floor( new_model )
    # except:
    #     new_model=np.floor(new_model)
    # pdb.set_trace()
    if save_dir is not None:
        plt.clf( )
        plt.imshow(new_model_xz[int(desired_z/2)])
        plt.title("middle slice of the model for factor of {}".format(factors[0]),fontsize=12)
        plt.savefig(os.path.join(save_dir,'middle slice for factor of {}.png'.format(factors[0])),dpi=600)
        print("factor {} is saved".format(factors))
    # 
    # try:
    #     difference = np.unique( new_model ) -np.unique(model)
    #     if np.count_nonzero(difference) >0:
    #         new_model = np.floor( new_model )
    # except:
    #     new_model=np.floor(new_model)
    return new_model_xz.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="changing resolution of the 3D model")
    parser.add_argument(
        "--filepath",
        type=str,
        default="./",
        help="filepath of the 3D model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="save-dir of the 3D model",
    )
    global args
    args = parser.parse_args()
    voxel_size = [0.3, 0.3, 0.3]
    new_voxel_size = [0.35,0.4,0.5,0.6, 0.9, 1.2,1.5,1.8,2.1,2.4,2.7,3.0]
    param=[new_voxel_size]
    new_voxel_size =np. array(new_voxel_size)
    factors_list = voxel_size[0] /new_voxel_size 
    factors_list = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,1.2,1.5])
    new_voxel_size=0.3  / factors_list
    modelname=os.path.basename(args.filepath)
    print("model is {}".format(args.filepath))
    print("target dir is {}".format(args.save_dir))
    prefix,afterfix=modelname.split('.')
    save_dir=os.path.join(args.save_dir, prefix+"resolution_cv_near" )
    try:
        os.makedirs(save_dir)
    except:
        pass

    model=np.load(args.filepath).astype("uint8")

    
    # plt.imshow(model[int(model.shape[0]/2)])
    # plt.title("middle slice of the model for factor of 1",fontsize=12)
    # plt.savefig(os.path.join(save_dir,"middle slice for factor of 1.png"))
    starttime = timeit.default_timer( )
    print( "The start time is :" , starttime )
    time_list=[]
    for i, factor in enumerate(factors_list):
        new_name=prefix+"{}_{}".format(new_voxel_size[i],np.around(factor,3))+'.'+afterfix
        factors=[factor, factor,factor ]
        new = model3D_resize( model,factors,save_dir )
        np.save(os.path.join(save_dir,new_name),new)
        print("voxel size {} factor {} is saved".format(new_voxel_size[i],factors))
        print( "The time difference is :" , timeit.default_timer( ) - starttime )
        time_list.append(timeit.default_timer( ) - starttime)
    pdb.set_trace()
    param.append(time_list)

    with open(os.path.join(save_dir,f"{prefix}resolution_time.json"),'w') as f:
        json.dump(param, f, indent=2)
        
    # np.save( os.path.join( save_dir ,modelname ) ,model)
#    new_name=prefix+"_{}".format('z0.5')+'.'+afterfix
#    factors=[0.5 ,1 ,1 ]
#    new = resize( model,factors,save_dir )
#
#    np.save(os.path.join(save_dir,new_name),new)
#    print("factor {} is saved".format(factors))
#    
#    new_name=prefix+"_{}".format('y0.5')+'.'+afterfix
#    factors=[1 ,0.5 ,1 ]
#    new = resize( model,factors,save_dir )
#    np.save(os.path.join(save_dir,new_name),new)
#    print("factor {} is saved".format(factors))
#    
#    new_name=prefix+"_{}".format('x0.5')+'.'+afterfix
#    factors=[1 ,1 ,0.5 ]
#    new = resize( model,factors,save_dir )
#    np.save(os.path.join(save_dir,new_name),new)
#    print("factor {} is saved".format(factors))
#    pdb.set_trace()
    # for factor in [0.95, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]:
    #     new_name=prefix+"_z{}".format(factor)+'.'+afterfix
    #     factors=[factor, 1,1 ]
    #     new = resize( model,factors,save_dir )
    #     np.save(os.path.join(save_dir,new_name),new)
    #     print("factor {} is saved".format(factors))
    # print( "The time difference is :" , timeit.default_timer( ) - starttime )
    
    # for factor in [0.95, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]:
    #     new_name=prefix+"_y{}".format(factor)+'.'+afterfix
    #     factors=[1 ,factor ,1 ]
    #     new = resize( model,factors,save_dir )
    #     np.save(os.path.join(save_dir,new_name),new)
    #     print("factor {} is saved".format(factors))
    # print( "The time difference is :" , timeit.default_timer( ) - starttime )
    
    # for factor in [0.95, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]:
    #     new_name=prefix+"_x{}".format(factor)+'.'+afterfix
    #     factors=[1, 1,factor ]
    #     new = resize( model,factors,save_dir )
    #     np.save(os.path.join(save_dir,new_name),new)
    #     print("factor {} is saved".format(factors))
    # print( "The time difference is :" , timeit.default_timer( ) - starttime )
    

