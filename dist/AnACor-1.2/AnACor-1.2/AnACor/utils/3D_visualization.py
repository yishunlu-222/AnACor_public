import pdb
import numpy as np
# from main3_general_pl import partial_illumination_selection
# from utils_ib import partial_illumination_selection,kp_rotation
from utils_rt import slice_sampling
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import plotly.io as pio
import os

def line_plotly(ray, point_on_axis,color='green',scale=10):

    direction_scaled = [scale * i for i in ray]

    # Calculate the end point of the line
    x_end = point_on_axis[2] + direction_scaled[2]
    y_end = point_on_axis[1] + direction_scaled[1]
    z_end = point_on_axis[0] + direction_scaled[0]

    # Create a line plot representing the direction of goniometer rotation
    line = go.Scatter3d(x=[point_on_axis[2], x_end], 
                        y=[point_on_axis[1], y_end],
                        z=[point_on_axis[0], z_end],
                        mode='lines',
                        line=dict(width=6, color=color))
    return 

def transform_all(sample_coordinate,total_rotation_matrix,centre_point_on_axis):
    for i, coord in enumerate(sample_coordinate):
        tmp_coord= coord -centre_point_on_axis
        tmp_coord=np.dot(total_rotation_matrix,tmp_coord)
        tmp_coord+=centre_point_on_axis
        sample_coordinate[i]=tmp_coord
    zz_all, yy_all, xx_all = np.split(sample_coordinate.T, 3)
    return zz_all[0], yy_all[0], xx_all[0],sample_coordinate

def matrix_2_axis_angle(rotation_matrix):
     #from rotation matrix to axis angle
    rotation = Rotation.from_matrix(rotation_matrix)
    axis_angle = rotation.as_rotvec()

    # Extract the axis and angle components
    axis = axis_angle / np.linalg.norm(axis_angle)  # Normalize the axis vector
    angle = np.linalg.norm(axis_angle)  
    print("axis",axis)
    print("angle",angle)
    return axis,angle


def test_partial_illumination_selection_real(filename,real_matrix,kp,Z,Y,X,omega_d,kp_rotation_axis):
    """
        Z-axis is the rotation axis
        X-axis is the X-ray direction
        [Z,Y,X]
    """
    
    os.makedirs("result_{}_{}_{}".format(Z,Y,X),exist_ok=True)
    factor=0.1
    centre_point_on_axis=(np.array([Z,Y,X]) *factor ).astype(int)
    pixel_size = 0.3 * 1e-3 
    omega_rotation_axis=np.array([1,0,0])
    xray=np.array([0,0,1])
    label_list=np.load(filename)
    omega=np.pi/180*omega_d
    

    # [x,y,z],kappa=matrix_2_axis_angle(real_matrix)
    # kp_rotation_axis=-np.array([z,y,x])
    # kappa=-kappa

    # pdb.set_trace()
    # omega+=np.pi/180*90

    
    width =int(0.240/pixel_size/2*factor)
    height=int(0.150/pixel_size/2*factor)
    xray_region=[ centre_point_on_axis[1]-height,centre_point_on_axis[1]+height,centre_point_on_axis[0]-width,centre_point_on_axis[0]+width]  

    zz, yy, xx = np.where(label_list == 3)  
    zz_al, yy_al, xx_al = np.where(label_list >0) 
    crystal_coordinate = np.stack((zz,yy,xx),axis=1)
    sample_coordinate= np.stack((zz_al,yy_al,xx_al),axis=1)
        # Define the x-ray region   
    # Iterate through the array to calculate the covered coordinates
    covered_coords = []


    omega_matrix=kp_rotation(omega_rotation_axis, omega)

    total_rotation_matrix=np.dot(real_matrix,omega_matrix)
    """ rotate the X-ray relatively to match the absorption correction """
    # rotated_xray=np.dot(total_rotation_matrix,xray)

    zz, yy, xx,crystal_coordinate =  transform_all(crystal_coordinate,total_rotation_matrix,centre_point_on_axis)
    zz_all, yy_all, xx_all,sample_coordinate =  transform_all(sample_coordinate,total_rotation_matrix,centre_point_on_axis)


    for  coord in crystal_coordinate: 
                if partial_illumination_selection(xray_region, total_rotation_matrix, coord, centre_point_on_axis,rotated=True):
                    covered_coords.append(coord)

    # pdb.set_trace()
    # Create a scatter3D plot for all points, indicating the whole 10x10x10 box
    """
    In Plotly, the axes are conventionally interpreted as [X, Y, Z] just like in most common 3D plotting or graphic systems. So it does not follow the same axis convention as numpy, which is often used in image processing where the convention can be [Z, Y, X].
    """

    scatter3d_array = go.Scatter3d(x=xx_all,
                                   y=yy_all,
                                   z=zz_all,
                                   mode='markers',
                                   marker=dict(size=1, 
                                               color='rgba(50, 50, 50, 0.5)', # Semi-transparent
                                               symbol='square'))

    # Generate x, y, and z lists for points in the covered_coords
    x_covered = [coord[2] for coord in covered_coords]
    y_covered = [coord[1] for coord in covered_coords]
    z_covered = [coord[0] for coord in covered_coords]
    # xx=[]
    # yy=[]
    # zz=[]
    # x_covered = []
    # y_covered = []
    # z_covered = []
    # pdb.set_trace()
    # Create a scatter3D plot for the covered region
    scatter3d_covered = go.Scatter3d(x=xx,
                                     y=yy,
                                     z=zz,
                                     mode='markers',
                                     marker=dict(size=6, 
                                                 color='red', # Solid red
                                                 symbol='square')) 


    scatter3d_covered_2 = go.Scatter3d(x=x_covered,
                                     y=y_covered,
                                     z=z_covered,
                                     mode='markers',
                                     marker=dict(size=6, 
                                                 color='rgba(0, 0, 0, 1)', # Solid red
                                                 symbol='square')) 
    # Create a line plot representing the direction of the x-ray
    line_0 = line_plotly(omega_rotation_axis, centre_point_on_axis,color='green',scale=100)
    line_1 = line_plotly(kp_rotation_axis, centre_point_on_axis,color='purple',scale=100)
    line_2=line_plotly(xray,centre_point_on_axis,color='black',scale=100)

    region_point_1=np.array([xray_region[3],xray_region[0],0])
    region_point_2=np.array([xray_region[3],xray_region[1],0])
    region_point_3=np.array([xray_region[2],xray_region[0],0])
    region_point_4=np.array([xray_region[2],xray_region[1],0])
    line_region_1=line_plotly(xray,region_point_1,color='blue',scale=100)
    line_region_2=line_plotly(xray,region_point_2,color='blue',scale=100)
    line_region_3=line_plotly(xray,region_point_3,color='blue',scale=100)
    line_region_4=line_plotly(xray,region_point_4,color='blue',scale=100)
    # Create a layout object
    layout = go.Layout(scene=dict(aspectmode='cube',            xaxis=dict(range=[0, 1000*factor]),
        yaxis=dict(range=[0, 1000*factor]),
        zaxis=dict(range=[0, 1000*factor])))

    # Create a figure and add the scatter plots
    fig = go.Figure(data=[scatter3d_array, scatter3d_covered,scatter3d_covered_2, line_0,line_1,line_2,line_region_1,line_region_2,line_region_3,line_region_4], layout=layout)
    fig.update_layout(scene_camera=dict(
    eye=dict(x=1, y=-0.5, z=0), up=dict(x=0, y=-1, z=0)
))
    # # Get the current default configuration
    # template = go.layout.Template()

    # # Update the default camera settings
    # template.layout.scene.camera.eye = dict(x=0, y=1, z=0)

    # # Set the updated configuration as the new default
    # pio.templates.default = template
    # pdb.set_trace()
    fig.write_html("./result_{}_{}_{}/{}_omega_{}_axis_at_centre_at{}.html".format(Z,Y,X,kp,omega_d,centre_point_on_axis))
    fig.write_image("./result_{}_{}_{}/{}_omega_{}_axis_centre_at{}.png".format(Z,Y,X,kp,omega_d,centre_point_on_axis),width=1600, height=1200)
    # pdb.set_trace()
    # Show the figure
    # fig.show()

def plot_projection(raw_model, Y,Z,width,height):
    new=raw_model.mean(axis=1)
    xray_region=[ Y-height,Y+height,Z-width,Z+width]  
    plt.imshow(new)
    plt.hlines(y=xray_region[2], color='blue', linestyle='--', xmin=xray_region[0], xmax=xray_region[1])
    plt.hlines(y=xray_region[3], color='blue', linestyle='--', xmin=xray_region[0], xmax=xray_region[1])
    plt.vlines(x=xray_region[0], color='blue', linestyle='--', ymin=xray_region[2], ymax=xray_region[3])
    plt.vlines(x=xray_region[1], color='blue', linestyle='--', ymin=xray_region[2], ymax=xray_region[3])
    # plt.axvline(x=-0.5, color='blue', linestyle='--', xmin=0.3, xmax=0.7)
    plt.plot(Y, Z, 'ro') 
    # plt.show()
    plt.savefig("./centre_at_{}_{}_{}_ZX.png".format(Z,Y,X),dpi=300)
    plt.clf()
    new=raw_model.mean(axis=2)
    xray_region=[ Y-height,Y+height,Z-width,Z+width]  
    plt.imshow(new)
    plt.hlines(y=xray_region[2], color='blue', linestyle='--', xmin=xray_region[0], xmax=xray_region[1])
    plt.hlines(y=xray_region[3], color='blue', linestyle='--', xmin=xray_region[0], xmax=xray_region[1])
    plt.vlines(x=xray_region[0], color='blue', linestyle='--', ymin=xray_region[2], ymax=xray_region[3])
    plt.vlines(x=xray_region[1], color='blue', linestyle='--', ymin=xray_region[2], ymax=xray_region[3])
    # plt.axvline(x=-0.5, color='blue', linestyle='--', xmin=0.3, xmax=0.7)
    plt.plot(Y, Z, 'ro') 
    # plt.show()
    plt.savefig("./centre_at_{}_{}_{}_ZY.png".format(Z,Y,X),dpi=300)
    plt.clf()   

def plotting_sampling(filename,sampling_number=10):
    """
        Z-axis is the rotation axis
        X-axis is the X-ray direction
        [Z,Y,X]
    """
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}

    factor=0.1

    label_list=np.load(filename)

    coord_list = slice_sampling(label_list, sampling=sampling_number,
                                rate_list=rate_list, auto=False)
    pdb.set_trace()

    zz, yy, xx = np.where(label_list == 3)  
    print('len of zz',len(zz))
    zz_all, yy_all, xx_all = np.where(label_list >0) 
        # Define the x-ray region   
    # Iterate through the array to calculate the covered coordinates
    covered_coords = coord_list


    # pdb.set_trace()
    # Create a scatter3D plot for all points, indicating the whole 10x10x10 box
    """
    In Plotly, the axes are conventionally interpreted as [X, Y, Z] just like in most common 3D plotting or graphic systems. So it does not follow the same axis convention as numpy, which is often used in image processing where the convention can be [Z, Y, X].
    """

    scatter3d_array = go.Scatter3d(x=xx_all,
                                   y=yy_all,
                                   z=zz_all,
                                   mode='markers',
                                   marker=dict(size=1, 
                                               color='rgba(50, 50, 50, 0.5)', # Semi-transparent
                                               symbol='square'))

    # Generate x, y, and z lists for points in the covered_coords
    x_covered = [coord[2] for coord in covered_coords]
    y_covered = [coord[1] for coord in covered_coords]
    z_covered = [coord[0] for coord in covered_coords]
    # xx=[]
    # yy=[]
    # zz=[]
    # x_covered = []
    # y_covered = []
    # z_covered = []
    # pdb.set_trace()
    # Create a scatter3D plot for the covered region
    scatter3d_covered = go.Scatter3d(x=xx,
                                     y=yy,
                                     z=zz,
                                     mode='markers',
                                     marker=dict(size=6, 
                                                 color='red', # Solid red
                                                 symbol='square')) 


    scatter3d_covered_2 = go.Scatter3d(x=x_covered,
                                     y=y_covered,
                                     z=z_covered,
                                     mode='markers',
                                     marker=dict(size=6, 
                                                 color='rgba(0, 0, 0, 1)', # Solid red
                                                 symbol='square')) 
    # Create a line plot representing the direction of the x-ray

    # Create a layout object
    layout = go.Layout(scene=dict(aspectmode='cube',            xaxis=dict(range=[0, 1000*factor]),
        yaxis=dict(range=[0, 1000*factor]),
        zaxis=dict(range=[0, 1000*factor])))

    # Create a figure and add the scatter plots
    fig = go.Figure(data=[scatter3d_array, scatter3d_covered,scatter3d_covered_2], layout=layout)
    fig.update_layout(scene_camera=dict(
    eye=dict(x=1, y=-0.5, z=0), up=dict(x=0, y=-1, z=0)
))
    # # Get the current default configuration
    # template = go.layout.Template()

    # # Update the default camera settings
    # template.layout.scene.camera.eye = dict(x=0, y=1, z=0)

    # # Set the updated configuration as the new default
    # pio.templates.default = template
    # pdb.set_trace()
    # fig.write_html("./result_{}_{}_{}/{}_omega_{}_axis_at_centre_at{}.html".format(Z,Y,X,kp,omega_d,centre_point_on_axis))
    # fig.write_image("./result_{}_{}_{}/{}_omega_{}_axis_centre_at{}.png".format(Z,Y,X,kp,omega_d,centre_point_on_axis),width=1600, height=1200)
    # pdb.set_trace()
    # Show the figure
    fig.show()





if __name__ == "__main__":
    # test_partial_illumination_selection_basic()
    # test_partial_illumination_selection_kp()
    # kp_rotation_axis = np.array([0.642788,-0.766044,0]) 


    """===================plotting==================="""

    # raw = "D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/16010_tomobar_cropped_f.npy"
    # raw_model=np.load(raw)
    # for Z in [560,570,580]:
    #     Y=493
    #     width =int(0.240/0.0003/2)
    #     height=int(0.150/0.0003/2)
    #     X=Y # as after rotation, they can be swapped
    #     plot_projection(Y,Z,width,height)

    """===================ploty on 3D on partial illumination==================="""
    # for Z in [480,490, 500,510,520,530,540,550,560,570,580]:
    #     Y=493
    #     X=Y # as after rotation, they can be swapped

    #     for angle in [0,30,45,60,90,120,135,150,180,210,225,240,270,300,315,330,360]:
    #         real_matrix=np.array([[ 1, 0,  0],
    #         [0,  1,  0],
    #         [0, 0,  1]]) #k0
    #         kp='k0'
    #         test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
            
    #         real_matrix =np.array([[ 0.61388157,  0.16035909,  0.77293879],
    #         [-0.46140928,  0.86736106,  0.18651078],
    #         [-0.64050831, -0.47113666,  0.60644814]]) #km70_pm120
    #         kp='km70_pm120'
    #         test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
            

    #         real_matrix=np.array([[ 0.61388157, -0.32399183,  0.71984631],
    #         [-0.32399183,  0.72813857,  0.60402277],
    #         [-0.71984631, -0.60402277,  0.34202014]]) #km70
    #         kp='km70'
    #         test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
        
    #         print("angle {} is finished".format(angle))

    """===================ploty on 3D on slice sampling==================="""
    """===================ploty on 3D on slice sampling==================="""
    """===================ploty on 3D on slice sampling==================="""
    filename='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/partial_illumination/16010_tomobar_cropped_f_0.1.npy'
    plotting_sampling(filename,sampling_number=50)