import pdb
import numpy as np
# from main3_general_pl import partial_illumination_selection
from utils_lite import partial_illumination_selection,kp_rotation
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import plotly.io as pio
import os
    # mm: panel 12, (185.80,9.06)
    # px: panel 12, (1080.22,52.70)
    # mm, raw image: (185.80,446.63)
    # px, raw image: (1080.22,2596.70)


# Panel:
#   name: row-12
#   type: SENSOR_PAD
#   identifier:
#   pixel_size:{0.172,0.172}
#   image_size: {2463,195}
#   trusted_range: {-1,1e+06}
#   thickness: 0.32
#   material: Si
#   mu: 150.879
#   gain: 1
#   pedestal: 0
#   fast_axis: {-0.999997,-0.0022969,-0.0011057}
#   slow_axis: {-0.00233309,0.999422,0.0339201}
#   origin: {186.091,-8.63189,-258.838}
#   distance: 258.587
#   pixel to millimeter strategy: ParallaxCorrectedPxMmStrategy
#     mu: 150.879
#     t0: 0.32

# Panel:
#   name: row-00
#   type: SENSOR_PAD
#   identifier:
#   pixel_size:{0.172,0.172}
#   image_size: {2463,195}
#   trusted_range: {-1,1e+06}
#   thickness: 0.32
#   material: Si
#   mu: 150.879
#   gain: 1
#   pedestal: 0
#   fast_axis: {-0.999997,-0.0022969,-0.0011057}
#   slow_axis: {0.00142141,-0.142352,-0.989815}
#   origin: {186.3,-245.384,43.8058}
#   distance: 249.515
#   pixel to millimeter strategy: ParallaxCorrectedPxMmStrategy
#     mu: 150.879
#     t0: 0.32




def test_partial_illumination_selection_kp():


    xray_region = [3, 7, 3, 7]

    point_on_axis = np.array([5, 5, 5])
    axis =np.array([1,1,1])
    # total_rotation_matrix=kp_rotation(axis,np.pi/6)
    plot_voxels_plotly(point_on_axis,axis,xray_region,omega=np.pi/6)


def plot_voxels_plotly(point_on_axis= np.array([5, 5, 5]),rotation_axis=np.array([1,0,0]),xray_region=[3, 7, 3, 7] ,omega=np.pi / 4,xray=np.array([0,0,1])):



    # Define the x-ray region   
    # Iterate through the array to calculate the covered coordinates
    covered_coords = []
    x = []
    y = []
    z = []
    total_rotation_matrix = kp_rotation(rotation_axis, omega)

    """ rotate the X-ray relatively to match the absorption correction """
    rotated_xray=np.dot(total_rotation_matrix.T,xray)

 
    for  i in range(10):
        for j in range(10):
            for k in range(10):
                coord = np.array([i, j, k])
                x.append(k)
                y.append(j)
                z.append(i)
                if partial_illumination_selection(xray_region, total_rotation_matrix, coord, point_on_axis) is True:

                    covered_coords.append(coord)


    # Create a scatter3D plot for all points, indicating the whole 10x10x10 box
    """
    In Plotly, the axes are conventionally interpreted as [X, Y, Z] just like in most common 3D plotting or graphic systems. So it does not follow the same axis convention as numpy, which is often used in image processing where the convention can be [Z, Y, X].
    """
    scatter3d_array = go.Scatter3d(x=x,
                                   y=y,
                                   z=z,
                                   mode='markers',
                                   marker=dict(size=6, 
                                               color='rgba(0, 0, 0, 0.2)', # Semi-transparent
                                               symbol='square'))
    pdb.set_trace()
    # Generate x, y, and z lists for points in the covered_coords
    z_covered = [coord[0] for coord in covered_coords]
    y_covered = [coord[1] for coord in covered_coords]
    x_covered = [coord[2] for coord in covered_coords]

    # Create a scatter3D plot for the covered region
    scatter3d_covered = go.Scatter3d(x=x_covered,
                                     y=y_covered,
                                     z=z_covered,
                                     mode='markers',
                                     marker=dict(size=6, 
                                                 color='red', # Solid red
                                                 symbol='square')) 

    # Create a line plot representing the direction of the x-ray
    line = line_plotly(rotation_axis, point_on_axis,color='green')
    line_2=line_plotly(rotated_xray,point_on_axis,color='black')
    # Create a layout object
    layout = go.Layout(scene=dict(aspectmode='cube'))

    # Create a figure and add the scatter plots
    fig = go.Figure(data=[scatter3d_array, scatter3d_covered, line,line_2], layout=layout)

    # Show the figure
    fig.show()

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
    return line
def test_partial_illumination_selection_basic():
    # this test assumes the rotation axis is the Z axis and the rotation axis goes through the origin
    # If your rotation axis is not around the origin, you will need to perform a translation so that your axis of rotation passes through the origin, do the rotation, then translate back.

    xray_region = [3, 7, 3, 7]
    # define the rotation matrix for 45 degrees rotation around Z axis

    # total_rotation_matrix = np.array([[1,0,0],
    #                                   [0,np.cos(omega), np.sin(omega) ],
    #                                   [0,-np.sin(omega), np.cos(omega)]])
    # coord = [5, 5, 5]
    # assert partial_illumination_selection(xray_region, total_rotation_matrix, coord), "Test 0 failed"
    point_on_axis = np.array([5, 5, 5])
    axis =np.array([1,0,0])
    for omega in range(0, 360, 45):

        # omega = np.pi /180 *45 # omega  # 45 degree rotation
        omega=omega*np.pi/180
        total_rotation_matrix = kp_rotation(axis,omega)


        # # X-ray region (assuming it's in the middle of a 10x10x10 cube)


        coord = [5, 5, 5]
        assert partial_illumination_selection(xray_region, total_rotation_matrix, coord,point_on_axis), "angle_{}:  Test 1 failed".format(omega*180/np.pi)
        print("angle_{}: Test 1 passed".format(omega*180/np.pi))


        coord = [3.5, 3.5, 3.5]
        assert  partial_illumination_selection(xray_region, total_rotation_matrix, coord,point_on_axis), "angle_{}: Test 2 failed".format(omega*180/np.pi)
        print("angle_{}: Test 2 passed".format(omega*180/np.pi))

        # Test 3: point inside the X-ray region after rotation
        coord = [6.5, 6.5, 6.5]  # this point will not be inside the X-ray region after 45 degrees rotation
        assert partial_illumination_selection(xray_region, total_rotation_matrix, coord,point_on_axis), "angle_{}: Test 3 failed".format(omega*180/np.pi)
        print("angle_{}: Test 3 passed".format(omega*180/np.pi))

        # Test 4: 
        coord = [2, 2, 2]  # this point will be outside the X-ray region after 90 degrees rotation
        assert not partial_illumination_selection(xray_region, total_rotation_matrix, coord,point_on_axis), "angle_{}:  Test 4 failed".format(omega*180/np.pi)
        print("angle_{}: Test 4 passed".format(omega*180/np.pi))

        # Test 5: maximum  distance is larger than the X-ray region (7-5)^3 * 3=24 and always outside the X-ray region,
        coord = [8, 1, 7]  # this point will be inside the X-ray region after 45 degrees rotation
        assert not  partial_illumination_selection(xray_region, total_rotation_matrix, coord,point_on_axis), "angle_{}:  Test 5 failed".format(omega*180/np.pi)
        print("angle_{}: Test 5 passed".format(omega*180/np.pi))


    plot_voxels_plotly(omega=np.pi/180*128)

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
if __name__ == "__main__":
    # test_partial_illumination_selection_basic()
    # test_partial_illumination_selection_kp()
    kp_rotation_axis = np.array([0.642788,-0.766044,0]) 


    """===================plotting==================="""

    # raw = "D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/16010_tomobar_cropped_f.npy"
    # raw_model=np.load(raw)
    # for Z in [560,570,580]:
    #     Y=493
    #     width =int(0.240/0.0003/2)
    #     height=int(0.150/0.0003/2)
    #     X=Y # as after rotation, they can be swapped
    #     plot_projection(Y,Z,width,height)

    """===================ploty on 3D==================="""
    for Z in [480,490, 500,510,520,530,540,550,560,570,580]:
        Y=493
        X=Y # as after rotation, they can be swapped

        for angle in [0,30,45,60,90,120,135,150,180,210,225,240,270,300,315,330,360]:
            real_matrix=np.array([[ 1, 0,  0],
            [0,  1,  0],
            [0, 0,  1]]) #k0
            kp='k0'
            test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
            
            real_matrix =np.array([[ 0.61388157,  0.16035909,  0.77293879],
            [-0.46140928,  0.86736106,  0.18651078],
            [-0.64050831, -0.47113666,  0.60644814]]) #km70_pm120
            kp='km70_pm120'
            test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
            

            real_matrix=np.array([[ 0.61388157, -0.32399183,  0.71984631],
            [-0.32399183,  0.72813857,  0.60402277],
            [-0.71984631, -0.60402277,  0.34202014]]) #km70
            kp='km70'
            test_partial_illumination_selection_real('16010_tomobar_cropped_f_0.1.npy',real_matrix,kp,Z=Z,Y=Y,X=X,kp_rotation_axis=kp_rotation_axis,omega_d=angle)
        
            print("angle {} is finished".format(angle))
    # R = np.array([[[0.613882, 0.1067, 0.782154],
    #                  [-0.323992, 0.93758, 0.126386],
    #                  [-0.719846, -0.330997, 0.610133]]])
    # U=np.array([[[-0.5844, -0.2488, -0.7723],
    #                   [-0.7113, -0.3010,  0.6352],
    #                   [-0.3906,  0.9206, -0.0011]]])
    # B=np.array([[[0.0195, 0.0000, 0.0000],
    #                   [0.0068, 0.0200, 0.0000],
    #                   [0.0061, 0.0081, 0.0199]]])
    # A=np.array([[[-0.0178, -0.0112, -0.0154],
    #                   [-0.0121, -0.0009,  0.0126],
    #                   [-0.0014,  0.0184, -0.0000]]])
    # print('A is {}'.format(np.dot(U,B)))
    # print('R*U is {}'.format(np.dot(R,U)))
    # print('R*A is {}'.format(np.dot(R,A)))