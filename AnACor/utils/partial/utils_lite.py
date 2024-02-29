import numpy as np
import pdb
# import matplotlib.pyplot as plt
from numba import jit


def kp_rotation(axis,theta):
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

def partial_illumination_selection(xray_region, total_rotation_matrix, coord, point_on_axis,rotated=False):
        
    y_min_xray, y_max_xray, z_min_xray, z_max_xray = xray_region
    if rotated is False:

        # point_on_axis[2] = coord[2]
        # Define the rotation matrix for rotation around Z-axis
        # Translate the voxel to the origin, rotate it, then translate it back
        translated_coord = np.array(coord) - point_on_axis
        rotated_translated_coord = np.dot(total_rotation_matrix, translated_coord)
        # rotated_coord = np.around(rotated_translated_coord + point_on_axis)
        rotated_coord = rotated_translated_coord + point_on_axis
        # Check if the rotated voxel coordinates are within the x-ray boundaries
        
    else:
        rotated_coord=coord
    # print('rotated_coord is {}'.format(rotated_coord))
    if y_min_xray <= rotated_coord[1] <= y_max_xray and z_min_xray <= rotated_coord[0] <= z_max_xray:

        return True
    else:
        return False



def partial_illumination_selection_v1(xray_region, total_rotation_matrix,coord,point_on_axis):
    y_min_xray,y_max_xray,z_min_xray,z_max_xray = xray_region 
    # noticed that, in numpy, the y-axis is going downwards, so the value of y_min_xray is larger than y_max_xray
    #assumes the rotation axis is the Z axis and the rotation axis goes through the origin
    # point_on_axis=[0,0,0]
    # If your rotation axis is not around the origin, you will need to perform a translation so that your axis of rotation passes through the origin, do the rotation, then translate back.
    corners = np.array([[z_min_xray, y_min_xray, 0],
                        [z_min_xray, y_max_xray, 0],
                        [z_max_xray, y_min_xray, 0],
                        [z_max_xray, y_max_xray, 0]])

    rotated_corners = np.dot(total_rotation_matrix,  (corners - point_on_axis).T).T + point_on_axis
    # Rotate the voxel coordinates
    z_min_rot, y_min_rot, _ = rotated_corners.min(axis=0)
    z_max_rot, y_max_rot, _ = rotated_corners.max(axis=0)
    coord[2]=0
    rotated_p = np.dot(total_rotation_matrix, np.array(coord))
    pdb.set_trace()
    # Check if the rotated voxel coordinates are within the rotated x-ray boundaries
    if z_min_rot <= rotated_p[0] <= z_max_rot and y_min_rot <= rotated_p[1] <= y_max_rot:
        return True
    else:
        return False



# @jit(nopython=True)
def cal_length_mm(path_2,voxel_size):
    # pdb.set_trace()
    voxel_length_z=voxel_size[0]
    voxel_length_y = voxel_size[1]
    voxel_length_x = voxel_size[2]
    path_ray = path_2[0]

    classes = np.array(path_2[1])
    posi = np.array(path_2[2])
    # create a list of index of the boundary between different classes
    
    # boundary = np.insert(classes, 0, classes[0])-np.append(classes, classes[-1])
    boundary = np.concatenate((np.array([classes[0]]),classes)) - \
                np.concatenate((classes, np.array([classes[-1]])))
    boundary_posi = np.where(boundary!=0)[0]-1
    # boundary_posi = np.insert(boundary_posi, 0, 0)
    # boundary_posi = np.append(boundary_posi, len(path_ray)-1)
    boundary_posi = np.concatenate((np.array([0]), boundary_posi))
    boundary_posi = np.concatenate(( boundary_posi,np.array([len(path_ray)-1])))

    cr_l_2 = 0
    lo_l_2 = 0
    li_l_2 = 0
    bu_l_2 = 0
    
        # total_length = ( path_ray[-1][1] - path_ray[0][1] )/ (np.sin(np.abs(omega)))
    total_length=np.sqrt(((path_ray[-1][1]  - path_ray[0][1] ) * voxel_length_y ) ** 2 +
                         ((path_ray[-1][0]  - path_ray[0][0] ) * voxel_length_z ) ** 2 +
                         ( (path_ray[-1][2]  - path_ray[0][2] ) * voxel_length_x )** 2)
    
    boundary_posi = boundary_posi[1:]
    posi_sum=posi.sum()
    for j, lengths in enumerate(posi):
            
            if classes[boundary_posi[j]]==3:
                cr_l_2+= lengths/posi_sum * total_length
            elif classes[boundary_posi[j]]==1:
                li_l_2+= lengths/posi_sum * total_length
            elif classes[boundary_posi[j]]==2:
                lo_l_2+= lengths/posi_sum * total_length
            elif classes[boundary_posi[j]]==4:
                bu_l_2+= lengths/posi_sum * total_length
            else:
                pass
            


    return li_l_2, lo_l_2, cr_l_2,bu_l_2



@jit(nopython=True)
def cal_coord_mm(theta ,phi,coord,face,shape,label_list,max_step=10,full_iteration=False):
    """

    :param theta: the scattering angle in vertical y axis
    :param phi: the scattering angle in horizonal x axis
    :param coord: coordinate of the point in the crystal
    :return: path_2=[ [coordinates],[classes_position],[classes involved]   ]
            coordinates: voxel coordinates of the ray-tracing path
            classes_position: the position in [coordinates] for the boundary between different classes
            classes involved: the classes on the ray-tracing path
    e.g.
    path_2 = [[...],[0, 1, 58], ['cr', 'li', 'lo']], coordinates[0] belongs to 'cr' and coordinates[1:57] belongs to 'lo'
    coordinates[58:] belongs to 'lo'
    """

    #calculate increment of x,y then calculate z accordingly
    # when y or z has one increment ,what would ,x and yor z changes !!!
    # theta = cos-1(F/ distance between detector point and the sample point)
    # phi= sin-1( L difference/2 / distance between detector point and the sample point) in a plane triangle
    z,y,x =coord
    z_max,y_max,x_max=shape

    # path_1 =[(z,y,i) for i in  range(x, x_max, 1)]
    # x_max -= 1
    # y_max -= 1
    # z_max -= 1
    path_2 = [coord]

    classes=[3]
    classes_len=[]
    cls_len=0
    step_size =1
    counter=0
    if face =="BACKZY":
        #   in backface case, there are no need to consider tantheta = inf as when it emits
        # at this face, the theta and phi will be away from 90 degree
        #   Also, no need to discuss the case when theta> 90 degree
        assert np.abs(theta)<= np.pi/2
        increment_ratio_x = -1
        increment_ratio_y = np.tan(theta) / np.cos(phi)
        increment_ratio_z = np.tan(phi)
        # increment/decrement along x axis
        
        position=x
        momentum=True
        while position > 0 :
            step_size =min(step_size,max_step)
            position -=step_size
            cls_len +=step_size
            increment = x-position
        # for increment in range(x-0+1):
            # the absorption also count that coordinate in the  path_2\
            # decrement on x axis
            if theta>0:

                new_x = np.floor(x + increment * increment_ratio_x) # this -1 represents that the opposition of direction
                                                 # between the lab x-axis and the wavevector
                new_y = np.floor(y - increment * increment_ratio_y)

                new_z = np.floor(z + increment * increment_ratio_z)
            else:
                new_x = np.round(
                    x + increment * increment_ratio_x)  # this -1 represents that the opposition of direction
                # between the lab x-axis and the wavevector
                new_y = np.round(y - increment * increment_ratio_y)

                new_z = np.round(z + increment * increment_ratio_z)
            
            if new_y >= y_max:
                new_y = y_max - 1
            elif new_y < 0:
                new_y = 0

            if new_x >= x_max:
                new_x = x_max - 1
            elif new_x < 0:
                new_x = 0

            if new_z >= z_max:
                new_z = z_max - 1
            elif new_z < 0:
                new_z = 0

            potential_coord = (int(  (new_z)),
                                int(  (new_y)),
                                int(  (new_x)))
            _cls = label_list[potential_coord]



            if _cls != classes[counter]:
                # pdb.set_trace()
                if momentum:
                    position += step_size
                    cls_len -= step_size
                    # go back to the next one of the previous position
                    step_size = 1
                    # keep the step size as 1 until it meets the boundary
                    momentum=False
                    continue
                else:

                    classes_len.append(cls_len)
                    step_size = 1
                    cls_len = 0
                    momentum=True
            else:
                if momentum:
                    step_size += 1

            if full_iteration is False :
                if _cls == 0:
                    # classes_len.append(cls_len-step_size)
                    break


             
            path_2.append(np.array(potential_coord))
            classes.append(_cls)
            counter += 1
    elif face == 'LEYX':
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
            increment_ratio_z = -1


            position=z
            momentum=True
            while position > 0 :
                step_size =min(step_size,max_step)
                position -=step_size
                cls_len +=step_size
                increment = z-position
            # for increment in range(z-0+1):
                # decrement on z-axis
                # new_x = x + -1 * increment/(np.tan(np.abs(phi)))
                # new_y = y - increment * np.tan(theta) / np.sin(np.abs(phi))
                # new_z = z + increment*-1
                if theta > 0:
                    new_x = np.floor(x + -1 * increment * increment_ratio_x)
                    new_y = np.floor(y - increment * increment_ratio_y)
                    new_z = np.floor(z + increment * increment_ratio_z)
                else:
                    new_x = np.round(x + -1 * increment * increment_ratio_x)
                    new_y = np.round(y - increment * increment_ratio_y)
                    new_z = np.round(z + increment * increment_ratio_z)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position += step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:
                        
                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1
        else:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi-theta) /   np.sin(np.abs(phi))
            increment_ratio_z = -1


            position=z
            momentum=True
            while position > 0 :
                step_size =min(step_size,max_step)
                position -=step_size
                cls_len +=step_size
                increment = z-position
            # for increment in range(z - 0 + 1):
                # decrement on z-axis
                # new_x = x + 1 * increment / (np.tan(np.abs(phi)) )
                # new_y = y - increment * np.tan(np.pi-theta) /   np.sin(np.abs(phi))
                # new_z = z + increment * -1
                if theta > 0:
                    new_x = np.floor(x + 1 * increment * increment_ratio_x)
                    new_y = np.floor(y - increment * increment_ratio_y)
                    new_z = np.floor(z + increment * increment_ratio_z)
                else:
                    new_x = np.round( x + 1 * increment * increment_ratio_x)
                    new_y = np.round(y - increment * increment_ratio_y)
                    new_z = np.round(z + increment * increment_ratio_z)
                
                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    if momentum:
                        position += step_size 
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:
                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break


                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1

    elif face == 'RIYX':

        if np.abs(theta) < np.pi/2:
            increment_ratio_x = 1 / ( np.tan(np.abs(phi)) )
            increment_ratio_y = np.tan(theta)/  np.sin(np.abs(phi))
            increment_ratio_z = 1

            position=z
            momentum=True
            while position < z_max :
                step_size =min(step_size,max_step)
                position +=step_size
                cls_len +=step_size
                increment = position-z

            # for increment in range(z_max-z):
                # increment on z-axis
                # new_x = x + -1 * increment/ ( np.tan(np.abs(phi)) )
                # new_y = y - increment*np.tan(theta)/  np.sin(np.abs(phi))
                # new_z = z + increment*1
                if theta > 0:
                    new_x = np.floor(x + -1 * increment * increment_ratio_x)
                    new_y = np.floor(y - increment * increment_ratio_y)
                    new_z = np.floor(z + increment * 1)
                else:
                    new_x = np.round(x + -1 * increment * increment_ratio_x)
                    new_y = np.round(y - increment* increment_ratio_y)
                    new_z = np.round(z + increment* 1)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position -= step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:

                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

             
                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1
        else:
            increment_ratio_x = 1 / ( np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi-theta)/ np.sin(np.abs(phi))
            increment_ratio_z = 1


            position=z
            momentum=True
            while position < z_max :
                step_size =min(step_size,max_step)
                position +=step_size
                cls_len +=step_size
                increment = position-z
            # for increment in range(z_max-z+1):
                # increment on z-axis
                # new_x = x + 1 * increment/ ( np.tan(np.abs(phi)))
                # new_y = y - increment*np.tan(np.pi-theta)/ np.sin(np.abs(phi))
                # new_z = z + increment*1
                if theta > 0:
                    new_x = np.floor(x + 1 * increment * increment_ratio_x)
                    new_y = np.floor(y - increment * increment_ratio_y)
                    new_z = np.floor(z + increment*1)
                else:
                    new_x = np.round(x + 1 * increment * increment_ratio_x)
                    new_y = np.round(y - increment * increment_ratio_y)
                    new_z = np.round(z + increment*1)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position -= step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:

                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

             

                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1

    elif face == 'BOTZX':
        assert theta < 0
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = np.cos(np.abs(phi))/(np.tan(np.abs(theta)))
            increment_ratio_y = -1
            increment_ratio_z = np.sin(phi)/ ( np.tan(np.abs(theta)) )
            
            position=y
            momentum=True
            while position < y_max :
                step_size =min(step_size,max_step)
                position +=step_size
                cls_len +=step_size
                increment = position-y            
            
            # for increment in range(y_max-y):
                # decrement on y-axis
                # new_x = x + -1 * increment * np.cos(np.abs(phi))/(np.tan(np.abs(theta)))
                # new_y = y - increment*-1
                # new_z = z + increment*np.sin(phi)/ ( np.tan(np.abs(theta)) )
                new_x = np.round(x + -1 * increment * increment_ratio_x)
                new_y = np.round(y - increment * increment_ratio_y)
                new_z = np.round(z + increment * increment_ratio_z)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position -= step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:

                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

       

                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1

        else:
            increment_ratio_x = np.cos(np.abs(phi)) / ( np.tan(np.pi-np.abs(theta)) )
            increment_ratio_y = -1
            increment_ratio_z = np.sin(phi) / ( np.tan(np.pi-np.abs(theta)) )
            
            position=y
            momentum=True
            while position < y_max :
                step_size =min(step_size,max_step)
                position +=step_size
                cls_len +=step_size
                increment = position-y
            # for increment in range(y_max - y + 1):
                # decrement on y-axis
                # new_x = x + 1 * increment * np.cos(np.abs(phi)) / ( np.tan(np.abs(np.pi-theta)) )
                # new_y = y - increment * -1
                # new_z = z - increment * np.sin(phi) / ( np.tan(np.abs(np.pi-theta)) ) #
                new_x =np.round( x + 1 * increment * increment_ratio_x)
                new_y =np.round(  y - increment * increment_ratio_y)
                new_z =np.round(  z - increment * increment_ratio_z) #

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position -= step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:


                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1

    elif face == 'TOPZX':
        assert theta > 0
        # 
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = np.cos(np.abs(phi))/ ( np.tan(np.abs(theta)))
            increment_ratio_y = 1
            increment_ratio_z = np.sin(phi)/ ( np.tan(np.abs(theta)))

            position=y
            momentum=True
            while position > 0 :
                step_size =min(step_size,max_step)
                position -=step_size
                cls_len +=step_size
                increment = y-position
            # for increment in range(y-0+1):
                # decrement on y-axis
                # new_x = x + -1 * increment * np.cos(np.abs(phi))/ ( np.tan(np.abs(theta)))
                # new_y = y - increment*1
                # new_z = z + increment * np.sin(phi)/ ( np.tan(np.abs(theta)))
                new_x = np.floor(x + -1 * increment * increment_ratio_x)
                new_y = np.floor(y - increment*increment_ratio_y)
                new_z = np.floor(z + increment * increment_ratio_z)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]



                if _cls != classes[counter]:
                    # pdb.set_trace()
                    if momentum:
                        position += step_size
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:

                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break

         
                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1
        else:
            increment_ratio_x = np.cos(np.abs(phi))/ ( np.tan((np.pi-np.abs(theta))) )
            increment_ratio_y = 1
            increment_ratio_z = np.sin(-phi) / ( np.tan((np.pi-np.abs(theta))) )

            position=y
            momentum=True
            while position > 0 :
                step_size =min(step_size,max_step)
                position -=step_size
                cls_len +=step_size
                increment = y-position

                # for increment in range(y - 0 + 1):
                # decrement on y-axis
                # new_x = x + 1 * increment * np.cos(np.abs(phi))/ ( np.tan(np.abs(np.pi-theta)) )
                # new_y = y - increment * 1
                # new_z = z + increment* np.sin(phi) / ( np.tan(np.abs(theta)))
                new_x = np.floor(x + 1 * increment * increment_ratio_x)
                new_y = np.floor(y - increment * 1)
                new_z = np.floor(z + increment* increment_ratio_z)

                if new_y >= y_max:
                    new_y = y_max - 1
                elif new_y < 0:
                    new_y = 0

                if new_x >= x_max:
                    new_x = x_max - 1
                elif new_x < 0:
                    new_x = 0

                if new_z >= z_max:
                    new_z = z_max - 1
                elif new_z < 0:
                    new_z = 0

                potential_coord = (int(  (new_z)),
                                   int(  (new_y)),
                                   int(  (new_x)))
                _cls = label_list[potential_coord]


                # pdb.set_trace()
                if _cls != classes[counter]:
                    
                    if momentum:
                        position += step_size 
                        cls_len -= step_size
                        # go back to the next one of the previous position
                        step_size = 1
                        # keep the step size as 1 until it meets the boundary
                        momentum=False
                        continue
                    else:

                        classes_len.append(cls_len)
                        step_size = 1
                        cls_len = 0
                        momentum=True
                else:
                    if momentum:
                        step_size += 1

                if full_iteration is False :
                    if _cls == 0:
                        # classes_len.append(cls_len-step_size)
                        break


                 
                path_2.append(np.array(potential_coord))
                classes.append(_cls)
                counter += 1
        
    elif face == 'FRONTZY':
        #   in frontface case, there are no need to consider tantheta = inf as when it emits
        # at this face, the theta and phi will be away from 90 degree
        #   Also, no need to discuss the case when theta< 90 degree

        assert np.abs(theta) > np.pi/2
        increment_ratio_x = -1
        increment_ratio_y = np.tan(np.pi-theta) / np.cos(np.abs(phi))
        increment_ratio_z = np.tan(phi)


        position=x
        momentum=True
        while position < x_max :
            step_size =min(step_size,max_step)
            position +=step_size
            cls_len +=step_size
            increment = position-x
        # for increment in range(x_max-x):
            # the absorption also count that coordinate in the  path_2\
            # decrement on x axis
            if theta > 0:
                new_x = np.floor(
                    x - increment * increment_ratio_x)  # this -1 represents that the opposition of direction
                # between the lab x-axis and the wavevector
                new_y = np.floor(y - increment * increment_ratio_y)
                new_z = np.floor(z - increment * increment_ratio_z)
            else:
                new_x = np.round(x - increment * increment_ratio_x) # this -1 represents that the opposition of direction
                                                 # between the lab x-axis and the wavevector
                new_y = np.round(y - increment * increment_ratio_y)
                new_z = np.round(z - increment * increment_ratio_z)

            if new_y >= y_max:
                new_y = y_max - 1
            elif new_y < 0:
                new_y = 0

            if new_x >= x_max:
                new_x = x_max - 1
            elif new_x < 0:
                new_x = 0

            if new_z >= z_max:
                new_z = z_max - 1
            elif new_z < 0:
                new_z = 0

            potential_coord = (int(  (new_z)),
                                int(  (new_y)),
                                int(  (new_x)))
            _cls = label_list[potential_coord]



            if _cls != classes[counter]:
                # pdb.set_trace()
                if momentum:
                    position -= step_size
                    cls_len -= step_size
                    # go back to the next one of the previous position
                    step_size = 1
                    # keep the step size as 1 until it meets the boundary
                    momentum=False
                    continue
                else:

                    classes_len.append(cls_len)
                    step_size = 1
                    cls_len = 0
                    momentum=True
            else:
                if momentum:
                    step_size += 1

            if full_iteration is False :
                if _cls == 0:
                    # classes_len.append(cls_len-step_size)
                    break


            path_2.append(np.array(potential_coord))
            classes.append(_cls)
            counter += 1

    else:
        raise RuntimeError("unexpected ray out face")

    if full_iteration:
        classes_len.append(cls_len)
    return path_2,classes,classes_len



def top_increment_ratio(theta,phi):
    assert theta > 0
    if np.abs(theta) < np.pi / 2:
        increment_ratio_x = -np.cos(np.abs(phi)) / (np.tan(np.abs(theta)))
        increment_ratio_y = 1
        increment_ratio_z = np.sin(phi) / (np.tan(np.abs(theta)))
    else:
        increment_ratio_x = np.cos(np.abs(phi)) / (np.tan((np.pi - np.abs(theta))))
        increment_ratio_y = 1
        increment_ratio_z = np.sin(-phi) / (np.tan((np.pi - np.abs(theta))))

    increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios

def back_increment_ratio(theta,phi):
    assert np.abs(theta) < np.pi / 2
    increment_ratio_x = -1
    increment_ratio_y = np.tan(theta) / np.cos(phi)
    increment_ratio_z = np.tan(phi)

    increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios


def left_increment_ratio(theta,phi):
    if np.abs(theta) < np.pi / 2:
        increment_ratio_x = -1 / (np.tan(np.abs(phi)))
        increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
        increment_ratio_z = -1

    else:
        increment_ratio_x = 1 / (np.tan(np.abs(phi)))
        increment_ratio_y = np.tan(np.pi - theta) / np.sin(np.abs(phi))
        increment_ratio_z = -1


    increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios


def right_increment_ratio(theta,phi):
    if np.abs(theta) < np.pi / 2:
        increment_ratio_x = -1 / (np.tan(np.abs(phi)))
        increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
        increment_ratio_z = 1

    else:
        increment_ratio_x = 1 / (np.tan(np.abs(phi)))
        increment_ratio_y = np.tan(np.pi - theta) / np.sin(np.abs(phi))
        increment_ratio_z = 1

    increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios


def front_increment_ratio(theta,phi):

    increment_ratio_x = 1
    increment_ratio_y = np.tan(np.pi - theta) / np.cos(np.abs(phi))
    increment_ratio_z = -np.tan(phi)

    increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios


def bottom_increment_ratio(theta, phi):
    assert theta < 0
    if np.abs(theta) < np.pi / 2:
        increment_ratio_x = -np.cos(np.abs(phi)) / (np.tan(np.abs(theta)))
        increment_ratio_y = -1
        increment_ratio_z = np.sin(phi) / (np.tan(np.abs(theta)))

    else:
        increment_ratio_x = np.cos(np.abs(phi)) / (np.tan(np.pi - np.abs(theta)))
        increment_ratio_y = -1
        increment_ratio_z = -np.sin(phi) / (np.tan(np.pi - np.abs(theta)))

    increment_ratios = increment_ratio_z, increment_ratio_y, increment_ratio_x

    return increment_ratios

@jit(nopython=True)
def difference_length(start,end,voxel_size):
        z1,y1,x1=start
        z2,y2,x2=end
        z_voxel_size, y_voxel_size, x_voxel_size = voxel_size
        length = np.sqrt( ((x2-x1+0.5)*x_voxel_size ) ** 2 +
                          ((z2-z1+0.5)*z_voxel_size) ** 2 + 
                          ((y2-y1+0.5)*y_voxel_size) ** 2)

        return length


# @jit(nopython=True)
def cal_path2_bisection(path_2,voxel_size):

    # pdb.set_trace()
    path_ray = path_2[0]
    classes = path_2[1]
    # z_voxel_size, y_voxel_size, x_voxel_size = voxel_size
    total_LineLength = difference_length( np.array(path_ray[1]), np.array(path_ray[0]),voxel_size )
    # Pythagorean theorem
    cr_l_2 =  difference_length( np.array(path_ray[2]) ,np.array(path_ray[0]),voxel_size )

    lo_l_2 = 0
    air_l_2 = 0
    bu_l_2 = 0

    for j, cls in enumerate(classes):
        if j < 3:
            # first 3 are definitely coord, air , crystal_outer
            continue

        if 'lo' in cls:
            if 'inner' in cls:
                continue
            lo_l_2 += difference_length(np.array(path_ray[j]),np.array(path_ray[j-1]),voxel_size)
        elif 'bu' in cls:
            if 'inner' in cls:
                continue
            bu_l_2 += difference_length(np.array(path_ray[j]),np.array(path_ray[j-1]),voxel_size)
        elif 'air' in cls:
            if 'inner' in cls:
                continue
            air_l_2 += difference_length(np.array(path_ray[j]) , np.array(path_ray[j-1]),voxel_size)
        else:
            print('\n')
            print('ERROR : undefined classes')
            print('\n')
    li_l_2  = total_LineLength - lo_l_2 - bu_l_2 - cr_l_2 - air_l_2
    # can add the other class path
    return li_l_2, lo_l_2, cr_l_2,bu_l_2




@jit(nopython=True)
def bisection(counter, CrystalLongest, CrystalShortest, resolution, label_list, increment_ratios, coord, boundary, cls):

    crystalDifference = CrystalLongest - CrystalShortest
    increment_ratio_z, increment_ratio_y, increment_ratio_x  = increment_ratios
    z, y, x = coord
    CrystalMiddle = (CrystalLongest + CrystalShortest) / 2
    z_max, y_max, x_max = label_list.shape
    x_max -= 1
    y_max -= 1
    z_max -= 1

    while crystalDifference > resolution:
        counter += 1
        CrystalMiddle = (CrystalLongest + CrystalShortest) / 2

        new_x = np.floor(
            x + CrystalMiddle  * increment_ratio_x)  # this -1 represents that the opposition of direction
        # between the lab x-axis and the wavevector
        new_y = np.floor(y - CrystalMiddle * increment_ratio_y)

        new_z = np.floor(z + CrystalMiddle * increment_ratio_z)

        # if new_y >= y_max:
        #     new_y = y_max - 1
        # elif new_y < 0:
        #     new_y = 0

        # if new_x >= x_max:
        #     new_x = x_max - 1
        # elif new_x < 0:
        #     new_x = 0

        # if new_z >= z_max:
        #     new_z = z_max - 1
        # elif new_z < 0:
        #     new_z = 0

        potential_coord = (int((new_z)),
                           int((new_y)),
                           int((new_x)))

        if boundary == 'inner':
            # calculate the inner boundary of the class closer to the central, starting from the centre
            # from the centre approaching to the outer if searching for specific class
            if label_list[potential_coord] == cls:
                CrystalLongest = CrystalMiddle
            else:
                CrystalShortest = CrystalMiddle
        else:  # calculate the outer boundary of the class closer to the central, starting from the outer shell
            # from outer approaching to the centre if searching for specific class
            if label_list[potential_coord] == cls:
                CrystalShortest = CrystalMiddle
            else:
                CrystalLongest = CrystalMiddle
        crystalDifference = CrystalLongest - CrystalShortest


    # new_x = np.floor(x + (CrystalMiddle-1) * increment_ratio_x)  # this -1 represents that the opposition of direction
    # # between the lab x-axis and the wavevector
    # new_y = np.floor(y - (CrystalMiddle-1)  * increment_ratio_y)
    #
    # new_z = np.floor(z + (CrystalMiddle-1)  * increment_ratio_z)

    new_x = np.floor(x + np.floor(CrystalMiddle) * increment_ratio_x)  # this -1 represents that the opposition of direction
    # between the lab x-axis and the wavevector
    new_y = np.floor(y - np.floor(CrystalMiddle)  * increment_ratio_y)

    new_z = np.floor(z + np.floor(CrystalMiddle)  * increment_ratio_z)
    potential_coord = (int((new_z)),
                       int((new_y)),
                       int((new_x)) )


    return potential_coord, CrystalMiddle,counter

def average(potential_coord,air_outermost_potential_coord):
        difference = np.abs(np.array(potential_coord) - np.array(air_outermost_potential_coord))
        ave = (difference[0] + difference[1] + difference[2] ) /3
        return ave


@jit(nopython=True)
def increments(face, theta, phi, z_max, y_max, x_max,z,y,x):
    if face == 'TOPZX':

        # increment_ratios = top_increment_ratio(theta, phi)
        assert theta > 0
        if np.abs(theta) < np.pi / 2:
            increment_ratio_x = -np.cos(np.abs(phi)) / (np.tan(np.abs(theta)))
            increment_ratio_y = 1
            increment_ratio_z = np.sin(phi) / (np.tan(np.abs(theta)))
        else:
            increment_ratio_x = np.cos(np.abs(phi)) / (np.tan((np.pi - np.abs(theta))))
            increment_ratio_y = 1
            increment_ratio_z = np.sin(-phi) / (np.tan((np.pi - np.abs(theta))))

        increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = y

    elif face == "BOTZX":
        # increment_ratios = bottom_increment_ratio(theta, phi)

        assert theta < 0
        if np.abs(theta) < np.pi / 2:
            increment_ratio_x = -np.cos(np.abs(phi)) / (np.tan(np.abs(theta)))
            increment_ratio_y = -1
            increment_ratio_z = np.sin(phi) / (np.tan(np.abs(theta)))

        else:
            increment_ratio_x = np.cos(np.abs(phi)) / (np.tan(np.pi - np.abs(theta)))
            increment_ratio_y = -1
            increment_ratio_z = -np.sin(phi) / (np.tan(np.pi - np.abs(theta)))

        increment_ratios = increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = y_max-y

    elif face == "BACKZY":
        # increment_ratios = back_increment_ratio(theta, phi)
        assert np.abs(theta) < np.pi / 2
        increment_ratio_x = -1
        increment_ratio_y = np.tan(theta) / np.cos(phi)
        increment_ratio_z = np.tan(phi)

        increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = x

    elif face == "FRONTZY":
        # increment_ratios =front_increment_ratio(theta, phi)
        increment_ratio_x = 1
        increment_ratio_y = np.tan(np.pi - theta) / np.cos(np.abs(phi))
        increment_ratio_z = -np.tan(phi)

        increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = x_max-x

    elif face == "LEYX":
        # increment_ratios = left_increment_ratio(theta, phi)
        if np.abs(theta) < np.pi / 2:
            increment_ratio_x = -1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
            increment_ratio_z = -1

        else:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi - theta) / np.sin(np.abs(phi))
            increment_ratio_z = -1


        increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = z

    elif face == "RIYX":
        # increment_ratios = right_increment_ratio(theta, phi)
        if np.abs(theta) < np.pi / 2:
            increment_ratio_x = -1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
            increment_ratio_z = 1

        else:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi - theta) / np.sin(np.abs(phi))
            increment_ratio_z = 1

        increment_ratios =  increment_ratio_z, increment_ratio_y, increment_ratio_x
        AirLongest = z_max - z

    else:
        raise RuntimeError("unexpected ray out face")

    return increment_ratios, AirLongest

# @jit(nopython=True)
def iterative_bisection(theta, phi, coord, face, label_list):
    """
    :param theta:
    :param phi:
    :param coord:
    :param face:
    :param label_list:
    :param reference_path:
    :return:  path_2=[ [[outer boudary of sample],[outer boudary of crytal ],[other boundary_coordinates],...],[air, cr, other, classes, ... ]
    """

    counter=0
    resolution = 1
    #reference_path_ray, reference_posi, reference_classes = reference_path
    z_max, y_max, x_max = label_list.shape
    z, y, x = coord
    path_2 = []
    # classes_posi = [0]
    classes = ['coord']
    # a,b,c=coord
    # crd=(a,b,c)
    path_2.append(coord)
    # path_1 =[(z,y,i) for i in  range(x, x_max, 1)]
    x_max -= 1
    y_max -= 1
    z_max -= 1
    AirShortest = 0
    increment_ratios, AirLongest = increments(face, theta, phi, z_max, y_max, x_max, z, y, x)

    air_outermost_potential_coord, AirMiddle_outer,counter = bisection(counter, AirLongest, AirShortest, resolution, label_list,
                                                                   increment_ratios, coord,boundary='inner', cls=0)
    classes.append('air_outermost')
    path_2.append(air_outermost_potential_coord)

    # finding the boundary between outer boudary of the crystal
    CrystalLongest = AirMiddle_outer
    CrystalShortest = 0
    cr_outer_potential_coord, CrystalMiddle,counter = bisection(counter, CrystalLongest, CrystalShortest, resolution, label_list,
                                                            increment_ratios, coord, boundary='outer', cls=3)
    classes.append('cr_outer')
    path_2.append(cr_outer_potential_coord)
    # pdb.set_trace()
    # starting from the crystal to calculate the classes along the path to find the LOOP
    LoopLongest = AirMiddle_outer
    LoopShortest = CrystalMiddle
    potential_coord, LoopMiddle,counter = bisection(counter, LoopLongest, LoopShortest, resolution, label_list,
                                                increment_ratios, coord, boundary='inner', cls=2)
    
    if np.abs(np.array(potential_coord)-np.array(air_outermost_potential_coord)).mean() < 1:
        pass
    else:
        classes.append('lo_inner')
        path_2.append(np.array(potential_coord))
        LoopLongest = AirMiddle_outer
        LoopShortest = LoopMiddle
        potential_coord, LoopMiddle,counter = bisection(counter, LoopLongest, LoopShortest, resolution, label_list,
                                                    increment_ratios, coord, boundary='outer', cls=2)
        classes.append('lo_outer')
        path_2.append(np.array(potential_coord))

    # starting from the crystal to calculate the classes along the path to find the BUBBLE
    BubbleLongest = AirMiddle_outer
    BubbleShortest = CrystalMiddle
    potential_coord, BubbleMiddle,counter = bisection(counter, BubbleLongest, BubbleShortest, resolution, label_list,
                                                  increment_ratios, coord, boundary='inner', cls=4)
    if np.abs(np.array(potential_coord)-np.array(air_outermost_potential_coord)).mean() < 1:
        pass
    else:
        classes.append('bu_inner')
        path_2.append(np.array(potential_coord))
        BubbleLongest = AirMiddle_outer
        BubbleShortest = BubbleMiddle
        potential_coord, BubbleMiddle,counter = bisection(counter, BubbleLongest, BubbleShortest, resolution, label_list,
                                                      increment_ratios, coord, boundary='outer', cls=4)
        classes.append('bu_outer')
        path_2.append(np.array(potential_coord))

    # # starting from the crystal to calculate the classes along the path to find the other possible air
    # Air2Longest = AirMiddle_outer
    # Air2Shortest = CrystalMiddle
    # potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
    #                                             increment_ratios, coord, boundary='inner', cls=0)
    # if average(potential_coord,air_outermost_potential_coord)< 1:
    #     pass
    # else:
    #     classes.append('air_inner')
    #     path_2.append(np.array(potential_coord))
    #     Air2Longest = AirMiddle_outer
    #     Air2Shortest = Air2Middle
    #     potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
    #                                                 increment_ratios, coord, boundary='outer', cls=0)
    #     classes.append('air_outer')
    #     path_2.append(np.array(potential_coord))


    return (path_2, classes),counter


