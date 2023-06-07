"""
This is some core algorithms that are needed for numba to jit the code
and this is will be replaced by Ctype python code
"""

from numba import jit
import numpy as np



@jit(nopython=True)
def cal_coord_2(theta ,phi,coord,face,shape,label_list,full_iteration=False):
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
    path_2 = []
    num_classes = 1
    classes_posi = [0]
    classes=['cr']

    if face =="BACKZY":
        #   in backface case, there are no need to consider tantheta = inf as when it emits
        # at this face, the theta and phi will be away from 90 degree
        #   Also, no need to discuss the case when theta> 90 degree
        assert np.abs(theta)<= np.pi/2
        increment_ratio_x = -1
        increment_ratio_y = np.tan(theta) / np.cos(phi)
        increment_ratio_z = np.tan(phi)
        # increment/decrement along x axis
        for increment in range(x-0+1):
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
            if full_iteration is False:
                if label_list[potential_coord]==0:
                    break

            if increment == 0:
                pass
            elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                if label_list[potential_coord] == 1:
                    classes.append('li')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 2:
                    classes.append('lo')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 3:
                    classes.append('cr')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 4:
                    classes.append('bu')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 0:
                    classes.append('va')
                    classes_posi.append(increment)
                else:
                    raise RuntimeError('unexpected classes')

            path_2.append(potential_coord)

    elif face == 'LEYX':
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(theta) / np.sin(np.abs(phi))
            increment_ratio_z = -1
            for increment in range(z-0+1):
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

                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)
        else:
            increment_ratio_x = 1 / (np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi-theta) /   np.sin(np.abs(phi))
            increment_ratio_z = -1
            for increment in range(z - 0 + 1):
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
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)

    elif face == 'RIYX':

        if np.abs(theta) < np.pi/2:
            increment_ratio_x = 1 / ( np.tan(np.abs(phi)) )
            increment_ratio_y = np.tan(theta)/  np.sin(np.abs(phi))
            increment_ratio_z = 1
            for increment in range(z_max-z):
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
                # pdb.set_trace()
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)
        else:
            increment_ratio_x = 1 / ( np.tan(np.abs(phi)))
            increment_ratio_y = np.tan(np.pi-theta)/ np.sin(np.abs(phi))
            increment_ratio_z = 1
            for increment in range(z_max-z+1):
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
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)

    elif face == 'BOTZX':
        assert theta < 0
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = np.cos(np.abs(phi))/(np.tan(np.abs(theta)))
            increment_ratio_y = -1
            increment_ratio_z = np.sin(phi)/ ( np.tan(np.abs(theta)) )
            for increment in range(y_max-y):
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
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)

        else:
            increment_ratio_x = np.cos(np.abs(phi)) / ( np.tan(np.pi-np.abs(theta)) )
            increment_ratio_y = -1
            increment_ratio_z = np.sin(phi) / ( np.tan(np.pi-np.abs(theta)) )
            for increment in range(y_max - y + 1):
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

                potential_coord = (int(new_z),
                                   int(new_y),
                                   int(new_x))
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)

    elif face == 'TOPZX':
        assert theta > 0
        if np.abs(theta) < np.pi/2:
            increment_ratio_x = np.cos(np.abs(phi))/ ( np.tan(np.abs(theta)))
            increment_ratio_y = 1
            increment_ratio_z = np.sin(phi)/ ( np.tan(np.abs(theta)))

            for increment in range(y-0+1):
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
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)
        else:
            increment_ratio_x = np.cos(np.abs(phi))/ ( np.tan((np.pi-np.abs(theta))) )
            increment_ratio_y = 1
            increment_ratio_z = np.sin(-phi) / ( np.tan((np.pi-np.abs(theta))) )

            for increment in range(y - 0 + 1):
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
                # print(potential_coord)
                # print(label_list[potential_coord] )
                # if label_list[potential_coord]  !=0:
                #
                #     pdb.set_trace()
                if full_iteration is False :
                    if label_list[potential_coord] == 0:
                        break

                if increment == 0:
                    pass
                elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                    if label_list[potential_coord] == 1:
                        classes.append('li')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 2:
                        classes.append('lo')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 3:
                        classes.append('cr')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 4:
                        classes.append('bu')
                        classes_posi.append(increment)
                    elif label_list[potential_coord] == 0:
                        classes.append('va')
                        classes_posi.append(increment)
                    else:
                        raise RuntimeError('unexpected classes')

                path_2.append(potential_coord)

    elif face == 'FRONTZY':
        #   in frontface case, there are no need to consider tantheta = inf as when it emits
        # at this face, the theta and phi will be away from 90 degree
        #   Also, no need to discuss the case when theta< 90 degree

        assert np.abs(theta) > np.pi/2
        increment_ratio_x = -1
        increment_ratio_y = np.tan(np.pi-theta) / np.cos(np.abs(phi))
        increment_ratio_z = np.tan(phi)

        for increment in range(x_max-x):
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

            if full_iteration is False :
                if label_list[potential_coord] == 0 :
                    break

            if increment == 0:
                pass
            elif label_list[potential_coord] != label_list[path_2[increment - 1]]:
                if label_list[potential_coord] == 1:
                    classes.append('li')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 2:
                    classes.append('lo')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 3:
                    classes.append('cr')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 4:
                    classes.append('bu')
                    classes_posi.append(increment)
                elif label_list[potential_coord] == 0:
                    classes.append('va')
                    classes_posi.append(increment)
                else:
                    raise RuntimeError('unexpected classes')

            path_2.append(potential_coord)

    else:
        raise RuntimeError("unexpected ray out face")


    return path_2,classes_posi,classes

''' iterative bisection method to deterine the lengths even quicker 
'''



def norm_2_length(array):
    z,y,x=np.abs(array)
    length =  np.sqrt( (x+0.5) ** 2 + (z+0.5) ** 2 + (y+0.5) ** 2)
    return length


def cal_path2_bisection(path_2):
    # pdb.set_trace()
    path_ray = path_2[0]
    classes = path_2[1]
    total_LineLength = norm_2_length( np.array(path_ray[1]) - np.array(path_ray[0]) )
    # Pythagorean theorem
    cr_l_2 =  norm_2_length( np.array(path_ray[2]) - np.array(path_ray[0]) )

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
            lo_l_2 += norm_2_length(np.array(path_ray[j]) - np.array(path_ray[j-1]))
        elif 'bu' in cls:
            if 'inner' in cls:
                continue
            bu_l_2 += norm_2_length(np.array(path_ray[j]) - np.array(path_ray[j-1]))
        elif 'air' in cls:
            if 'inner' in cls:
                continue
            air_l_2 += norm_2_length(np.array(path_ray[j]) - np.array(path_ray[j-1]))
        else:
            print('\n')
            print('ERROR : undefined classes')
            print('\n')
    li_l_2  = total_LineLength - lo_l_2 - bu_l_2 - cr_l_2 - air_l_2
    # can add the other class path
    return li_l_2, lo_l_2, cr_l_2,bu_l_2


def cal_path2_plus22(path_2, coord, omega):
    # pdb.set_trace()
    path_ray = path_2[0]
    posi = path_2[1]
    classes = path_2[2]


    cr_l_2 = 0
    lo_l_2 = 0
    li_l_2 = 0
    bu_l_2 = 0
    for j, index in enumerate(posi):
        # pdb.set_trace()
        if  classes[j] == 'cr':
            if j < len(posi) - 1:
                x_cr = np.abs(path_ray[posi[j + 1] ][2] - path_ray[posi[j]][2])
                z_cr = np.abs(path_ray[posi[j + 1] ][0] - path_ray[posi[j] ][0])
                y_cr = np.abs(path_ray[posi[j + 1] ][1] - path_ray[posi[j] ][1])
            else:
                x_cr = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
                z_cr = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
                y_cr = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
            cr_l_2_total = np.sqrt( (x_cr+0.5) ** 2 + (z_cr+0.5) ** 2 + (y_cr+0.5) ** 2)
            cr_l_2 += cr_l_2_total

        elif classes[j] == 'li':
            if j < len(posi) - 1:
                x_li = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
                z_li = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
                y_li = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
            else:
                x_li = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
                z_li = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
                y_li = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
            li_l_2_total = np.sqrt(x_li ** 2 + z_li ** 2 + y_li ** 2)
            li_l_2 += li_l_2_total

        elif classes[j] == 'lo':
            if j < len(posi) - 1:
                x_lo = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
                z_lo = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
                y_lo = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
            else:
                x_lo = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
                z_lo = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
                y_lo = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
            lo_l_2_total = np.sqrt(x_lo ** 2 + z_lo ** 2 + y_lo ** 2)
            lo_l_2 += lo_l_2_total
        elif classes[j] == 'bu':
            if j < len(posi) - 1:
                x_bu = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
                z_bu = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
                y_bu = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
            else:
                x_bu = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
                z_bu = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
                y_bu = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
            bu_l_2_total = np.sqrt(x_bu ** 2 + z_bu ** 2 + y_bu ** 2)
            bu_l_2 += bu_l_2_total

    # can add the other class path
    return li_l_2, lo_l_2, cr_l_2,bu_l_2


def cal_num22(coord,path_1,path_2,theta,rotation_frame_angle,plus=True):
# @jit (nopython=True)
    """

    :param label_list:
    :param path_1:
    :param path_2:
    :param pixel_size:
    :param coe:
    :param rate_list:  {'li': 1, 'lo': 2, "cr": 3}
    :param jit:
    :return:
    """
    # li_l_2,lo_l_2,cr_l_2,bu_l_2=cal_path2_plus22(path_2,coord,theta)
    li_l_2,lo_l_2,cr_l_2,bu_l_2=cal_path2_bisection(path_2)
    if path_1 is not None:
        li_l_1, lo_l_1, cr_l_1,bu_l_1 = cal_path2_bisection(path_1)
        return li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2
    else:
        return li_l_2, lo_l_2, cr_l_2, bu_l_2

@jit(nopython=True)
def bisection(CrystalLongest, CrystalShortest, resolution, label_list, increment_ratios, coord, boundary, cls):

    crystalDifference = CrystalLongest - CrystalShortest
    increment_ratio_z, increment_ratio_y, increment_ratio_x  = increment_ratios
    z, y, x = coord
    CrystalMiddle = (CrystalLongest + CrystalShortest) / 2
    z_max, y_max, x_max = label_list.shape
    x_max -= 1
    y_max -= 1
    z_max -= 1

    while crystalDifference > resolution:

        CrystalMiddle = (CrystalLongest + CrystalShortest) / 2

        new_x = np.floor(
            x + CrystalMiddle  * increment_ratio_x)  # this -1 represents that the opposition of direction
        # between the lab x-axis and the wavevector
        new_y = np.floor(y - CrystalMiddle * increment_ratio_y)

        new_z = np.floor(z + CrystalMiddle * increment_ratio_z)

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
                       int((new_x)))


    return potential_coord, CrystalMiddle

@jit(nopython=True)
def average(potential_coord,air_outermost_potential_coord):
    difference = np.abs(np.array(potential_coord) - np.array(air_outermost_potential_coord))
    ave = (difference[0] + difference[1] + difference[2] ) /3
    return ave

@jit(nopython=True)
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

    resolution = 1
    #reference_path_ray, reference_posi, reference_classes = reference_path
    z_max, y_max, x_max = label_list.shape
    z, y, x = coord

    # path_1 =[(z,y,i) for i in  range(x, x_max, 1)]
    x_max -= 1
    y_max -= 1
    z_max -= 1

    path_2 = []
    # classes_posi = [0]
    classes = ['coord']

    if face == 'TOPZX':
        increment_ratios = top_increment_ratio(theta,phi)
        AirLongest = y

    elif face == "BOTZX":
        increment_ratios = bottom_increment_ratio(theta, phi)
        AirLongest = y_max-y

    elif face == "BACKZY":
        increment_ratios = back_increment_ratio(theta, phi)
        AirLongest = x

    elif face == "FRONTZY":
        increment_ratios =front_increment_ratio(theta, phi)
        AirLongest = x_max-x

    elif face == "LEYX":
        increment_ratios = left_increment_ratio(theta, phi)
        AirLongest = z

    elif face == "RIYX":
        increment_ratios = right_increment_ratio(theta, phi)
        AirLongest = z_max - z

    else:
        raise RuntimeError("unexpected ray out face")

    a,b,c=coord
    crd=(a,b,c)
    path_2.append(crd)

    AirShortest = 0
    air_outermost_potential_coord, AirMiddle_outer = bisection(AirLongest, AirShortest, resolution, label_list,
                                                                   increment_ratios, coord, boundary='inner', cls=0)
    classes.append('air_outermost')
    path_2.append(air_outermost_potential_coord)

    # finding the boundary between outer boudary of the crystal
    CrystalLongest = AirMiddle_outer
    CrystalShortest = 0
    cr_outer_potential_coord, CrystalMiddle = bisection(CrystalLongest, CrystalShortest, resolution, label_list,
                                                            increment_ratios, coord, boundary='outer', cls=3)
    classes.append('cr_outer')
    path_2.append(cr_outer_potential_coord)
    # pdb.set_trace()
    # starting from the crystal to calculate the classes along the path to find the LOOP
    LoopLongest = AirMiddle_outer
    LoopShortest = CrystalMiddle
    potential_coord, LoopMiddle = bisection(LoopLongest, LoopShortest, resolution, label_list,
                                                increment_ratios, coord, boundary='inner', cls=2)
    if average(potential_coord,air_outermost_potential_coord) < 1:
        pass
    else:
        classes.append('lo_inner')
        path_2.append(potential_coord)
        LoopLongest = AirMiddle_outer
        LoopShortest = LoopMiddle
        potential_coord, LoopMiddle = bisection(LoopLongest, LoopShortest, resolution, label_list,
                                                    increment_ratios, coord, boundary='outer', cls=2)
        classes.append('lo_outer')
        path_2.append(potential_coord)

    # starting from the crystal to calculate the classes along the path to find the BUBBLE
    BubbleLongest = AirMiddle_outer
    BubbleShortest = CrystalMiddle
    potential_coord, BubbleMiddle = bisection(BubbleLongest, BubbleShortest, resolution, label_list,
                                                  increment_ratios, coord, boundary='inner', cls=4)
    if average(potential_coord,air_outermost_potential_coord) < 1:
        pass
    else:
        classes.append('bu_inner')
        path_2.append(potential_coord)
        BubbleLongest = AirMiddle_outer
        BubbleShortest = BubbleMiddle
        potential_coord, BubbleMiddle = bisection(BubbleLongest, BubbleShortest, resolution, label_list,
                                                      increment_ratios, coord, boundary='outer', cls=4)
        classes.append('bu_outer')
        path_2.append(potential_coord)

    # starting from the crystal to calculate the classes along the path to find the other possible air
    Air2Longest = AirMiddle_outer
    Air2Shortest = CrystalMiddle
    potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
                                                increment_ratios, coord, boundary='inner', cls=0)
    if average(potential_coord,air_outermost_potential_coord)< 1:
        pass
    else:
        classes.append('air_inner')
        path_2.append(potential_coord)
        Air2Longest = AirMiddle_outer
        Air2Shortest = Air2Middle
        potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
                                                    increment_ratios, coord, boundary='outer', cls=0)
        classes.append('air_outer')
        path_2.append(potential_coord)


    return path_2, classes
