import numpy as np
import pdb
import matplotlib.pyplot as plt
from numba import jit
from skimage.draw import line
# import math
np.set_printoptions(suppress=True)


def dials_2_thetaphi_11(rotated_s1,L1=False):
    if L1 is True:
        # L1 is the incident beam and L2 is the diffracted so they are opposite
        rotated_s1 = -rotated_s1

    if rotated_s1[1] == 0:
        # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
        theta = np.arctan(-rotated_s1[2] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2) + 0.001))
        # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
        phi = np.arctan(-rotated_s1[0] / (rotated_s1[1] + 0.001))
    else:
        if rotated_s1[1] < 0:
            theta = np.arctan(-rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
            phi = np.arctan(-rotated_s1[0] / (rotated_s1[1]))
        else:
            if rotated_s1[2] < 0:
                theta = np.pi - np.arctan(
                    -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)

            else:
                theta = - np.pi - np.arctan(
                    -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
            phi = - np.arctan(-rotated_s1[0] / (-rotated_s1[1]))  # tan-1(-z/-x)
    return theta,phi


def which_face_2(coord,shape,theta,phi):
    # deciding which plane to go out, to see which direction (xyz) has increment of 1

    """
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
    #assert theta <= np.pi, phi <= np.pi/2
    z_max, y_max, x_max = shape
    x_max -= 1
    y_max -= 1
    z_max -= 1
    z,y,x=coord

    if np.abs(theta)<np.pi/2:

        theta_up = np.arctan(( y - 0) / (x - 0 + 0.001))
        theta_down = -np.arctan((y_max - y) / (x - 0 + 0.001))  # negative
        phi_right = np.arctan((z_max - z) / (x - 0 + 0.001))
        phi_left = -np.arctan((z - 0) / (x - 0 + 0.001))  # negative
        omega  = np.arctan( np.tan(theta) * np.cos(phi)  )

        if omega > theta_up:
            # at this case, theta is positive,
            # normally the most cases for theta > theta_up, the ray passes the top ZX plane
            # if the phis are smaller than both edge limits
            # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            side = (y - 0) * np.sin(abs(phi)) / np.tan(theta)  # the length of rotation is the projected length on x
            if side > (z-0) and phi < phi_left:
                face= 'LEYX'
            elif side > (z_max-z) and  phi > phi_right :
                face = 'RIYX'
            else:
                face= 'TOPZX'

        elif omega < theta_down:
            side = (y_max-y )* np.sin(abs(phi)) / np.tan(-theta)
            if side > (z-0) and phi < phi_left:
                face= 'LEYX'
            elif side > (z_max-z) and  phi > phi_right :
                face = 'RIYX'
            else:
                face= 'BOTZX'

        elif phi > phi_right:
            # when the code goes to this line, it means the theta is within the limits
            face= 'RIYX'
        elif phi < phi_left:
            face= 'LEYX'

        else:
            # ray passes through the back plane
            face="BACKZY"

    else:
        # theta is larger than 90 degree or smaller than -90
        theta_up = np.arctan(( y - 0) / (x_max - x + 0.001))
        theta_down = np.arctan((y_max - y) / (x_max - x + 0.001))  # negative
        phi_left = np.arctan((z_max - z) / (x_max - x + 0.001))  # it is the reverse of the top phi_left
        phi_right = -np.arctan((z - 0) / (x_max - x + 0.001))  # negative
    #
    #
        if (np.pi - theta) > theta_up and theta > 0:
            # at this case, theta is positive,
            # normally the most cases for theta > theta_up, the ray passes the top ZX plane
            # if the phis are smaller than both edge limits
            # the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            side = (y - 0) * np.sin(abs(phi)) / np.abs(np.tan(theta))
            if side > (z-0) and -phi < phi_right :
                face= 'LEYX'
            elif side > (z_max-z) and  -phi > phi_left :
                face = 'RIYX'
            else:
                face= 'TOPZX'
    #
        elif theta > theta_down-np.pi and theta <=0:
            side = (y_max-y )* np.sin(abs(phi)) / np.abs(np.tan(-theta))
            if side > (z-0) and  -phi < phi_right:
                face= 'LEYX'
            elif side > (z_max-z) and  -phi > phi_left:
                face = 'RIYX'
            else:
                face= 'BOTZX'

        # elif  -phi < phi_right:
        #     # when the code goes to this line, it means the theta is within the limits
        #     face=  'LEYX'
        # elif -phi > phi_left:
        #     face= 'RIYX'

        else:
            # ray passes through the back plane
            face='FRONTZY'
    # pdb.set_trace()
    return  face

def dials_2_numpy_11(vector):
    # (x',y',z') in standard vector but in numpy (z,y,x)
    # rotate the coordinate system about x'(z in numpy) for 180
    # vector =vector.astype(np.float32)
    # numpy_2_dials_1 = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
    #                             [-np.sin(np.pi), np.cos(np.pi), 0],
    #                             [0, 0, 1]],dtype=np.float32)
    numpy_2_dials_1 = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])

    back2 = numpy_2_dials_1.dot(vector)

    return  back2



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


def cal_path2_plus(path_2,voxel_size):
    # pdb.set_trace()
    voxel_length_z=voxel_size[0]
    voxel_length_y = voxel_size[1]
    voxel_length_x = voxel_size[2]
    path_ray = path_2[0]
    posi = path_2[1]
    classes = path_2[2]

    cr_l_2 = 0
    lo_l_2 = 0
    li_l_2 = 0
    bu_l_2 = 0


        # total_length = ( path_ray[-1][1] - path_ray[0][1] )/ (np.sin(np.abs(omega)))
    total_length=np.sqrt(((path_ray[-1][1]  - path_ray[0][1] ) * voxel_length_y ) ** 2 +
                         ((path_ray[-1][0]  - path_ray[0][0] ) * voxel_length_z ) ** 2 +
                         ( (path_ray[-1][2]  - path_ray[0][2] ) * voxel_length_x )** 2)
    for j, trans_index in enumerate(posi):

        if classes[j] == 'cr':
            if j < len(posi) - 1:
                cr_l_2 += total_length * ( (posi[j+1]-posi[j])/len(path_ray))
            else:
                cr_l_2 += total_length * ((len(path_ray)- posi[j]) / len(path_ray))
        elif classes[j] == 'li':
            if j < len(posi) - 1:
                li_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
            else:
                li_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
        elif classes[j] == 'lo':
            if j < len(posi) - 1:
                lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
            else:
                lo_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
        elif classes[j] == 'bu':
            if j < len(posi) - 1:
                bu_l_2 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
            else:
                bu_l_2 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
        else:
            pass
    #     # if classes[j] == 'cr':
    #     #     if j < len(posi) - 1:
    #     #         cr_l_2_total= np.abs(path_ray[posi[j+1] - 1][1] - coord[1]) + 0.5
    #     #     else:
    #     #         cr_l_2_total = np.abs(path_ray[-1][1] - coord[1]) + 0.5
    #     #     # cr_l_2_total = h_cr / (np.sin(np.abs(omega)))
    #     #     cr_l_2 += cr_l_2_total - residue
    #     #     residue = cr_l_2_total
    #     # elif classes[j] == 'li':
    #     #     if j < len(posi) - 1:
    #     #         h_li = np.abs(path_ray[posi[j+1] - 1][1] - coord[1]) + 0.5
    #     #     else:
    #     #         h_li = np.abs(path_ray[-1][1] - coord[1]) + 0.5
    #     #     li_l_2_total = h_li / (np.sin(np.abs(omega)))
    #     #     li_l_2 += li_l_2_total - residue
    #     #     residue = li_l_2_total
    #     # elif classes[j] == 'lo':
    #     #     if j < len(posi) - 1:
    #     #         h_lo = np.abs(path_ray[posi[j+1] - 1][1] - coord[1]) + 0.5
    #     #     else:
    #     #         h_lo = np.abs(path_ray[-1][1] - coord[1]) + 0.5
    #     #     lo_l_2_total = h_lo / (np.sin(np.abs(omega)))
    #     #     lo_l_2 += lo_l_2_total - residue
    #     #     residue = lo_l_2_total
    #     # elif classes[j] == 'bu':
    #     #     if j < len(posi) - 1:
    #     #         h_bu = np.abs(path_ray[posi[j+1] - 1][1] - coord[1]) + 0.5
    #     #     else:
    #     #         h_bu = np.abs(path_ray[-1][1] - coord[1]) + 0.5
    #     #     bu_l_2_total = h_bu / (np.sin(np.abs(omega)))
    #     #     bu_l_2 += bu_l_2_total - residue
    #     #     residue = bu_l_2_total
    # else:
    #     # Pythagorean theorem
    #     cr_l_2 = 0
    #     lo_l_2 = 0
    #     li_l_2 = 0
    #     bu_l_2 = 0
    #     for j, index in enumerate(posi):
    #         # pdb.set_trace()
    #         if  classes[j] == 'cr':
    #             if j < len(posi) - 1:
    #                 x_cr = np.abs(path_ray[posi[j + 1] ][2] - path_ray[posi[j]][2])
    #                 z_cr = np.abs(path_ray[posi[j + 1] ][0] - path_ray[posi[j] ][0])
    #                 y_cr = np.abs(path_ray[posi[j + 1] ][1] - path_ray[posi[j] ][1])
    #             else:
    #                 x_cr = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
    #                 z_cr = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
    #                 y_cr = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
    #             cr_l_2_total = np.sqrt( (x_cr+0.5) ** 2 + (z_cr+0.5) ** 2 + (y_cr+0.5) ** 2)
    #             cr_l_2 += cr_l_2_total
    #
    #         elif classes[j] == 'li':
    #             if j < len(posi) - 1:
    #                 x_li = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
    #                 z_li = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
    #                 y_li = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
    #             else:
    #                 x_li = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
    #                 z_li = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
    #                 y_li = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
    #             li_l_2_total = np.sqrt(x_li ** 2 + z_li ** 2 + y_li ** 2)
    #             li_l_2 += li_l_2_total
    #
    #         elif classes[j] == 'lo':
    #             if j < len(posi) - 1:
    #                 x_lo = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
    #                 z_lo = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
    #                 y_lo = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
    #             else:
    #                 x_lo = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
    #                 z_lo = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
    #                 y_lo = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
    #             lo_l_2_total = np.sqrt(x_lo ** 2 + z_lo ** 2 + y_lo ** 2)
    #             lo_l_2 += lo_l_2_total
    #         elif classes[j] == 'bu':
    #             if j < len(posi) - 1:
    #                 x_bu = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
    #                 z_bu = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
    #                 y_bu = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
    #             else:
    #                 x_bu = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
    #                 z_bu = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
    #                 y_bu = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
    #             bu_l_2_total = np.sqrt(x_bu ** 2 + z_bu ** 2 + y_bu ** 2)
    #             bu_l_2 += bu_l_2_total
    #         else:
    #             pass
    # can add the other class path
    return li_l_2, lo_l_2, cr_l_2,bu_l_2


def cal_num(path_2,voxel_size):
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
    li_l_2,lo_l_2,cr_l_2,bu_l_2=cal_path2_plus(path_2,voxel_size)
    # if path_1 is not None:
    #     li_l_1, lo_l_1, cr_l_1,bu_l_1 = cal_path2_plus(path_1, voxel_size)
    #     return li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2
    # else:
    return li_l_2, lo_l_2, cr_l_2, bu_l_2


def cal_rate(numbers,coefficients,exp=True ):
    mu_li, mu_lo, mu_cr,mu_bu = coefficients

    if len(numbers)==8:
        li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
    else:
        li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
    if exp:
        abs = np.exp(-((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) )
                    ))
    else:
        abs = ((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) ))
    return  abs

def cal_rate_single(numbers,coefficients,exp=True ):
    mu_li, mu_lo, mu_cr,mu_bu = coefficients
    assert len(numbers)==4
    li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers

    if exp:
        abs = np.exp(-((mu_li * ( li_l_2) +
                     mu_lo * ( lo_l_2) +
                     mu_cr * (cr_l_2) +
                         mu_bu * ( bu_l_2) )
                    ))
    else:
        abs = ((mu_li * (li_l_2) +
                     mu_lo * (lo_l_2) +
                     mu_cr * ( cr_l_2) +
                         mu_bu * (bu_l_2) ))
    return  abs

def cal_path2(path_2,coord,label_list,rate_list,omega):
    # Pythagorean theorem
    # order:
    # but it has error that require the accuracy of the label_list,
    # e.g. 'lo', (398, 375, 472), 'cr', (397, 374, 471), 'li', (396, 373, 471), 'cr', (396, 372, 470),

    # if np.abs(omega) < 1/180*np.pi:
    #     # if the scattering beam is in too small angle, treat as a straight line
    #     li_list_2 = []
    #     lo_list_2 = []
    #     cr_list_2 = []
    #     for i in path_2:
    #         if label_list[i] == rate_list['li']:
    #             li_list_2.append(i)
    #         elif label_list[i] == rate_list['lo']:
    #             lo_list_2.append(i)
    #         elif label_list[i] == rate_list['cr']:
    #             cr_list_2.append(i)
    #         else:
    #             pass
    #     li_l_2, lo_l_2, cr_l_2 = len(li_list_2) ,len(lo_list_2),len(cr_list_2)
    #     return     li_l_2,lo_l_2,cr_l_2


    # path_ray=[]
    # for k,i in enumerate(path_2):
    #     if k==0:
    #         path_ray.append('cr')
    #         path_ray.append(i)
    #     else:
    #         if label_list[i] == rate_list['li']:
    #             # pdb.set_trace()
    #             if label_list[path_2[k]] != label_list[path_2[k-1]]:
    #                 path_ray.append('li')
    #                 path_ray.append(i)
    #             else:
    #                 path_ray.append(i)
    #             # li_list_2.append(i)
    #         elif label_list[i] ==rate_list['lo']:
    #             if label_list[path_2[k]] != label_list[path_2[k-1]]:
    #                 path_ray.append('lo')
    #                 path_ray.append(i)
    #             else:
    #                 path_ray.append(i)
    #             # lo_list_2.append(i)
    #         elif label_list[i] ==rate_list['cr']:
    #             if label_list[path_2[k]] != label_list[path_2[k-1]]:
    #                 path_ray.append('cr')
    #                 path_ray.append(i)
    #             else:
    #                 path_ray.append(i)
    #                 # cr_list_2.append(i)
    #         else:
    #             # currently only three classes
    #             pass
    #
    # posi=[]
    #
    # # the hypothesis is that all components only appear once, not repeated
    #
    # classes=[]
    # for k,element in enumerate(path_ray):
    #     if type(element) is str:
    #         posi.append(k)
    #         classes.append(element)

    path_ray, posi = path_2
    if np.abs(omega)>1/180*np.pi:
        cr_l_2 = 0
        lo_l_2 = 0
        li_l_2 = 0
        residue = 0
        for j,index in enumerate(posi):

            if path_ray[index]=='cr':
                if j < len(posi)-1:
                    h_cr = np.abs(path_ray[posi[j+1]-1][1] - coord[1] + 0.5)
                else:
                    h_cr = np.abs(path_ray[-1][1] - coord[1] + 0.5)
                cr_l_2_total = h_cr / ( np.sin(np.abs(omega)))
                cr_l_2 += cr_l_2_total-residue
                residue =cr_l_2_total

            elif path_ray[index]=='li':
                if j < len(posi)-1:
                    h_li = np.abs(path_ray[posi[j+1]-1][1] - coord[1] + 0.5)
                else:
                    h_li = np.abs(path_ray[-1][1] - coord[1] + 0.5)
                li_l_2_total = h_li / (np.sin(np.abs(omega)))
                li_l_2 += li_l_2_total-residue
                residue = li_l_2_total

            elif path_ray[index]=='lo':
                if j < len(posi)-1:
                    h_lo = np.abs(path_ray[posi[j+1]-1][1] - coord[1] + 0.5)
                else:
                    h_lo = np.abs(path_ray[-1][1] - coord[1] + 0.5)
                lo_l_2_total = h_lo / (np.sin(np.abs(omega)) )
                lo_l_2 += lo_l_2_total-residue
                residue =lo_l_2_total

    else:
        # Pythagorean theorem
        cr_l_2 = 0
        lo_l_2 = 0
        li_l_2 = 0
        residue = 0
        for j, index in enumerate(posi):
            if path_ray[index] == 'cr':
                if j < len(posi) - 1:
                    x_cr = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] + 1][2] )
                    z_cr = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] + 1][0] )
                    y_cr = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] + 1][1] )
                else:
                    x_cr = np.abs(path_ray[-1][2] - path_ray[posi[j] + 1][2] )
                    z_cr = np.abs(path_ray[-1][0] - path_ray[posi[j] + 1][0] )
                    y_cr = np.abs(path_ray[-1][1] - path_ray[posi[j] + 1][1] )
                cr_l_2_total = np.sqrt(x_cr**2+z_cr**2+y_cr**2)
                cr_l_2 += cr_l_2_total

            elif path_ray[index] == 'li':
                if j < len(posi) - 1:
                    x_li = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] + 1][2] )
                    z_li = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] + 1][0] )
                    y_li = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] + 1][1] )
                else:
                    x_li = np.abs(path_ray[-1][2] - path_ray[posi[j] + 1][2] )
                    z_li = np.abs(path_ray[-1][0] - path_ray[posi[j] + 1][0] )
                    y_li = np.abs(path_ray[-1][1] - path_ray[posi[j] + 1][1] )
                li_l_2_total = np.sqrt(x_li**2+z_li**2+y_li**2)
                li_l_2 += li_l_2_total

            elif path_ray[index] == 'lo':
                if j < len(posi) - 1:
                    x_lo = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] + 1][2])
                    z_lo = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] + 1][0])
                    y_lo = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] + 1][1])
                else:
                    x_lo = np.abs(path_ray[-1][2] - path_ray[posi[j] + 1][2])
                    z_lo = np.abs(path_ray[-1][0] - path_ray[posi[j] + 1][0])
                    y_lo = np.abs(path_ray[-1][1] - path_ray[posi[j] + 1][1])
                lo_l_2_total = np.sqrt(x_lo**2+z_lo**2+y_lo**2)
                lo_l_2 += lo_l_2_total

    # can add the other class path
    return li_l_2,lo_l_2,cr_l_2

##### iterative bisection method##########
def angle2vector ( theta , phi ) :
    # the theta defined here is the supplementary angle of that in spherical coordinate
    # so sin(theta) = cos(azimuthal angle)
    x = np.cos( theta ) * np.cos( phi )
    y = np.cos( theta ) * np.sin( phi )
    z = np.sin( theta )
    return np.array( [x , y , z] )


def vector2angle ( vector ) :
    # the theta defined here is the supplementary angle of that in spherical coordinate
    # so sin(theta) = cos(azimuthal angle)
    x , y , z = vector
    theta = np.arcsin( z )
    phi = np.arctan2( y , x )
    return theta , phi


def cube_face ( ray_origin , ray_direction , cube_size , L1 = False ) :
    """
    Determine which face of a cube a ray is going out.
    ray casting method
    To find the distance along the vector where the intersection point
    with a plane occurs, you can use the dot product of the vector and
      the plane normal to find the component of the vector that is
      perpendicular to the plane. Then, you can use this perpendicular
      component and the plane equation to solve for the distance along
      the vector to the intersection point.
      t = (plane_distance - np.dot(vector_origin, plane_normal)) /
         np.dot(vector, plane_normal)
    Args:
        ray_origin (tuple): the origin of the ray, as a tuple of (x, y, z) coordinates
        ray_direction (tuple): the direction of the ray, as a unit vector tuple of (x, y, z) coordinates
        cube_center (tuple): the center of the cube, as a tuple of (x, y, z) coordinates
        cube_size (float): the size of the cube, as a scalar value
    /*  'FRONTZY' = 1;
*   'LEYX' = 2 ;
*   'RIYX' = 3;
    'TOPZX' = 4;
    'BOTZX' = 5;
    "BACKZY" = 6 ;

    Returns:
        str: the name of the face that the ray intersects with first, or None if the ray doesn't intersect with the cube
    """
    # Determine the minimum and maximum x, y, and z coordinates of the cube

    # min_x = cube_center[0] - cube_size / 2
    # max_x = cube_center[0] + cube_size / 2
    # min_y = cube_center[1] - cube_size / 2
    # max_y = cube_center[1] + cube_size / 2
    # min_z = cube_center[2] - cube_size / 2
    # max_z = cube_center[2] + cube_size / 2
    # if L1 is True:
    #     ray_direction = -ray_direction
    # L1=False

    min_x = 0
    max_x = cube_size[2]
    min_y = 0
    max_y = cube_size[1]
    min_z = 0
    max_z = cube_size[0]
    # Calculate the t values for each face of the cube
    tx_min = (min_x - ray_origin[2]) / ray_direction[2]
    tx_max = (max_x - ray_origin[2]) / ray_direction[2]
    ty_min = (min_y - ray_origin[1]) / ray_direction[1]
    ty_max = (max_y - ray_origin[1]) / ray_direction[1]
    tz_min = (min_z - ray_origin[0]) / ray_direction[0]
    tz_max = (max_z - ray_origin[0]) / ray_direction[0]
    # print("tx min is {}".format(tx_min))
    # print("ty min  is {}".format(ty_min))
    # print("tz min  is {}".format(tz_min))
    # print("tx max  is {}".format(tx_max))
    # print("ty max  is {}".format(ty_max))
    # print("tz max  is {}".format(tz_max))
    # Determine which face is intersected first
    # t_mini = max( tx_min , ty_min , tz_min )
    # t_max = min( tx_max , ty_max , tz_max )
    t_numbers = [tx_min , ty_min , tz_min , tx_max , ty_max , tz_max]
    non_negative_numbers = [num for num in t_numbers if num >= 0]
    # if L1 is True:
    try :
        t_min = min( non_negative_numbers )
    except :
        t_min = max( non_negative_numbers )
        print( "t_min is max at {}".format( ray_direction ) )
        print( "t_min is max at {}".format( ray_origin ) )
    # else:
    # try:
    #         t_min=max(t_mini,t_max)
    # except:
    #     pdb.set_trace()

    # print(t_numbers)
    # pdb.set_trace()
    # if t_min > t_max :
    #     # The ray doesn't intersect with the cube
    #     return None
    # print( "t_min is {}".format( t_min ) )
    if t_min == tx_min :
        # The ray intersects with the left face of the cube]
        if L1 is True :
            return "FRONTZY"
        else :
            return "BACKZY"
    elif t_min == tx_max :
        # The ray intersects with the right face of the cube
        if L1 is True :
            return "BACKZY"
        else :
            return "FRONTZY"
    elif t_min == ty_min :
        # The ray intersects with the bottom face of the cube
        if L1 is True :
            return 'BOTZX'
        else :
            return 'TOPZX'
    elif t_min == ty_max :
        # The ray intersects with the top face of the cube
        if L1 is True :
            return 'TOPZX'
        else :
            return 'BOTZX'
    elif t_min == tz_min :
        # The ray intersects with the front face of the cube
        if L1 is True :
            return 'RIYX'
        else :
            return 'LEYX'
    elif t_min == tz_max :
        # The ray intersects with the back face of the cube
        if L1 is True :
            return 'LEYX'
        else :
            return 'RIYX'
    else :
        RuntimeError( 'face determination has a problem with direction {}'
                      'and position {}'.format( ray_direction , ray_origin ) )


# @jit(nopython=True)
def which_face_matrix(matrix_coord,diffracted_beam,shape,exit=True):
    """

    :param coord: coord = [x,y-shape[1]+1,z]
    :param diffracted_beam:
    :param shape:
    :return:
    """
    diffracted_beam = diffracted_beam.astype(np.float32)
    coord = np.array([matrix_coord[0], matrix_coord[1] - shape[1] + 1, matrix_coord[2]],dtype=np.float32)
    new_shape = np.array(shape,dtype=np.float32) - 1
    z_max, y_max, x_max = new_shape
    new_shape = numpy_2_dials(new_shape)
    vertice_0 = numpy_2_dials(np.array([0, 0, 0],dtype=np.float32))
    vertice_1 = numpy_2_dials(np.array([z_max, 0, 0],dtype=np.float32))
    vertice_2 = numpy_2_dials(np.array([z_max, 0, x_max],dtype=np.float32))
    vertice_3 = numpy_2_dials(np.array([0, 0, x_max],dtype=np.float32))
    vertice_4 = numpy_2_dials(np.array([0, y_max, 0],dtype=np.float32))
    vertice_5 = numpy_2_dials(np.array([z_max, y_max, 0],dtype=np.float32))
    vertice_6 = numpy_2_dials(np.array([z_max, y_max, x_max],dtype=np.float32))
    vertice_7 = numpy_2_dials(np.array([0, y_max, x_max],dtype=np.float32))

    normal_back = points_2_planes(vertice_0, vertice_1, vertice_4, vertice_5)
    normal_front = points_2_planes(vertice_2, vertice_3, vertice_6, vertice_7)
    normal_top = points_2_planes(vertice_0, vertice_1, vertice_2, vertice_3)
    normal_bot = points_2_planes(vertice_4, vertice_5, vertice_6, vertice_7)
    normal_left = points_2_planes(vertice_0, vertice_3, vertice_4, vertice_7)
    normal_right = points_2_planes(vertice_1, vertice_2, vertice_5, vertice_6)

    planes_list = ['TOPZX', 'BOTZX', 'RIYX', 'LEYX', 'BACKZY', 'FRONTZY']
    # planes_list=['BACKZY','FRONTZY']
    # incident_vector = np.array([-0.0013731, 0.999999,
    #                             0])  # x,y,z in lab coordinate  z,y,x=incident_vector  # changing into my coordinate system

    for plane in planes_list:
        if plane == 'TOPZX':
            normal = normal_top
            planePoint = vertice_3
        elif plane == 'RIYX':
            normal = normal_right
            planePoint = vertice_6
        elif plane == 'LEYX':
            normal = normal_left
            planePoint = vertice_7
        elif plane == 'BOTZX':
            normal = normal_bot
            planePoint = vertice_7
        elif plane == 'BACKZY':
            normal = normal_back
            planePoint = vertice_5
        elif plane == 'FRONTZY':
            normal = normal_front
            planePoint = vertice_7

        intersection, si = LinePlaneCollision(normal, planePoint, diffracted_beam, coord, epsilon=1e-6)
        # print('\n')
        # print(intersection)
        # print(si)
        # pdb.set_trace()


        if exit is True:
            # path_2 so the vector positive direction is the exit plane
            if intersection is None:
                continue
            elif si < 0:  # the dials vector direction is the same as the real diffracted beam
                continue
            else:
                # i_x, i_y, i_z =np.round( intersection, decimals=6)
                i_x, i_y, i_z = intersection
                i_x, i_y, i_z = np.round(i_z),np.round(i_z),np.round(i_z)
                if   i_x  >=0  and i_x <=  new_shape[0]+1 \
                        and  i_y  >= new_shape[1]-1 and i_y <= 0  \
                        and i_z  >= 0   and i_z <=  new_shape[2]+1:
                    #out = plane
                    return  plane
                else:
                    pass
        else:
            # path_2 so the vector negative direction is the enter plane
            if intersection is None:
                continue
            elif si > 0:
                continue
            else:
                # i_x, i_y, i_z = np.round(intersection, decimals=6)
                i_x, i_y, i_z = np.round(intersection)
                #i_x, i_y, i_z = np.round(i_z), np.round(i_z), np.round(i_z)
                if   i_x  >=0  and i_x <=  new_shape[0]+1 \
                        and  i_y  >= new_shape[1]-1 and i_y <= 0  \
                        and i_z  >= 0   and i_z <=  new_shape[2]+1:
                    #out = plane
                    return  plane
                else:
                    pass
    #return  out

#
# @jit(nopython=True)
def numpy_2_dials(vector):
    # (x',y',z') in standard vector but in numpy (z,y,x)
    # rotate the coordinate system about x'(z in numpy) for 180
    # vector =vector.astype(np.float32)
    # numpy_2_dials_1 = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
    #                             [-np.sin(np.pi), np.cos(np.pi), 0],
    #                             [0, 0, 1]],dtype=np.float32)
    numpy_2_dials_1 = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]],dtype=np.float32)
    #back1 = np.dot(numpy_2_dials_1, vector)
    back1 = numpy_2_dials_1.dot(vector)

    # reflect the coordinate system about y'x' plane (yz in numpy)
    numpy_2_dials_2 = np.array([[-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],dtype=np.float32)
    # back2 = np.dot(numpy_2_dials_2, back1)
    back2 = numpy_2_dials_2.dot(back1)

    return  back2

# @jit(nopython=True)
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    #https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    #https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    rayDirection = rayDirection.astype(np.float32)
    ndotu = planeNormal.dot(rayDirection)
    if np.abs(ndotu) < epsilon:
        return None,None

    si = planeNormal.dot(planePoint - rayPoint) / ndotu
    intersection = rayPoint + si * rayDirection

    return intersection,si

# @jit(nopython=True)
def points_2_planes(p1,p2,p3,p4):
    z1, y1, x1 = p1
    z2, y2, x2 = p2
    z3, y3, x3 = p3
    # z4, y4, x4 = p4
    v1 = np.array([z3 - z1, y3 - y1, x3 - x1],dtype=np.float32)
    v2 = np.array([z2 - z1, y2 - y1, x2 - x1],dtype=np.float32)
    normal=np.cross(v1,v2)
    normal=normal/np.linalg.norm(normal)
    # d_1 = (normal[0] * z1 + normal[1] * y1 + normal[2] * x1)
    # d_2 = (normal[0] * z4 + normal[1] * y4 + normal[2] * x4)

    # try:
    #     assert  d_1 == d_2  # verifying the fourth vertice is on that plane
    # except:
    #     pdb.set_trace()
    return normal


# def cal_path1_plus(path_1, coord, rotate_angle):
#
#
#     # the hypothesis is that all components only appear once, not repeated
#     cr_l_1 = 0
#     lo_l_1 = 0
#     li_l_1 = 0
#     bu_l_1 = 0
#     residue = 0
#     path_ray = path_1[0]
#     posi = path_1[1]
#     classes = path_1[2]
#     # for k, element in enumerate(path_ray):
#     #     if type(element) is str:
#     #         posi.append(k)
#     #         classes.append(element)
#
#     # if np.abs(rotate_angle) < 2 / 180 * np.pi or np.abs(rotate_angle - np.pi/2) < 2 / 180 * np.pi \
#     #         or np.abs(rotate_angle - np.pi) < 2 / 180 * np.pi or np.abs(rotate_angle - 1.5 * np.pi) < 2 / 180 * np.pi:
#     if np.abs(np.sin(rotate_angle)) < 0.02:
#         # Pythagorean theorem
#         cr_l_1 = 0
#         lo_l_1 = 0
#         li_l_1 = 0
#         bu_l_1 = 0
#         for j, index in enumerate(posi):
#
#             if classes[j] == 'cr':
#                 if j < len(posi) - 1:
#                     x_cr = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
#                     z_cr = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0])
#                     y_cr = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1])
#                 else:
#                     x_cr = np.abs(path_ray[-1][2] - path_ray[posi[j]][2])
#                     z_cr = np.abs(path_ray[-1][0] - path_ray[posi[j]][0])
#                     y_cr = np.abs(path_ray[-1][1] - path_ray[posi[j]][1])
#                 cr_l_1_total = np.sqrt(x_cr ** 2 + z_cr ** 2 + y_cr ** 2)
#                 cr_l_1 += cr_l_1_total
#
#             elif classes[j] == 'li':
#                 if j < len(posi) - 1:
#                     x_li = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
#                     z_li = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0])
#                     y_li = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1])
#                 else:
#                     x_li = np.abs(path_ray[-1][2] - path_ray[posi[j]][2])
#                     z_li = np.abs(path_ray[-1][0] - path_ray[posi[j]][0])
#                     y_li = np.abs(path_ray[-1][1] - path_ray[posi[j]][1])
#                 li_l_1_total = np.sqrt(x_li ** 2 + z_li ** 2 + y_li ** 2)
#                 li_l_1 += li_l_1_total
#
#             elif classes[j] == 'lo':
#                 if j < len(posi) - 1:
#                     x_lo = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
#                     z_lo = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0])
#                     y_lo = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1])
#                 else:
#                     x_lo = np.abs(path_ray[-1][2] - path_ray[posi[j]][2])
#                     z_lo = np.abs(path_ray[-1][0] - path_ray[posi[j]][0])
#                     y_lo = np.abs(path_ray[-1][1] - path_ray[posi[j]][1])
#                 lo_l_1_total = np.sqrt(x_lo ** 2 + z_lo ** 2 + y_lo ** 2)
#                 lo_l_1 += lo_l_1_total
#             elif classes[j] == 'bu':
#                 if j < len(posi) - 1:
#                     x_bu = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
#                     z_bu = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0])
#                     y_bu = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1])
#                 else:
#                     x_bu = np.abs(path_ray[-1][2] - path_ray[posi[j]][2])
#                     z_bu = np.abs(path_ray[-1][0] - path_ray[posi[j]][0])
#                     y_bu = np.abs(path_ray[-1][1] - path_ray[posi[j]][1])
#                 bu_l_1_total = np.sqrt(x_bu ** 2 + z_bu ** 2 + y_bu ** 2)
#                 bu_l_1 += bu_l_1_total
#     else:
#
#         # total_length = ( path_ray[-1][1] - path_ray[0][1] )/ (np.sin(np.abs(omega)))
#         total_length = np.sqrt((path_ray[-1][1] - path_ray[0][1]) ** 2 +
#                                (path_ray[-1][0] - path_ray[0][0]) ** 2 +
#                                (path_ray[-1][2] - path_ray[0][2]) ** 2)
#         for j, trans_index in enumerate(posi):
#
#             if classes[j] == 'cr':
#                 if j < len(posi) - 1:
#                     cr_l_1 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
#                 else:
#                     cr_l_1 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
#             elif classes[j] == 'li':
#                 if j < len(posi) - 1:
#                     li_l_1 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
#                 else:
#                     li_l_1 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
#             elif classes[j] == 'lo':
#                 if j < len(posi) - 1:
#                     lo_l_1 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
#                 else:
#                     lo_l_1 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
#             elif classes[j] == 'bu':
#                 if j < len(posi) - 1:
#                     bu_l_1 += total_length * ((posi[j + 1] - posi[j]) / len(path_ray))
#                 else:
#                     bu_l_1 += total_length * ((len(path_ray) - posi[j]) / len(path_ray))
#         # for j, trans_index in enumerate(posi):
#         #         if classes[j] == 'cr':
#         #             if j < len(posi) - 1:
#         #                 h_cr = np.abs(path_ray[posi[j + 1] - 1][1] - coord[1]) + 0.5
#         #             else:
#         #                 h_cr = np.abs(path_ray[-1][1] - coord[1]) + 0.5
#         #             cr_l_1_total = h_cr / np.abs(np.sin(rotate_angle))
#         #             cr_l_1 += cr_l_1_total - residue
#         #             residue = cr_l_1_total
#         #
#         #         elif classes[j] == 'li':
#         #             if j < len(posi) - 1:
#         #                 h_li = np.abs(path_ray[posi[j + 1] - 1][1] - coord[1]) + 0.5
#         #             else:
#         #                 h_li = np.abs(path_ray[-1][1] - coord[1]) + 0.5
#         #             li_l_1_total = h_li / np.abs(np.sin(rotate_angle))
#         #             li_l_1 += li_l_1_total - residue
#         #             residue = li_l_1_total
#         #
#         #         elif classes[j] == 'lo':
#         #             if j < len(posi) - 1:
#         #                 h_lo = np.abs(path_ray[posi[j + 1] - 1][1] - coord[1]) + 0.5
#         #             else:
#         #                 h_lo = np.abs(path_ray[-1][1] - coord[1]) + 0.5
#         #             lo_l_1_total = h_lo / np.abs(np.sin(rotate_angle))
#         #             lo_l_1 += lo_l_1_total - residue
#         #             residue = lo_l_1_total
#         #         elif classes[j] == 'bu':
#         #             if j < len(posi) - 1:
#         #                 h_bu = np.abs(path_ray[posi[j + 1] - 1][1] - coord[1]) + 0.5
#         #             else:
#         #                 h_bu = np.abs(path_ray[-1][1] - coord[1]) + 0.5
#         #             bu_l_1_total = h_bu / np.abs(np.sin(rotate_angle))
#         #             bu_l_1 += bu_l_1_total - residue
#         #             residue = bu_l_1_total
#                 # pdb.set_trace()
#     return li_l_1, lo_l_1, cr_l_1,bu_l_1
#
#
# #@jit(nopython=True)
# def cal_path222_plus(path_2, coord, omega):
#
#         path_ray = path_2[0]
#         posi = path_2[1]
#         classes = path_2[2]
#
#         cr_l_2 = 0
#         lo_l_2 = 0
#         li_l_2 = 0
#         bu_l_2 = 0
#         for j, index in enumerate(posi):
#             # pdb.set_trace()
#             if  classes[j] == 'cr':
#                 if j < len(posi) - 1:
#                     x_cr = np.abs(path_ray[posi[j + 1] ][2] - path_ray[posi[j]][2])
#                     z_cr = np.abs(path_ray[posi[j + 1] ][0] - path_ray[posi[j] ][0])
#                     y_cr = np.abs(path_ray[posi[j + 1] ][1] - path_ray[posi[j] ][1])
#                 else:
#                     x_cr = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
#                     z_cr = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
#                     y_cr = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
#                 cr_l_2_total = np.sqrt( (x_cr+0.5) ** 2 + (z_cr+0.5) ** 2 + (y_cr+0.5) ** 2)
#                 cr_l_2 += cr_l_2_total
#
#             elif classes[j] == 'li':
#                 if j < len(posi) - 1:
#                     x_li = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2])
#                     z_li = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
#                     y_li = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
#                 else:
#                     x_li = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
#                     z_li = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
#                     y_li = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
#                 li_l_2_total = np.sqrt(x_li ** 2 + z_li ** 2 + y_li ** 2)
#                 li_l_2 += li_l_2_total
#
#             elif classes[j] == 'lo':
#                 if j < len(posi) - 1:
#                     x_lo = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
#                     z_lo = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
#                     y_lo = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
#                 else:
#                     x_lo = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
#                     z_lo = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
#                     y_lo = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
#                 lo_l_2_total = np.sqrt(x_lo ** 2 + z_lo ** 2 + y_lo ** 2)
#                 lo_l_2 += lo_l_2_total
#             elif classes[j] == 'bu':
#                 if j < len(posi) - 1:
#                     x_bu = np.abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j] ][2])
#                     z_bu = np.abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j] ][0])
#                     y_bu = np.abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j] ][1])
#                 else:
#                     x_bu = np.abs(path_ray[-1][2] - path_ray[posi[j] ][2])
#                     z_bu = np.abs(path_ray[-1][0] - path_ray[posi[j] ][0])
#                     y_bu = np.abs(path_ray[-1][1] - path_ray[posi[j] ][1])
#                 bu_l_2_total = np.sqrt(x_bu ** 2 + z_bu ** 2 + y_bu ** 2)
#                 bu_l_2 += bu_l_2_total
#             else:
#                 pass
#     # can add the other class path
#         return li_l_2, lo_l_2, cr_l_2,bu_l_2


#
# @jit(nopython=True)
# def cal_coord_1_anti(angle,coord,face_1,shape,label_list):
#     """
#
#     :param angle:
#     :param coord:
#     :param face_1:
#     :param shape:
#     :param label_list:
#     :return:
#     """
#
#     z,y,x =coord
#     z_max,y_max,x_max=shape
#
#     path_1 =[]
#     classes_posi = [0]
#     classes = ['cr']
#     x_max -= 1
#     y_max -= 1
#     z_max -= 1
#
#     tan = np.tan(angle)
#
#
#     if   face_1 == 'FRONTZY':
#         for increment in range(x_max-x+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             if angle < np.pi /2:
#                 new_x = np.round(x + increment * 1)  # this -1 represents that the opposition of direction
#                 # between the lab x-axis and the wavevector
#                 new_y = np.round(y + increment * tan * -1)
#                 new_z = z
#             else:
#                 new_x =np.floor( x + increment*1) # this -1 represents that the opposition of direction
#                                       # between the lab x-axis and the wavevector
#                 new_y =np.floor( y + increment * tan*-1)
#                 new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord] == 0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == 'BOTZX':
#         for increment in range(y_max-y+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = np.round(x + -1* increment/tan) # this -1 represents that the opposition of direction
#                                              # between the lab x-axis and the wavevector
#             new_y = np.round(y + increment*1)
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord]==0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == 'BACKZY':
#         for increment in range(x-0+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = x + increment*-1 # this -1 represents that the opposition of direction
#                                              # between the lab x-axis and the wavevector
#             if angle<np.pi:
#                 new_y = np.floor(y + increment*tan*1)
#             else:
#                 new_y = np.round(y + increment * tan * 1)
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord]==0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == 'TOPZX':
#         for increment in range(y - 0 + 1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = np.floor(x + increment * 1 / (tan))  # this -1 represents that the opposition of direction
#             # between the lab x-axis and the wavevector
#             new_y = y + increment * -1
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int((new_z)),
#                                int((new_y)),
#                                int((new_x)))
#             if label_list[potential_coord] == 0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#         # pdb.set_trace()
#     else:
#         raise RuntimeError("unexpected ray out face")
#
#     return path_1,classes_posi,classes
#
# @jit(nopython=True)
# def cal_coord_1(angle,coord,face_1,shape,label_list):
#     """
#
#     :param angle:
#     :param coord:
#     :param face_1:
#     :param shape:
#     :param label_list:
#     :return:
#     """
#
#     z,y,x =coord
#     z_max,y_max,x_max=shape
#
#     path_1 =[]
#     classes_posi = [0]
#     classes = ['cr']
#     x_max -= 1
#     y_max -= 1
#     z_max -= 1
#
#     tan = np.tan(angle)
#
#     if   face_1 == '1_FRONT':
#         for increment in range(x_max-x+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             if angle < np.pi /2:
#                 new_x = np.round(x + increment * 1)  # this -1 represents that the opposition of direction
#                 # between the lab x-axis and the wavevector
#                 new_y = np.round(y + increment * tan * 1)
#                 new_z = z
#             else:
#                 new_x =np.floor( x + increment*1) # this -1 represents that the opposition of direction
#                                       # between the lab x-axis and the wavevector
#                 new_y =np.floor( y + increment * tan*-1)
#                 new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord] == 0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == '1_bot':
#         for increment in range(y_max-y+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = np.round(x + 1* increment/tan) # this -1 represents that the opposition of direction
#                                              # between the lab x-axis and the wavevector
#             new_y = np.round(y + increment*1)
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord]==0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == '1_back':
#         for increment in range(x-0+1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = x + increment*-1 # this -1 represents that the opposition of direction
#                                              # between the lab x-axis and the wavevector
#             if angle<np.pi:
#                 new_y = np.floor(y + increment* -tan*1)
#             else:
#                 new_y = np.round(y + increment * tan * 1)
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int(  (new_z)),
#                            int(  (new_y)),
#                            int(  (new_x)))
#             if label_list[potential_coord]==0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#     elif face_1 == '1_top':
#         for increment in range(y - 0 + 1):
#             # the absorption also count that coordinate in the  path_2\
#             # decrement on x axis
#             new_x = np.floor(x + increment * 1 / (tan))  # this -1 represents that the opposition of direction
#             # between the lab x-axis and the wavevector
#             new_y = y + increment * -1
#             new_z = z
#
#             if new_y >= y_max:
#                 new_y = y_max - 1
#             elif new_y < 0:
#                 new_y = 0
#
#             if new_x >= x_max:
#                 new_x = x_max - 1
#             elif new_x < 0:
#                 new_x = 0
#
#             if new_z >= z_max:
#                 new_z = z_max - 1
#             elif new_z < 0:
#                 new_z = 0
#
#             potential_coord = (int((new_z)),
#                                int((new_y)),
#                                int((new_x)))
#             if label_list[potential_coord] == 0:
#                 break
#
#             if increment == 0:
#                 pass
#             elif label_list[potential_coord] != label_list[path_1[increment - 1]]:
#                 if label_list[potential_coord] == 1:
#                     classes.append('li')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 2:
#                     classes.append('lo')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 3:
#                     classes.append('cr')
#                     classes_posi.append(increment)
#                 elif label_list[potential_coord] == 4:
#                     classes.append('bu')
#                     classes_posi.append(increment)
#                 else:
#                     raise RuntimeError('unexpected classes')
#
#             path_1.append(potential_coord)
#         # pdb.set_trace()
#     else:
#         raise RuntimeError("unexpected ray out face")
#
#     return path_1,classes_posi,classes

# def which_face_1(coord, shape, rotation_frame_angle):
#     """
#
#     :param coord:
#     :param shape:
#     :param rotation_frame_angle:
#     :return:
#     """
#     """
#      the detector and the x-ray clockwise rotation is positive
#     """
#     # it rotates clockwisely about the gonimeter positive x-axis in dials (z-axis in mine), right hand rule
#     assert rotation_frame_angle >= 0
#     z_max, y_max, x_max = shape
#     z, y, x = coord
#     omega_front = np.arctan((y_max - y) / (x_max - x + 0.0001))
#     omega_bot = np.arctan((y_max - y) / (x - 0 + 0.0001)) + np.pi * 0.5
#     omega_back = np.arctan((y - 0) / (x - 0 + 0.0001)) + np.pi
#     omega_top = np.arctan((y-0) / (x_max - x+ 0.0001)) + np.pi * 1.5
#
#
#
#     if rotation_frame_angle <= omega_front:
#         face_1 = '1_FRONT'
#     elif rotation_frame_angle < omega_bot and rotation_frame_angle > omega_front:
#         face_1 = '1_bot'
#     elif rotation_frame_angle < omega_back and rotation_frame_angle > omega_bot:
#         face_1 = '1_back'
#     elif rotation_frame_angle < omega_top and rotation_frame_angle > omega_back:
#         face_1 = '1_top'
#     elif rotation_frame_angle <= 2 * np.pi and rotation_frame_angle > omega_top:
#         face_1 = '1_FRONT'
#     else:
#         raise RuntimeError('rotataion angle is larger 360 degree \n'
#                            'the angle is {:4f}, please check'.format(rotation_frame_angle * 180 / np.pi))
#     return face_1

# def extract_row_column(coord_1):
#     row=[]
#     col=[]
#     for coord in coord_1:
#         z,y,x = coord
#         row.append(y)
#         col.append(x)
#     return np.array(row), np.array(col)
#
# def end_point(y,x,theta,shape):
#
#     z_max, y_max, x_max = shape
#
#     if theta >= 90:
#         the = (180-theta)/180 * np.pi
#         x_diff = x_max - x
#
#         height = x_diff * np.tan(the)
#         if height > y:
#             x_end = y / (np.tan(the)+0.001) + x
#             y_end = 0
#         else:
#             y_end = y - height
#             x_end = x_max - 1
#
#     elif 0 < theta < 90 :
#         the = (theta)/180 * np.pi
#         x_diff = x
#         height = x_diff * np.tan(the)
#
#         if height > y:
#             x_end = x - y / (np.tan(the)+0.001)
#             y_end = 0
#         else:
#             y_end = y - height
#             x_end = 0
#
#     elif -90 < theta <= 0 :
#         the = (-theta)/180 * np.pi
#         x_diff = x_max - x
#         height = x_diff * np.tan(the)
#         if height > y_max - y:
#             x_end = x - y / (np.tan(the)+0.001)
#             y_end = y_max
#         else:
#             y_end = y + height
#             x_end = 0
#
#     elif  theta <= -90 :
#         the = (180+theta)/180 * np.pi
#         x_diff = x
#         height = x_diff * np.tan(the)
#         if height > y_max - y:
#             x_end = y / (np.tan(the)+0.001) + x
#             y_end = y_max
#         else:
#             y_end = y + height
#             x_end = x_max
#
#     return  y_end,x_end
#
# def mask2rgb(mask, COLOR=None):
#     """
#
#     :param mask: input mask to be converted to rgb image
#     :param  COLOR: usd skimage:rgb 1:liquor: red; 2: loop, green ; 3: crystal,blue
#     :return: bgr image
#     """
#     if COLOR is None:
#         COLOR = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255],4: [255, 255, 0]}
#
#     rgb=np.zeros(mask.shape+(3,),dtype=np.uint8)
#
#     for i in np.unique(mask):
#         rgb[mask==i]=COLOR[i]
#
#     return rgb
#
# def plot_slices_1(coord,path_1,label_list,omega):
#     z, y, x = coord
#     shape = label_list.shape
#     y_end, x_end = end_point(y, x, omega, shape)
#     coord_1 =path_1[0]
#     row_1,col_1 = extract_row_column(coord_1)
#     row, col = line(y, x, int(np.round(y_end)),int(np.round(x_end)))
#     slice = label_list[z, :, :]
#     slice = mask2rgb(slice)
#     # pdb.set_trace()
#     # slice[row,col] = [255,255,255]
#     plt.imshow(slice)
#     plt.plot(col, row, "--w")
#     plt.plot(col_1, row_1, "--m")
#     plt.axis('off')
#     plt.show()
#
# def theta_calculation(coord,x_d,y_d,distance):
#     z,y,x=coord
#     theta=np.arctan((y_d-y)/(x+distance))
#     return theta
#
# def phi_calculation(coord,x_d,y_d,distance):
#     z,y,x=coord
#     phi=np.arctan((x_d-z)/(x+distance))
#     return phi
#
# def theta_central_calculation(detector_pixel, panel_origin, lab_origin,
#                               fast_axis,slow_axis, pixel_size_detector_X, pixel_size_detector_Y):
#     # only calculate the origin to the a pixel on the detector theta angle
#     # only calculate the angle so no rotational matrice is needed
#     # but here the x,z of labe are the z,x of mine, y is the same
#     pdy,pdx = detector_pixel
#     mmx,mmy,mmz = panel_origin + slow_axis * pdy * pixel_size_detector_Y + fast_axis   * pdx *pixel_size_detector_X
#     # z,y,x = lab_origin
#
#     #the detect
#     if mmz < 0:
#         theta= -np.arctan(mmy / mmz)
#     elif  mmy>0:
#         assert  np.arctan(mmy / mmz) > 0
#         theta =  np.pi - np.arctan(mmy / mmz)
#     else:
#         assert  mmy<0, np.arctan(mmy / mmz)<0
#         theta = - np.pi - np.arctan(mmy / mmz)
#     # pdb.set_trace()
#     return theta,mmx,mmy,mmz
#
# def phi_central_calculation(detector_pixel, panel_origin, lab_origin,
#                             fast_axis,slow_axis, pixel_size_detector_X,pixel_size_detector_Y):
#     # only calculate the origin to the a pixel on the detector phi angle
#     # only calculate the angle so no rotational matrice is needed
#     pdy,pdx = detector_pixel
#     mmx,mmy,mmz = panel_origin +  slow_axis * pdy * pixel_size_detector_Y + fast_axis * pdx *pixel_size_detector_X
#     # z,y,x = lab_origin
#     phi= np.arctan(mmx / np.abs(mmz)) # the phi can't be larger than 90 degree so not take into account
#
#     return phi,mmx,mmy,mmz



#
# def dials_2_thetaphi_2(rotated_s1,L1=False):
#     if L1 is True:
#         # L1 is the incident beam and L2 is the diffracted so they are opposite
#         rotated_s1 = -rotated_s1
#
#     if rotated_s1[1] == 0:
#         # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#         theta = np.arctan(-rotated_s1[2] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2) + 0.001))
#         # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#         phi = np.arctan(rotated_s1[0] / (rotated_s1[1] + 0.001))
#     else:
#         if rotated_s1[1] < 0:
#             theta = np.arctan(-rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = np.arctan(rotated_s1[0] / (rotated_s1[1]))
#         else:
#             if rotated_s1[2] < 0:
#                 theta = np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#
#             else:
#                 theta = - np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = - np.arctan(rotated_s1[0] / (-rotated_s1[1]))  # tan-1(-z/-x)
#     return theta,phi
#
# def dials_2_thetaphi_11(rotated_s1,L1=False):
#     if L1 is True:
#         # L1 is the incident beam and L2 is the diffracted so they are opposite
#         rotated_s1 = -rotated_s1
#
#     if rotated_s1[1] == 0:
#         # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#         theta = np.arctan(-rotated_s1[2] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2) + 0.001))
#         # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#         phi = np.arctan(rotated_s1[0] / (-rotated_s1[1] + 0.001))
#     else:
#         if rotated_s1[1] < 0:
#             theta = np.arctan(-rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = np.arctan(rotated_s1[0] / (-rotated_s1[1]))
#         else:
#             if rotated_s1[2] < 0:
#                 theta = np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#
#             else:
#                 theta = - np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = - np.arctan(rotated_s1[0] / (rotated_s1[1]))  # tan-1(-z/-x)
#     return theta,phi
#
# def dials_2_thetaphi_1(rotated_s1,L1=False):
#     if L1 is True:
#         # L1 is the incident beam and L2 is the diffracted so they are opposite
#         rotated_s1 = -rotated_s1
#
#     if rotated_s1[1] == 0:
#         # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#         theta = np.arctan(-rotated_s1[2] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2) + 0.001))
#         # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#         phi = np.arctan(-rotated_s1[0] / (-rotated_s1[1] + 0.001))
#     else:
#         if rotated_s1[1] < 0:
#             theta = np.arctan(-rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = np.arctan(-rotated_s1[0] / (-rotated_s1[1]))
#         else:
#             if rotated_s1[2] < 0:
#                 theta = np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#
#             else:
#                 theta = - np.pi - np.arctan(
#                     -rotated_s1[2] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[1] ** 2))  # tan-1(y/-x)
#             phi = - np.arctan(-rotated_s1[0] / (rotated_s1[1]))  # tan-1(-z/-x)
#     return theta,phi
#
# def dials_2_thetaphi(rotated_s1,L1=False):
#     if L1 is True:
#         # L1 is the incident beam and L2 is the diffracted so they are opposite
#         rotated_s1 = -rotated_s1
#
#     if rotated_s1[2] == 0:
#         # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#         theta = np.arctan(rotated_s1[1] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2) + 0.001))
#         # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#         phi = np.arctan(rotated_s1[0] / (rotated_s1[2] + 0.001))
#     else:
#         if rotated_s1[2] < 0:
#             theta = np.arctan(rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#             phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))
#         else:
#             if rotated_s1[1] > 0:
#                 theta = np.pi - np.arctan(
#                     rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#
#             else:
#                 theta = - np.pi - np.arctan(
#                     rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#             phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))  # tan-1(-z/-x)
#     return theta,phi
#
#
# def which_face_1_anti(coord, shape,rotation_frame_angle):
#     """
#
#     :param coord:
#     :param shape:
#     :param rotation_frame_angle:
#     :return:
#     """
#     """
#      the detector and the x-ray anti-clockwise rotation is positive
#     """
#     # sample rotates clockwisely about the gonimeter positive x-axis in dials (z-axis in mine), right hand rule
#     # so the x-ray and the detector should rotate anti-clockwisely
#     assert  rotation_frame_angle >= 0
#     z_max, y_max, x_max = shape
#     z,y,x =coord
#     omega_front= np.arctan((y-0)/(x_max-x+0.0001))
#     omega_top = np.arctan((y - 0) / ( x-0 + 0.0001)) + np.pi * 0.5
#     omega_back = np.arctan((y_max-y)/(x-0+0.0001)) + np.pi
#     omega_bot = np.arctan((y_max-y)/(x_max-x+0.0001)) + np.pi * 1.5
#
#     if rotation_frame_angle< omega_front:
#         face_1='FRONTZY'
#     elif rotation_frame_angle < omega_top and rotation_frame_angle > omega_front:
#         face_1 = 'TOPZX'
#     elif rotation_frame_angle <omega_back and rotation_frame_angle > omega_top:
#         face_1 = 'BACKZY'
#     elif rotation_frame_angle <omega_bot and rotation_frame_angle > omega_back:
#         face_1 = 'BOTZX'
#     elif rotation_frame_angle <= 2*np.pi and rotation_frame_angle > omega_bot:
#         face_1 = 'FRONTZY'
#     else:
#         pass
#         #raise  RuntimeError('rotataion angle is larger 360 degree \n'
#                             # 'the angle is {:4f}, please check'.format(rotation_frame_angle*180/np.pi))
#     return  face_1

#
# def fill_ab_map(data,name):
#     y, x = np.nonzero(data)
#     index = 0
#
#     # iteration x times
#     for i, xx in enumerate(x):
#         try:
#             index = xx + 1
#             data[y[i]][index:x[i + 1]] = data[y[i]][xx]
#             if xx == x[-1]:  # if x=2400, the last point in a y value but not the last point
#                 data[y[i]][index:-1] = data[y[i]][xx]
#         except:
#             data[y[i]][index:-1] = data[y[i]][xx]
#
#     data = np.transpose(data, [1, 0])
#     for j in range(data.shape[0]):
#         y_list = np.unique(y)
#         for k, yy in enumerate(y_list):
#             try:
#                 # index = yy+1
#                 data[j][yy:y_list[k + 1] - 1] = data[j][yy]
#             except:
#                 data[j][yy:] = data[j][yy]
#                 # data[j][-1] = data[j][yy]
#
#     data = np.transpose(data, [1, 0])
#
#     fig = plt.figure(figsize=(16, 8))
#     fig.set_figwidth(16)
#     plt.imshow(data)
#     plt.colorbar()
#     plt.savefig('{}.png'.format(name))
#     plt.show()
#
# def arrayresize(data, output_voxel_size=(128,128,128)):
#     #https://discuss.pytorch.org/t/pytorch-resize-3d-numpy-array/70338/4
#     import torch
#     import torch.nn.functional as F
#
#
#     data = data.astype(np.float32)
#     data = torch.from_numpy(data)
#     data =torch.unsqueeze(torch.unsqueeze(data,dim=0),dim=0)
#
#     if len(data.shape) !=  5:
#         raise  RuntimeError('the input array should be in N,C,H,W or N,C,D,H,W')
#     # N, C, D,H, W = data.shape
#
#     z,y,x =output_voxel_size
#     d = torch.linspace(-1, 1, z)
#     h = torch.linspace(-1, 1, y)
#     w = torch.linspace(-1, 1, z)
#     meshx, meshy, meshz = torch.meshgrid((d, h, w))
#     grid = torch.stack((meshx, meshy, meshz), 3)
#     grid = grid.unsqueeze(0)
#     out = F.grid_sample(data, grid, align_corners=True)
#     out =torch.squeeze(torch.squeeze(out,dim=0),dim=0)
#     npout = out.numpy()
#     return  npout
#
# def cal_path2(path_2,coord,label_list,rate_list):
#     # Pythagorean theorem
#
#
#     li_list_2=[]
#     lo_list_2=[]
#     cr_list_2=[]
#
#     for i in path_2:
#         if label_list[i] == rate_list['li']:
#             li_list_2.append(i)
#         elif label_list[i] ==rate_list['lo']:
#             lo_list_2.append(i)
#         elif label_list[i] ==rate_list['cr']:
#             cr_list_2.append(i)
#         else:
#             pass
#
#     if len(cr_list_2) > 0:
#         cr_l_2=np.sqrt( (cr_list_2[-1][0] - coord[0]) ** 2 +
#                         (cr_list_2[-1][1] - coord[1]) ** 2 +
#                         (cr_list_2[-1][2] - coord[2]) ** 2) + 0.5
#
#     else:
#         cr_l_2=0
#
#     if len(lo_list_2) > 0:
#         lo_l_2=np.sqrt( (lo_list_2[-1][0] - lo_list_2[0][0]) ** 2 +
#                         (lo_list_2[-1][1] - lo_list_2[0][1]) ** 2 +
#                         (lo_list_2[-1][2] - lo_list_2[0][2]) ** 2) + 0.5
#     else:
#         lo_l_2=0
#
#     if len(li_list_2) > 0:
#         li_l_2=np.sqrt( (li_list_2[-1][0] - li_list_2[0][0]) ** 2 +
#                         (li_list_2[-1][1] - li_list_2[0][1]) ** 2 +
#                         (li_list_2[-1][2] - li_list_2[0][2]) ** 2) + 0.5
#     else:
#         li_l_2=0
#
#     return li_l_2,lo_l_2,cr_l_2

# if __name__ == '__main__':
#     name='test'
#     data = np.load('{}.npy'.format(name))
#     fill_ab_map(data,name)