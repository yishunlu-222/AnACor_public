import numpy as np
import pdb
# import matplotlib.pyplot as plt
from numba import jit


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

    increment_ratio_x = -1
    increment_ratio_y = np.tan(np.pi-theta) / np.cos(np.abs(phi))
    increment_ratio_z = np.tan(phi)

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

# @jit(nopython=True)
# def difference_length(start,end,voxel_size):
#         z1,y1,x1=start
#         z2,y2,x2=end
#         z_voxel_size, y_voxel_size, x_voxel_size = voxel_size
#         length = np.sqrt( ((x2-x1+0.5)*x_voxel_size ) ** 2 +
#                           ((z2-z1+0.5)*z_voxel_size) ** 2 + 
#                           ((y2-y1+0.5)*y_voxel_size) ** 2)

#         return length

def difference_length(start,end,voxel_size):
        z1,y1,x1=start
        z2,y2,x2=end
        z_voxel_size, y_voxel_size, x_voxel_size = voxel_size
        length = np.sqrt( ((x2-x1)*x_voxel_size ) ** 2 +
                          ((z2-z1)*z_voxel_size) ** 2 + 
                          ((y2-y1)*y_voxel_size) ** 2)

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




# @jit(nopython=True)
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
        # print(crystalDifference)
        # print(CrystalMiddle)
        # print(CrystalLongest
        #         )
        # print(CrystalShortest)

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
        increment_ratio_x = -1
        increment_ratio_y = np.tan(np.pi-theta) / np.cos(np.abs(phi))
        increment_ratio_z = np.tan(phi)

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
    
    if np.abs(np.array(potential_coord)-np.array(air_outermost_potential_coord)).mean() < 1 :
        pass
    else:

        LoopLongest = AirMiddle_outer
        LoopShortest = LoopMiddle
        potential_coord_2, LoopMiddle,counter = bisection(counter, LoopLongest, LoopShortest, resolution, label_list,
                                                    increment_ratios, coord, boundary='outer', cls=2)
        if np.abs(np.array(potential_coord_2)-np.array(potential_coord)).mean() < 2:
            pass
        else:
            classes.append('lo_inner')
            path_2.append(potential_coord)
            classes.append('lo_outer')
            path_2.append(potential_coord_2)

    # starting from the crystal to calculate the classes along the path to find the BUBBLE
    BubbleLongest = AirMiddle_outer
    BubbleShortest = CrystalMiddle
    potential_coord, BubbleMiddle,counter = bisection(counter, BubbleLongest, BubbleShortest, resolution, label_list,
                                                  increment_ratios, coord, boundary='inner', cls=4)
    if np.abs(np.array(potential_coord)-np.array(air_outermost_potential_coord)).mean() < 1:
        pass
    else:

        BubbleLongest = AirMiddle_outer
        BubbleShortest = BubbleMiddle
        potential_coord_2, BubbleMiddle,counter = bisection(counter, BubbleLongest, BubbleShortest, resolution, label_list,
                                                      increment_ratios, coord, boundary='outer', cls=4)
        if np.abs(np.array(potential_coord_2)-np.array(potential_coord)).mean() < 2:
            pass
        else:
            classes.append('bu_inner')
            path_2.append(potential_coord)
            classes.append('bu_outer')
            path_2.append(potential_coord_2)

    # # starting from the crystal to calculate the classes along the path to find the other possible air
    # Air2Longest = AirMiddle_outer
    # Air2Shortest = CrystalMiddle
    # potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
    #                                             increment_ratios, coord, boundary='inner', cls=0)
    # if average(potential_coord,air_outermost_potential_coord)< 1:
    #     pass
    # else:
    #     classes.append('air_inner')
    #     path_2.append(potential_coord)
    #     Air2Longest = AirMiddle_outer
    #     Air2Shortest = Air2Middle
    #     potential_coord, Air2Middle = bisection(Air2Longest, Air2Shortest, resolution, label_list,
    #                                                 increment_ratios, coord, boundary='outer', cls=0)
    #     classes.append('air_outer')
    #     path_2.append(potential_coord)


    return (path_2, classes),counter


