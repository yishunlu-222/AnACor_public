import numpy as np
import pdb
import matplotlib.pyplot as plt
from numba import jit
from skimage.draw import line
# import math
import ctypes as ct
from skimage import morphology, filters
from skimage.segmentation import watershed
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans,MiniBatchKMeans
from scipy.spatial import cKDTree,distance
from sklearn.decomposition import PCA
import time
np.set_printoptions(suppress=True)

np.random.seed(0)

def create_probability_distribution(coords, centroid):

    dists = distance.cdist(coords, centroid.reshape(1, -1), 'euclidean').flatten()
    # Inverse distances as probabilities (closer points have higher probability)
    probabilities = 1 / (1 + dists)
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    return probabilities

def generate_sampling(label_list, cr=3, dim='z', sampling_size=5000, auto=True,method='even',sampling_ratio=None,kmeans_cluster=1000):
    """
    Probability Sampling: Every member of the population has a known (non-zero) probability of being selected. This includes:
    ###
    Simple Random Sampling: As mentioned above, this involves selecting random points in a completely unbiased way where every member of the dataset has an equal chance of getting selected. This is the default behaviour of the numpy's random.choice() function.
    
    ###
    Stratified Sampling: This involves dividing the population into homogeneous subgroups (or "strata") and then taking a simple random sample from each subgroup.
    Stratified sampling is a method of sampling that involves dividing a population into homogeneous subgroups known as strata, and then selecting a simple random sample from each stratum. In this case, the different values in your distribution can be thought of as different strata.

    Given that your objective is to generate a set of 5000 crystal voxel coordinates, you can perform stratified sampling as follows:

    Create strata: Divide your distribution into different strata, where each stratum corresponds to a range of voxel values or even single voxel values, depending on the distribution of voxel values in your data. This division should reflect the structure of your distribution, and there should be enough voxels in each stratum to make reasonable estimates.

    Allocate samples to strata: The next step is to decide how many samples to draw from each stratum. In proportional allocation, you would choose a number of samples from each stratum proportional to the size of the stratum in relation to the total population.

    Sample from strata: Now, for each stratum, you need to randomly select the required number of samples. The coordinates of these samples represent the coordinates of your crystal voxels.
    
    ###
    Cluster Sampling: The population is divided into clusters (groups) and a set of clusters are chosen at random. All observations in the selected clusters form the sample.

    Systematic Sampling: Involves selecting items from an ordered population using a skip or an interval. For example, you might sample every 10th item from your dataset.
    """
    zz, yy, xx = np.where(label_list == cr)  
    crystal_coordinate = np.stack((zz,yy,xx),axis=1)
    
    # Find the indices of the non-zero elements directly
    if sampling_ratio is not None:
        sampling=int(len(crystal_coordinate)*sampling_ratio/100) # sampling ratio in %
        sampling_size=1/sampling_ratio*100
    else:
        # if auto:
        #     # When sampling ~= N/2000, the results become stable
        #     sampling = len(crystal_coordinate) // 2000
            
        # else:
            if len(crystal_coordinate) < sampling_size:
                sampling = len(crystal_coordinate)
            else:
                sampling = len(crystal_coordinate) // sampling_size
   
    print(" The sampling number is {}".format(sampling))
    if method =='even':
        coord_list_even=[]
        # zz, yy, xx = np.where(label_list == rate_list['cr'])  # this line occupies 1GB, why???
        # #crystal_coordinate = zip(zz, yy, xx)  # can be not listise to lower memory usage
        # crystal_coordinate = np.stack((zz,yy,xx),axis=1)
        # seg = int(np.round(len(crystal_coordinate) / sampling))
        seg = sampling

        coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
        print(" {} voxels in even sampling are calculated".format(len(coordinate_list)))
        for i in coordinate_list:
            coord_list_even.append(crystal_coordinate[i])
       
        coord_list =  np.array(coord_list_even)
    
    elif method =='slice':
        output_lengths = []
        if dim == 'z':
            index = 0

        elif dim == 'y':
            index = 1

        elif dim == 'x':
            index = 2
        zz_u = np.unique(crystal_coordinate[:, index])

        # Sort the crystal_coordinate array using the np.argsort() function
        sorted_indices = np.argsort(crystal_coordinate[:, index])
        crystal_coordinate = crystal_coordinate[sorted_indices]
        # total_size=len(crystal_coordinate)

        # Use np.bincount() to count the number of occurrences of each value in the array
        output_lengths = np.bincount(
            crystal_coordinate[:, index], minlength=len(zz_u))
        zz_u = np.insert(zz_u, 0, np.zeros(len(output_lengths)-len(zz_u)))
        # Compute the sampling distribution
        if sampling / len(output_lengths) < 0.5:
            sorted_indices = np.argsort(output_lengths)[::-1]  # descending order
            sampling_distribution = np.zeros(len(output_lengths))
            sampling_distribution[sorted_indices[:sampling]] = 1
            sampling_distribution=sampling_distribution.astype(int)
        else:
            sampling_distribution = np.round(
                output_lengths / output_lengths.mean() * sampling / len(output_lengths)).astype(int)

        coord_list = []
    
        # Use boolean indexing to filter the output array based on the sampling distribution
        for i, sampling_num in enumerate(sampling_distribution):
            if sampling_num == 0:
                continue
            # output_layer = crystal_coordinate[crystal_coordinate[:, index] == zz_u[i]]
            # Use np.random.choice() to randomly sample elements from the output arrays
            before = output_lengths[:i].sum()
            after = output_lengths[:i + 1].sum()
            output_layer = crystal_coordinate[before: after]
            numbers = []
            # for k in range(sampling_num):

            #     numbers.append(int(output_lengths[i]/(sampling_num+1) * (k+1)))

            # for num in numbers:
            #     coord_list.append(output_layer[num])
            sampled_indices = np.random.choice(range(len(output_layer)), size=int(sampling_num), replace=False)
            coord_list.extend(output_layer[sampled_indices])
            
        print(" {} voxels in slice sampling are calculated".format(len(coord_list)))
        coord_list= np.array(coord_list)
        
    elif method =='random':
        # Random sampling
        coord_list_random=[]
        arr = np.array(range(len(crystal_coordinate)))
       
        samples =np.sort( np.random.choice(arr, sampling, replace=False))
        for i in samples:
            coord_list_random.append(crystal_coordinate[i])
        print(" {} voxels in random sampling are calculated".format(len(samples)))
        coord_list= np.array(coord_list_random)
    elif method =='evenrandom':
        # Random sampling
        coord_list_even = []
        interval_length =int(sampling_size)
        random_indices = [np.random.randint(i, i + interval_length) for i in range(0, len(crystal_coordinate), interval_length)]

        # Ensure the number of indices doesn't exceed the sampling value
        random_indices = random_indices[:sampling]

        print(" {} voxels in evenrandom sampling with randomness are calculated".format(len(random_indices)))

        for i in random_indices:
            coord_list_even.append(crystal_coordinate[i])

        coord_list = np.array(coord_list_even)
    elif method =='stratified':
        #https://scikit-learn.org/stable/modules/clustering.html

        from sklearn.cluster import AgglomerativeClustering,BisectingKMeans
        # model = AgglomerativeClustering(n_clusters=sampling)
        
        coordinate_list = np.linspace(0, len(crystal_coordinate), num=sampling, endpoint=False, dtype=int)
        # print(" {} voxels in even sampling are calculated".format(len(coordinate_list)))
        init_list=[]
        for i in coordinate_list:
            init_list.append(crystal_coordinate[i])

        batch_size= 2 ** (np.round(np.log2(1/sampling_ratio)).astype(int))
        init_list =  np.array(init_list)

        clustering='kmeans'
        if clustering=='kmeans':
            pca = PCA(n_components=3)
            transformed_coordinates = pca.fit_transform(crystal_coordinate)
            model = MiniBatchKMeans(n_clusters=sampling, batch_size=batch_size, init='k-means++',verbose=0).fit(transformed_coordinates)
            
            region_centroids  = pca.inverse_transform(model.cluster_centers_)
            # original_space_centroids = np.around(original_space_centroids).astype(int)
            # for coords in region_centroids:
            #     z, y, x = coords
            #     yes.append(label_list[z, y, x])
            # coord_list = np.array([coords for coords, label in zip(region_centroids, yes) if label == 3])
            # pdb.set_trace()
        elif clustering=='bisection':
            model = BisectingKMeans(n_clusters=sampling,init='random').fit(crystal_coordinate)

        print("Kmeans straified is applied")
        t1 = time.time()

        # kmeans = MiniBatchKMeans(n_clusters=sampling, batch_size=batch_size, n_init=10,init='k-means++',verbose=0).fit(crystal_coordinate)       
        labels = model.labels_
        # region_centroids = model.cluster_centers_
        region_centroids = np.around(region_centroids).astype(int)
        yes=[]
        for coords in region_centroids:
            z, y, x = coords
            yes.append(label_list[z, y, x])
        coord_list = np.array([coords for coords, label in zip(region_centroids, yes) if label == 3])
        t2=time.time()


        print("time for building kmeans is {}".format(t2-t1))
        t3=time.time()
        print("time for finding closest coordinate is {}".format(t3-t2))
        # pdb.set_trace()
    #     else:
    #         kmeans = MiniBatchKMeans(n_clusters=kmeans_cluster, batch_size=1024*8, n_init=10,init='random',verbose=0).fit(crystal_coordinate)       
    #         labels = kmeans.labels_
    #         region_centroids = kmeans.cluster_centers_
    #         region_centroids = np.around(region_centroids).astype(int)
    #         # clustered_coordinates=[]
    #         # for i in range(sampling):
    #         #     cluster_coords = crystal_coordinate[labels == i]
    #         #     clustered_coordinates.append(cluster_coords)
    #         # combined_3d_array = np.array(clustered_coordinates, dtype=object)
            
            
    #         sampled_points = []
    #         n_samples_per_cluster = sampling//kmeans_cluster
    #         remaining_samples = sampling - kmeans_cluster * n_samples_per_cluster
    #         for i in range(kmeans_cluster):
    #             cluster_points = crystal_coordinate[labels == i]
    #             probabilities = create_probability_distribution(cluster_points, region_centroids[i])
    #             sampled_indices = np.random.choice(len(cluster_points), n_samples_per_cluster, replace=False, p=probabilities)
    #             sampled_cluster_points = cluster_points[sampled_indices]
    #             sampled_points.extend(sampled_cluster_points)
                
    #         coords_set = set(map(tuple, crystal_coordinate))
    #         sampled_set = set(map(tuple, sampled_points))
    #         remaining_coords = np.array(list(coords_set - sampled_set))

    #         overall_centroid = np.mean(crystal_coordinate, axis=0)
    #         overall_probabilities = create_probability_distribution(remaining_coords, overall_centroid)
    #         remaining_indices = np.random.choice(len(remaining_coords), size=remaining_samples, replace=False)
    #         sampled_points.extend(remaining_coords[remaining_indices])
    #         coord_list = np.array(sampled_points)
    #         coord_list = np.sort(coord_list, axis=0)
        


    else:

        raise RuntimeError(f"The sampling method {method} is not defined")
    # pdb.set_trace()
    return coord_list




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


# def myframe_2_thetaphi(vector, L1=False):
#     if L1 is True:
#         # L1 is the incident beam and L2 is the diffracted so they are opposite
#         vector = -vector
#     z,y,x=vector
#     # y=-y # the y axis is opposite to the lab frame
#     # x=-x 
#     if z == 0 and x == 0:
#         theta = np.pi / 2
#         if y > 0:
#             theta *= -1

#         return theta,0

#     # Compute theta
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arcsin(np.abs(y / r)  # This gives theta in the range of 0 to pi

#     # Adjusting the range of theta to be from -pi to pi
#     if y > 0:
#         theta *= -1 

#     # Compute phi
#     phi = np.arctan2(z,-x)  # the x axis is opposite to the lab frame
#     if phi > np.pi / 2:
#         phi -= np.pi
#     elif phi < -np.pi / 2:
#         phi += np.pi

#     return theta, phi

def mse_diff(theta, phi, map_theta, map_phi,i):
    if (theta - map_theta) > 1e-5:
        print('i:{} theta is {}, map_theta is {}'.format(i,theta, map_theta))
    elif (phi - map_phi) > 1e-5:
        print('i: {} phi is {}, map_phi is {}'.format(i,phi, map_phi))
    else:
        print("i {} has no difference".format(i))

def thetaphi_2_myframe(theta,phi):
    z =np.cos(theta)*np.sin(phi)
    x =-np.cos(theta)*np.cos(phi)
    y =-np.sin(theta)
    return np.array([z,y,x])

def dials_2_thetaphi(rotated_s1,L1=False):
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


def which_face(coord,shape,theta,phi):
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

def dials_2_myframe(vector):
    # (x',y',z') in standard vector but in numpy (z,y,x)
    # rotate the coordinate system about x'(z in numpy) for 180
    # vector =vector.astype(np.float32)
    # numpy_2_dials_1 = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
    #                             [-np.sin(np.pi), np.cos(np.pi), 0],
    #                             [0, 0, 1]],dtype=np.float32)


    # (x',y',z') in standard vector but in numpy (z,y,x)
    # take the reflection about y'(y in numpy)
    numpy_2_dials_0 = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])
    # omega = np.pi
    # numpy_2_dials_1 =np.array([[1, 0, 0],
    #                             [0, np.cos(omega), np.sin(omega)],
    #                             [0, -np.sin(omega), np.cos(omega)]])
    # numpy_2_dials_1 = np.array([[1, 0, 0],
    #                             [0, -1, 0],
    #                             [0, 0, -1]])
    back2 = numpy_2_dials_0.dot(vector)

    return  back2

def myframe_2_dials(vector):
    # (x',y',z') in standard vector but in numpy (z,y,x)
    # rotate the coordinate system about x'(z in numpy) for 180
    # vector =vector.astype(np.float32)
    # numpy_2_dials_1 = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
    #                             [-np.sin(np.pi), np.cos(np.pi), 0],
    #                             [0, 0, 1]],dtype=np.float32)
    numpy_2_dials_1 = np.linalg.inv(np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]]))
    # omega = -np.pi
    # numpy_2_dials_1 =np.array([[1, 0, 0],
    #                             [0, np.cos(omega), np.sin(omega)],
    #                             [0, -np.sin(omega), np.cos(omega)]])
    # numpy_2_dials_2 = np.array([[0, 0, 1],
    #                             [0, 1, 0],
    #                             [1, 0, 0]]) #swap x and z
    back2 = numpy_2_dials_1.dot(vector)


    return  back2

@jit(nopython=True)
def cal_coord(theta ,phi,coord,face,shape,label_list,full_iteration=False):
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
            va_count = 0
            if full_iteration is False :
                if label_list[potential_coord] == 0 :
                    break
            # else:
            #     if label_list[potential_coord] == 0 :
            #         va_count += 1
            #     else:
            #         va_count = 0
            #     if va_count > 10:
            #         break
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
            # iterate 

    else:
        raise RuntimeError("unexpected ray out face")

    # if len(path_2) == 1:
    #     r_x_pos = np.abs( 0.5 /(np.cos(theta) * np.cos(phi)))
    #     r_y_pos= np.abs( 0.5 /(np.sin(theta)))
    #     r_z_pos = np.abs( 0.5 /(np.cos(theta) * np.sin(phi)))
    #     path_2[0]=(r_z_pos,r_y_pos,r_x_pos)

    return path_2,classes_posi,classes

@jit(nopython=True)
def cal_path_plus(path_2,voxel_size):
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
    # if len(path_2[0])==1:
    #     total_length=np.min(np.array(path_2[0]))
    #     pdb.set_trace()
    # else:

    total_length=np.sqrt(((path_ray[-1][1]  - path_ray[0][1] ) * voxel_length_y ) ** 2 +
                         ((path_ray[-1][0]  - path_ray[0][0] ) * voxel_length_z ) ** 2 +
                         ( (path_ray[-1][2]  - path_ray[0][2] ) * voxel_length_x )** 2)
    if len(path_ray) == 1:
        return 0,0,0,0
    
    proprotion=total_length/(len(path_ray)-1)
    # print(proprotion)
    for j, trans_index in enumerate(posi):

        if classes[j] == 'cr':
            if j < len(posi) - 1:
                cr_l_2 += proprotion * (posi[j+1]-posi[j])
            else:
                cr_l_2 += proprotion * (len(path_ray)- posi[j])
        elif classes[j] == 'li':
            if j < len(posi) - 1:
                li_l_2 +=  proprotion * (posi[j + 1] - posi[j]) 
                len_li = ((posi[j + 1] - posi[j]) / len(path_ray))
            else:
                li_l_2 +=  proprotion * (len(path_ray) - posi[j]) 
                len_li = ((len(path_ray) - posi[j]) / len(path_ray))
        elif classes[j] == 'lo':
            if j < len(posi) - 1:
                lo_l_2 +=  proprotion * (posi[j + 1] - posi[j]) 
            else:
                lo_l_2 +=  proprotion * (len(path_ray) - posi[j]) 
        elif classes[j] == 'bu':
            if j < len(posi) - 1:
                bu_l_2 +=  proprotion *(posi[j + 1] - posi[j]) 
            else:
                bu_l_2 +=  proprotion * (len(path_ray) - posi[j])
        else:
            pass
    # pdb.set_trace()
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
    # if len(path_2[0])==1:
    #     total_length=np.min(np.array(path_2[0]))
    #     pdb.set_trace()
    return li_l_2, lo_l_2, cr_l_2,bu_l_2


@jit(nopython=True)
def cal_rate(numbers,coefficients,exp=True ):
    mu_li, mu_lo, mu_cr,mu_bu = coefficients

    if len(numbers)==8:
        li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
    else:
        li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
        li_l_1, lo_l_1, cr_l_1, bu_l_1= 0,0,0,0
    if exp:
        absorp = np.exp(-((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) )
                    ))
    else:
        absorp = ((mu_li * (li_l_1  + li_l_2) +
                     mu_lo * (lo_l_1  + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                         mu_bu * (bu_l_1+ bu_l_2) ))
    return  absorp

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
        then the  minimum non-negative t is the normal of the face and that's what we want
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
    if L1 is True:
        ray_direction = -ray_direction
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

    if t_min == tx_min :
        # The ray intersects with the left face of the cube]
#    /*  'FRONTZY' = 1;
# *   'LEYX' = 2 ;
# *   'RIYX' = 3;
#     'TOPZX' = 4;
#     'BOTZX' = 5;
#     "BACKZY" = 6 ;
            return "BACKZY"
    elif t_min == tx_max :
        # The ray intersects with the right face of the cube

            return "FRONTZY"
    elif t_min == ty_min :
        # The ray intersects with the bottom face of the cube

            return 'TOPZX'
    elif t_min == ty_max :
        # The ray intersects with the top face of the cube

            return 'BOTZX'
    elif t_min == tz_min :
        # The ray intersects with the front face of the cube

            return 'LEYX'
    elif t_min == tz_max :
        # The ray intersects with the back face of the cube

            return 'RIYX'
    else :
        RuntimeError( 'face determination has a problem with direction {}'
                      'and position {}'.format( ray_direction , ray_origin ) )

    
    # if t_min == tx_min :
    #     # The ray intersects with the left face of the cube]
    #     if L1 is True :
    #         return "FRONTZY"
    #     else :
    #         return "BACKZY"
    # elif t_min == tx_max :
    #     # The ray intersects with the right face of the cube
    #     if L1 is True :
    #         return "BACKZY"
    #     else :
    #         return "FRONTZY"
    # elif t_min == ty_min :
    #     # The ray intersects with the bottom face of the cube
    #     if L1 is True :
    #         return 'BOTZX'
    #     else :
    #         return 'TOPZX'
    # elif t_min == ty_max :
    #     # The ray intersects with the top face of the cube
    #     if L1 is True :
    #         return 'TOPZX'
    #     else :
    #         return 'BOTZX'
    # elif t_min == tz_min :
    #     # The ray intersects with the front face of the cube
    #     if L1 is True :
    #         return 'RIYX'
    #     else :
    #         return 'LEYX'
    # elif t_min == tz_max :
    #     # The ray intersects with the back face of the cube
    #     if L1 is True :
    #         return 'LEYX'
    #     else :
    #         return 'RIYX'
    # else :
    #     RuntimeError( 'face determination has a problem with direction {}'
    #                   'and position {}'.format( ray_direction , ray_origin ) )

########## old ray-tracing  method##########
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
    #    # # the hypothesis is that all components only appear once, not repeated
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

def python_2_c_3d(label_list):
            # this is a one 1d conversion
            # z, y, x = label_list.shape
            # label_list_ctype = (ct.c_int8 * z * y * x)()
            # for i in range(z):
            #     for j in range(y):
            #         for k in range(x):
            #             label_list_ctype[i][j][k] = ct.c_int8(label_list[i][j][k])
            labelPtr = ct.POINTER(ct.c_int8)
            labelPtrPtr = ct.POINTER(labelPtr)
            labelPtrPtrPtr = ct.POINTER(labelPtrPtr)
            labelPtrCube = labelPtrPtr * label_list.shape[0]
            labelPtrMatrix = labelPtr * label_list.shape[1]
            matrix_tuple = ()
            for matrix in label_list:
                array_tuple = ()
                for row in matrix:
                    array_tuple = array_tuple + (row.ctypes.data_as(labelPtr),)
                matrix_ptr = ct.cast(labelPtrMatrix(
                    *(array_tuple)), labelPtrPtr)
                matrix_tuple = matrix_tuple + (matrix_ptr,)
            label_list_ptr = ct.cast(labelPtrCube(
                *(matrix_tuple)), labelPtrPtrPtr)
            return label_list_ptr