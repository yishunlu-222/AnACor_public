#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <stdio.h>

#include "GPU_reduction.cuh"

#define DEBUG 0

#include "timer.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)

void print_cuda_error(cudaError_t code)
{
	printf("CUDA error code: %d; string: %s;\n", (int)code, cudaGetErrorString(code));
}

__inline__ __device__ int cube_face(int64_t *ray_origin, double *ray_direction, int x_max, int y_max, int z_max, int L1)
{
	double t_min = x_max * y_max * z_max, dtemp = 0;
	int face_id = 0;

	// double tx_min = (min_x - ray_origin[2]) / ray_direction[2];
	//  dtemp = (0 - ray_origin[2]) / ray_direction[2];
	if (L1)
	{
		dtemp = -(0 - ray_origin[2]) / ray_direction[2];
	}
	else
	{
		dtemp = (0 - ray_origin[2]) / ray_direction[2];
	}
	if (dtemp >= 0)
	{
		t_min = dtemp;
		face_id = 1;
	}

	// double tx_max = (max_x - ray_origin[2]) / ray_direction[2];
	if (L1)
	{
		dtemp = -(x_max - ray_origin[2]) / ray_direction[2];
	}
	else
	{
		dtemp = (x_max - ray_origin[2]) / ray_direction[2];
	}
	// dtemp = (x_max - ray_origin[2]) / ray_direction[2];
	if (dtemp >= 0 && dtemp < t_min)
	{
		t_min = dtemp;
		face_id = 2;
	}

	// double ty_min = (min_y - ray_origin[1]) / ray_direction[1];
	//  dtemp = (0 - ray_origin[1]) / ray_direction[1];
	if (L1)
	{
		dtemp = -(0 - ray_origin[1]) / ray_direction[1];
	}
	else
	{
		dtemp = (0 - ray_origin[1]) / ray_direction[1];
	}
	if (dtemp >= 0 && dtemp < t_min)
	{
		t_min = dtemp;
		face_id = 3;
	}

	// double ty_max = (max_y - ray_origin[1]) / ray_direction[1];
	//  dtemp = (y_max - ray_origin[1]) / ray_direction[1];
	if (L1)
	{
		dtemp = -(y_max - ray_origin[1]) / ray_direction[1];
	}
	else
	{
		dtemp = (y_max - ray_origin[1]) / ray_direction[1];
	}
	if (dtemp >= 0 && dtemp < t_min)
	{
		t_min = dtemp;
		face_id = 4;
	}
	// double tz_min = (min_z - ray_origin[0]) / ray_direction[0];
	//  dtemp = (0 - ray_origin[0]) / ray_direction[0];
	if (L1)
	{
		dtemp = -(0 - ray_origin[0]) / ray_direction[0];
	}
	else
	{
		dtemp = (0 - ray_origin[0]) / ray_direction[0];
	}
	if (dtemp >= 0 && dtemp < t_min)
	{
		t_min = dtemp;
		face_id = 5;
	}

	// double tz_max = (max_z - ray_origin[0]) / ray_direction[0];
	//  dtemp = (z_max - ray_origin[0]) / ray_direction[0];
	if (L1)
	{
		dtemp = -(z_max - ray_origin[0]) / ray_direction[0];
	}
	else
	{
		dtemp = (z_max - ray_origin[0]) / ray_direction[0];
	}
	if (dtemp >= 0 && dtemp < t_min)
	{
		t_min = dtemp;
		face_id = 6;
	}

	if (face_id == 1)
	{ // tx_min
		return 6;
	}
	else if (face_id == 2)
	{ // tx_max
		return 1;
	}
	else if (face_id == 3)
	{ // 3 ty_min
		return 4;
	}
	else if (face_id == 4)
	{ // 4 ty_max
		return 5;
	}
	else if (face_id == 5)
	{ // 5 tz_min
		return 2;
	}
	else if (face_id == 6)
	{ // 6 tz_max
		return 3;
	}
	else
	{
		return 0;
	}
}

__global__ void rt_gpu_get_face(int *d_face, int64_t *d_coord_list, double *d_rotated_s1, double *d_xray, int x_max, int y_max, int z_max, int len_coord_list)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1);
	// printf("d_xray %f %f %f\n", d_xray[0], d_xray[1], d_xray[2]);
	// printf("d_rotated_s1 %f %f %f\n", d_rotated_s1[0], d_rotated_s1[1], d_rotated_s1[2]);
	int64_t coord[3];
	double ray_direction[3];

	if (pos < len_coord_list)
	{
		coord[0] = d_coord_list[3 * pos + 0];
		coord[1] = d_coord_list[3 * pos + 1];
		coord[2] = d_coord_list[3 * pos + 2];

		if (is_ray_incomming == 1)
		{
			ray_direction[0] = d_xray[0];
			ray_direction[1] = d_xray[2];
			ray_direction[2] = d_xray[1];
		}
		else
		{
			ray_direction[0] = d_rotated_s1[0];
			ray_direction[1] = d_rotated_s1[2];
			ray_direction[2] = d_rotated_s1[1];
		}

		int face = cube_face(coord, ray_direction, x_max, y_max, z_max, is_ray_incomming);
		d_face[id] = face;
	}
}

__inline__ __device__ void get_theta_phi(double *theta, double *phi, double *ray_direction, int L1)
{
	if (L1 == 1)
	{
		ray_direction[0] = -ray_direction[0];
		ray_direction[1] = -ray_direction[1];
		ray_direction[2] = -ray_direction[2];
	}

	if (ray_direction[1] == 0)
	{
		(*theta) = atan(-ray_direction[2] / (-sqrt(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]) + 0.001));
		(*phi) = atan(-ray_direction[0] / (ray_direction[1] + 0.001));
	}
	else
	{
		if (ray_direction[1] < 0)
		{
			(*theta) = atan(-ray_direction[2] / sqrt(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			(*phi) = atan(-ray_direction[0] / (ray_direction[1]));
		}
		else
		{
			if (ray_direction[2] < 0)
			{
				(*theta) = M_PI - atan(-ray_direction[2] / sqrt(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			}
			else
			{
				(*theta) = -M_PI - atan(-ray_direction[2] / sqrt(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			}
			(*phi) = -atan(-ray_direction[0] / (-ray_direction[1]));
		}
	}
}

__global__ void rt_gpu_angles(double *d_angles, double *d_rotated_s1, double *d_xray, int nBatches)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t batch = (id >> 1);
	int is_ray_incomming = id & 1;

	double theta = 0, phi = 0;
	double ray_direction[3];

	if (batch < nBatches)
	{
		if (is_ray_incomming == 1)
		{
			ray_direction[0] = d_xray[0];
			ray_direction[1] = d_xray[1];
			ray_direction[2] = d_xray[2];
		}
		else
		{
			ray_direction[0] = d_rotated_s1[0];
			ray_direction[1] = d_rotated_s1[1];
			ray_direction[2] = d_rotated_s1[2];
		}

		get_theta_phi(&theta, &phi, ray_direction, is_ray_incomming);

		// printf("pos=[%d; %d] theta=%f; phi=%f;\n", (int) (2*id + 0), (int) (2*id + 1), theta, phi);

		d_angles[2 * id + 0] = theta;
		d_angles[2 * id + 1] = phi;
	}
}

__inline__ __device__ void get_increment_ratio(
	double *increment_ratio_x,
	double *increment_ratio_y,
	double *increment_ratio_z,
	double theta,
	double phi,
	int face)
{
	if (face == 1)
	{
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
		*increment_ratio_z = tan(phi);
	}
	else if (face == 2)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
		else
		{
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
	}
	else if (face == 3)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
		else
		{
			*increment_ratio_x = 1 / (tan(fabs(phi)));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
	}
	else if (face == 4)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(phi) / tan(fabs(theta));
		}
		else
		{
			*increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
		}
	}
	else if (face == 5)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(fabs(theta)));
		}
		else
		{
			*increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(M_PI - fabs(theta)));
		}
	}
	else if (face == 6)
	{
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(theta) / cos(phi);
		*increment_ratio_z = tan(phi);
	}
}

__global__ void rt_gpu_increments(double *d_increments, double *d_angles)
{
	// store increments according to different faces and different thetas
	// and for one single reflection, the increments are the same
	// so we only need to store the increments for one single reflection with
	// different crystal voxel positions
	size_t id = threadIdx.x;
	size_t batch = blockIdx.x;
	int face = id % 6;
	int is_ray_incomming = id / 6;

	double theta = 0, phi = 0;
	if (is_ray_incomming == 1)
	{
		theta = d_angles[4 * batch + 2 + 0];
		phi = d_angles[4 * batch + 2 + 1];
	}
	else
	{
		theta = d_angles[4 * batch + 0];
		phi = d_angles[4 * batch + 1];
	}

	double ix = 0, iy = 0, iz = 0;
	get_increment_ratio(&ix, &iy, &iz, theta, phi, face + 1);

	d_increments[36 * batch + 3 * threadIdx.x + 0] = ix;
	d_increments[36 * batch + 3 * threadIdx.x + 1] = iy;
	d_increments[36 * batch + 3 * threadIdx.x + 2] = iz;
}

__inline__ __device__ void get_new_coordinates(
	int *new_x, int *new_y, int *new_z,
	int64_t x, int64_t y, int64_t z,
	double increment_ratio_x, double increment_ratio_y, double increment_ratio_z,
	int increment, double theta, int face)
{
	if (face == 1)
	{
		if (theta > 0)
		{
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = (int)(x - increment * increment_ratio_x);
			*new_y = (int)(y - increment * increment_ratio_y);
			*new_z = (int)(z - increment * increment_ratio_z);
		}
		else
		{
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = (int)(x - increment * increment_ratio_x + 0.5);
			*new_y = (int)(y - increment * increment_ratio_y + 0.5);
			*new_z = (int)(z - increment * increment_ratio_z + 0.5);
		}
	}
	else if (face == 2)
	{
		if (fabs(theta) < M_PI / 2)
		{
			if (theta > 0)
			{
				*new_x = (int)(x + -1 * increment * increment_ratio_x);
				*new_y = (int)(y - increment * increment_ratio_y);
				*new_z = (int)(z + increment * increment_ratio_z);
			}
			else
			{
				*new_x = (int)(x + -1 * increment * increment_ratio_x + 0.5);
				*new_y = (int)(y - increment * increment_ratio_y + 0.5);
				*new_z = (int)(z + increment * increment_ratio_z + 0.5);
			}
		}
		else
		{
			if (theta > 0)
			{
				*new_x = (int)(x + 1 * increment * increment_ratio_x);
				*new_y = (int)(y - increment * increment_ratio_y);
				*new_z = (int)(z + increment * increment_ratio_z);
			}
			else
			{
				*new_x = (int)(x + 1 * increment * increment_ratio_x + 0.5);
				*new_y = (int)(y - increment * increment_ratio_y + 0.5);
				*new_z = (int)(z + increment * increment_ratio_z + 0.5);
			}
		}
	}
	else if (face == 3)
	{
		if (fabs(theta) < M_PI / 2)
		{
			if (theta > 0)
			{
				*new_x = (int)(x + -1 * increment * increment_ratio_x);
				*new_y = (int)(y - increment * increment_ratio_y);
				*new_z = (int)(z + increment * increment_ratio_z);
			}
			else
			{
				*new_x = (int)(x + -1 * increment * increment_ratio_x + 0.5);
				*new_y = (int)(y - increment * increment_ratio_y + 0.5);
				*new_z = (int)(z + increment * increment_ratio_z + 0.5);
			}
		}
		else
		{
			if (theta > 0)
			{
				*new_x = (int)(x + 1 * increment * increment_ratio_x);
				*new_y = (int)(y - increment * increment_ratio_y);
				*new_z = (int)(z + increment * 1);
			}
			else
			{
				*new_x = (int)(x + 1 * increment * increment_ratio_x + 0.5);
				*new_y = (int)(y - increment * increment_ratio_y + 0.5);
				*new_z = (int)(z + increment * 1 + 0.5);
			}
		}
	}
	else if (face == 4)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*new_x = (int)(x + -1 * increment * increment_ratio_x);
			*new_y = (int)(y - increment * increment_ratio_y);
			*new_z = (int)(z + increment * increment_ratio_z);
		}
		else
		{
			*new_x = (int)(x + 1 * increment * increment_ratio_x);
			*new_y = (int)(y - increment * increment_ratio_y);
			*new_z = (int)(z + increment * increment_ratio_z);
		}
	}
	else if (face == 5)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*new_x = (int)(x + -1 * increment * increment_ratio_x + 0.5);
			*new_y = (int)(y - increment * increment_ratio_y + 0.5);
			*new_z = (int)(z + increment * increment_ratio_z + 0.5);
		}
		else
		{
			*new_x = (int)(x + 1 * increment * increment_ratio_x + 0.5);
			*new_y = (int)(y - increment * increment_ratio_y + 0.5);
			*new_z = (int)(z - increment * increment_ratio_z + 0.5);
		}
	}
	else if (face == 6)
	{
		if (theta > 0)
		{
			*new_x = (int)(x + increment * increment_ratio_x);
			*new_y = (int)(y - increment * increment_ratio_y);
			*new_z = (int)(z + increment * increment_ratio_z);
		}
		else
		{
			*new_x = (int)(x + increment * increment_ratio_x + 0.5);
			*new_y = (int)(y - increment * increment_ratio_y + 0.5);
			*new_z = (int)(z + increment * increment_ratio_z + 0.5);
		}
	}
}

__inline__ __device__ void get_distance(double *total_length, double *nElements, double voxel_length_x, double voxel_length_y, double voxel_length_z, double increment_ratio_x, double increment_ratio_y, double increment_ratio_z, int x_max, int y_max, int z_max, int x, int y, int z, int face)
{
	double dist_x, dist_y, dist_z;
	if (blockIdx.x < 20 && threadIdx.x == 0)
	{
		printf("bl:%d; coord=[%d; %d; %d] i=[%f; %f; %f]\n", blockIdx.x, x, y, z, increment_ratio_x, increment_ratio_y, increment_ratio_z);
	}
	if (face == 1)
	{
		dist_x = ((double)(x_max - x));
		dist_y = ((double)(x_max - x)) * (increment_ratio_y);
		dist_z = ((double)(x_max - x)) * (increment_ratio_z);
		*nElements = dist_x;
	}
	else if (face == 2)
	{
		dist_x = ((double)z) * (increment_ratio_x);
		dist_y = ((double)z) * (increment_ratio_y);
		dist_z = ((double)z);
		*nElements = dist_z;
	}
	else if (face == 3)
	{
		dist_x = ((double)(z_max - z)) * (increment_ratio_x);
		dist_y = ((double)(z_max - z)) * (increment_ratio_y);
		dist_z = ((double)(z_max - z));
		*nElements = dist_z;
	}
	else if (face == 4)
	{
		dist_x = ((double)y) * (increment_ratio_x);
		dist_y = ((double)y);
		dist_z = ((double)y) * (increment_ratio_z);
		*nElements = dist_y;
	}
	else if (face == 5)
	{
		dist_x = ((double)(y_max - y)) * (increment_ratio_x);
		dist_y = ((double)(y_max - y));
		dist_z = ((double)(y_max - y)) * (increment_ratio_z);
		*nElements = dist_y;
	}
	else if (face == 6)
	{
		dist_x = ((double)x);
		dist_y = ((double)x) * (increment_ratio_y);
		dist_z = ((double)x) * (increment_ratio_z);
		*nElements = dist_x;
	}
	else
	{
		dist_x = 0;
		dist_y = 0;
		dist_z = 0;
	}

	*total_length = sqrt(
		(dist_x * voxel_length_x) * (dist_x * voxel_length_x) +
		(dist_y * voxel_length_y) * (dist_y * voxel_length_y) +
		(dist_z * voxel_length_z) * (dist_z * voxel_length_z));
}

__inline__ __device__ void get_distance_2(double *total_length, double s_sum, double voxel_length_x, double voxel_length_y, double voxel_length_z, double increment_ratio_x, double increment_ratio_y, double increment_ratio_z, int face, int id)
{
	double dist_x, dist_y, dist_z;
	if (face == 1)
	{
		dist_x = (s_sum - 1.0f);
		dist_y = (s_sum - 1.0f) * increment_ratio_y;
		dist_z = (s_sum - 1.0f) * increment_ratio_z;
	}
	else if (face == 2)
	{
		dist_x = (s_sum - 1.0f) * increment_ratio_x;
		dist_y = (s_sum - 1.0f) * increment_ratio_y;
		dist_z = (s_sum - 1.0f);
	}
	else if (face == 3)
	{
		dist_x = (s_sum - 1.0f) * increment_ratio_x;
		dist_y = (s_sum - 1.0f) * increment_ratio_y;
		dist_z = (s_sum - 1.0f);
	}
	else if (face == 4)
	{
		dist_x = (s_sum - 1.0f) * increment_ratio_x;
		dist_y = (s_sum - 1.0f);
		dist_z = (s_sum - 1.0f) * increment_ratio_z;
	}
	else if (face == 5)
	{
		dist_x = (s_sum - 1.0f) * increment_ratio_x;
		dist_y = (s_sum - 1.0f);
		dist_z = (s_sum - 1.0f) * increment_ratio_z;
	}
	else if (face == 6)
	{
		dist_x = (s_sum - 1.0f);
		dist_y = (s_sum - 1.0f) * increment_ratio_y;
		dist_z = (s_sum - 1.0f) * increment_ratio_z;
	}
	// 	if (face == 1)
	// {
	// 	dist_x = (s_sum  );
	// 	dist_y = (s_sum  ) * increment_ratio_y;
	// 	dist_z = (s_sum  ) * increment_ratio_z;
	// }
	// else if (face == 2)
	// {
	// 	dist_x = (s_sum  ) * increment_ratio_x;
	// 	dist_y = (s_sum  ) * increment_ratio_y;
	// 	dist_z = (s_sum  );
	// }
	// else if (face == 3)
	// {
	// 	dist_x = (s_sum  ) * increment_ratio_x;
	// 	dist_y = (s_sum  ) * increment_ratio_y;
	// 	dist_z = (s_sum  );
	// }
	// else if (face == 4)
	// {
	// 	dist_x = (s_sum  ) * increment_ratio_x;
	// 	dist_y = (s_sum  );
	// 	dist_z = (s_sum  ) * increment_ratio_z;
	// }
	// else if (face == 5)
	// {
	// 	dist_x = (s_sum  ) * increment_ratio_x;
	// 	dist_y = (s_sum  );
	// 	dist_z = (s_sum  ) * increment_ratio_z;
	// }
	// else if (face == 6)
	// {
	// 	dist_x = (s_sum  );
	// 	dist_y = (s_sum  ) * increment_ratio_y;
	// 	dist_z = (s_sum  ) * increment_ratio_z;
	// }
	else
	{
		dist_x = 0;
		dist_y = 0;
		dist_z = 0;
	}
	// if (id <2){
	// printf("id: %d dist_x: %f, dist_y: %f, dist_z: %f\n",id, dist_x, dist_y, dist_z);
	// }
	*total_length = sqrt(
		(dist_x * voxel_length_x) * (dist_x * voxel_length_x) +
		(dist_y * voxel_length_y) * (dist_y * voxel_length_y) +
		(dist_z * voxel_length_z) * (dist_z * voxel_length_z));
}

__global__ void rt_gpu_ray_classes(int *d_ray_classes, int8_t *d_label_list, int64_t *d_coord_list, int *d_face, double *d_angles, double *d_increments, int x_max, int y_max, int z_max, int diagonal)
{
	size_t id = blockIdx.x;
	int is_ray_incomming = id & 1; /*even number is the outgoing, odd number is incoming */
	size_t pos = (id >> 1);
	double increments[3];
	int face = 0;
	int64_t coord[3];
	double theta, phi;

	// Load coordinates
	coord[0] = d_coord_list[3 * pos + 0]; // z
	coord[1] = d_coord_list[3 * pos + 1]; // y
	coord[2] = d_coord_list[3 * pos + 2]; // x

	// Load face
	face = d_face[id];

	// Load angle
	theta = d_angles[4 * blockIdx.y + 2 * is_ray_incomming];
	// phi = d_angles[4*blockIdx.y + 2*is_ray_incomming + 1];

	// Load Increment
	size_t incr_pos = 36 * blockIdx.y + 18 * is_ray_incomming + 3 * (face - 1);
	// get_increment_ratio(&increments[0], &increments[1], &increments[2], theta, phi, face);
	increments[0] = d_increments[incr_pos + 0];
	increments[1] = d_increments[incr_pos + 1];
	increments[2] = d_increments[incr_pos + 2];

	// Calculate number of iterations
	int nIter = (int)((diagonal + blockDim.x - 1) / blockDim.x);
	for (int f = 0; f < nIter; f++)
	{
		// calculate position based on threads id
		// check if the position is within a cube_face
		// write into ray_direction
		int lpos = (f * blockDim.x + threadIdx.x);
		int x, y, z;
		get_new_coordinates(
			&x, &y, &z,
			coord[2], coord[1], coord[0],
			increments[0], increments[1], increments[2],
			lpos, theta, face);
		int label = 0;
		if (
			x < x_max && y < y_max && z < z_max &&
			x >= 0 && y >= 0 && z >= 0)
		{
			size_t cube_pos = INDEX_3D(
				z_max, y_max, x_max,
				z, y, x);
			label = (int)d_label_list[cube_pos];
		}
		if (lpos < diagonal)
		{
			size_t gpos = blockIdx.x * diagonal + lpos;

			d_ray_classes[gpos] = label;
		}
	}
}

__global__ void rt_gpu_absorption(int *d_ray_classes, double *d_absorption, int8_t *d_label_list, int64_t *d_coord_list, int *d_face, double *d_angles, double *d_increments, int x_max, int y_max, int z_max, double voxel_length_x, double voxel_length_y, double voxel_length_z, double coeff_li, double coeff_lo, double coeff_cr, double coeff_bu, int diagonal)
{
	size_t id = blockIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1); /* the right shift operation effectively divided the value of id by 2 (since shifting the bits to the right by 1 is equivalent to integer division by 2).*/
	double increments[3];
	int face = 0;
	int64_t coord[3];
	double theta, phi;
	__shared__ double s_absorption[256];
	__shared__ double s_li_absorption[256];
	__shared__ double s_lo_absorption[256];
	__shared__ double s_cr_absorption[256];
	__shared__ double s_bu_absorption[256];
	__shared__ double s_sum;

	int cr_l_2_int = 0;
	int li_l_2_int = 0;
	int bu_l_2_int = 0;
	int lo_l_2_int = 0;

	// Load coordinates
	coord[0] = d_coord_list[3 * pos + 0]; // z
	coord[1] = d_coord_list[3 * pos + 1]; // y
	coord[2] = d_coord_list[3 * pos + 2]; // x

	// Load face
	face = d_face[id];

	// Load angle
	theta = d_angles[4 * blockIdx.y + 2 * is_ray_incomming];
	// phi = d_angles[4*blockIdx.y + 2*is_ray_incomming + 1];

	// Load Increment
	size_t incr_pos = 36 * blockIdx.y + 18 * is_ray_incomming + 3 * (face - 1);
	// get_increment_ratio(&increments[0], &increments[1], &increments[2], theta, phi, face);
	increments[0] = d_increments[incr_pos + 0];
	increments[1] = d_increments[incr_pos + 1];
	increments[2] = d_increments[incr_pos + 2];

	// Calculate number of iterations
	// trick for ceiling
	int nIter = (int)((diagonal + blockDim.x - 1) / blockDim.x);

	for (int f = 0; f < nIter; f++)
	{
		// calculate position based on threads id
		// check if the position is within a cube_face
		// write into ray_direction
		int lpos = (f * blockDim.x + threadIdx.x);
		int x, y, z;
		get_new_coordinates(
			&x, &y, &z,
			coord[2], coord[1], coord[0],
			increments[0], increments[1], increments[2],
			lpos, theta, face);
		int label = 0;
		// if (threadIdx.x ==0 && blockIdx.x == 0 && lpos == 0) {
		/* there are 256 threads per block and nIter > 256, so the first thread of the first block will be repeatedly used to print the coordinates of the ray.
		 */
		if (DEBUG)
		{
			if (threadIdx.x == 0 && lpos == 0)
			{
				printf("coord is %d %d %d\n x: %d, y: %d, z: %d; increment_z: %f, increment_y: %f , increment_x: %f , \n lpos %d \n", (int)coord[2], (int)coord[1], (int)coord[0], (int)x, (int)y, (int)z, increments[2], increments[1], increments[0], lpos);
			}
		}

		if (
			x < x_max && y < y_max && z < z_max &&
			x >= 0 && y >= 0 && z >= 0)
		{
			size_t cube_pos = INDEX_3D(
				z_max, y_max, x_max,
				z, y, x);
			label = (int)d_label_list[cube_pos];

			if (label == 3)
				cr_l_2_int++;
			else if (label == 1)
				li_l_2_int++;
			else if (label == 2)
				lo_l_2_int++;
			else if (label == 4)
				bu_l_2_int++;
			else
			{
			}
		}
		if (lpos < diagonal)
		{
			size_t gpos = blockIdx.x * diagonal + lpos;
			d_ray_classes[gpos] = label;
		}
	}

	// Calculate number of valid elements
	double sum = cr_l_2_int + li_l_2_int + lo_l_2_int + bu_l_2_int;
	s_absorption[threadIdx.x] = sum;
	__syncthreads();
	sum = Reduce_SM(s_absorption);
	Reduce_WARP(&sum);
	__syncthreads();
	if (threadIdx.x == 0)
	{
		s_sum = sum;
	}
	// if(blockIdx.x < 6 && threadIdx.x == 0) {
	// 	printf("bl:%d; coord is [%d %d %d] cr_l_2_int: %d, li_l_2_int: %d, lo_l_2_int: %d, bu_l_2_int: %d,;\n",blockIdx.x, (int)coord[0],(int)coord[1],(int)coord[2],cr_l_2_int,li_l_2_int,lo_l_2_int,bu_l_2_int);
	// }
	__syncthreads();
	double total_length;
	// get_distance_2(&total_length, s_sum, voxel_length_x, voxel_length_y, voxel_length_z, increments[0], increments[1], increments[2], face,id);
	get_distance_2(&total_length, diagonal, voxel_length_x, voxel_length_y, voxel_length_z, increments[0], increments[1], increments[2], face, id);

	// double cr_l = (total_length * cr_l_2_int) / ((double)s_sum);
	// double li_l = (total_length * li_l_2_int) / ((double)s_sum);
	// double lo_l = (total_length * lo_l_2_int) / ((double)s_sum);
	// double bu_l = (total_length * bu_l_2_int) / ((double)s_sum);
	double cr_l = (total_length * cr_l_2_int) / ((double)diagonal);
	double li_l = (total_length * li_l_2_int) / ((double)diagonal);
	double lo_l = (total_length * lo_l_2_int) / ((double)diagonal);
	double bu_l = (total_length * bu_l_2_int) / ((double)diagonal);

	double absorption = 0;
	double li_absorption = 0;
	double lo_absorption = 0;
	double cr_absorption = 0;
	double bu_absorption = 0;
	s_absorption[threadIdx.x] = coeff_li * li_l + coeff_lo * lo_l + coeff_cr * cr_l + coeff_bu * bu_l;
	s_li_absorption[threadIdx.x] = coeff_li * li_l;
	s_lo_absorption[threadIdx.x] = coeff_lo * lo_l;
	s_cr_absorption[threadIdx.x] = coeff_cr * cr_l;
	s_bu_absorption[threadIdx.x] = coeff_bu * bu_l;
	__syncthreads();
	absorption = Reduce_SM(s_absorption);
	li_absorption = Reduce_SM(s_li_absorption);
	lo_absorption = Reduce_SM(s_lo_absorption);
	cr_absorption = Reduce_SM(s_cr_absorption);
	bu_absorption = Reduce_SM(s_bu_absorption);
	Reduce_WARP(&absorption);
	Reduce_WARP(&li_absorption);
	Reduce_WARP(&lo_absorption);
	Reduce_WARP(&cr_absorption);
	Reduce_WARP(&bu_absorption);
	__syncthreads();

	// calculation of the absorption for given ray
	if (threadIdx.x == 0)
	{
		d_absorption[id] = absorption;
	}
}

// ******************************************************************
// ******************************************************************
// ******************************************************************

void thetaphi_2_numpy_vector(double *vector, double theta, double phi)
{
	vector[0] = cos(theta) * sin(phi);
	vector[1] = sin(theta);
	vector[2] = cos(theta) * cos(phi);
}

int ray_tracing_path(int *h_face, double *h_angles, int *h_ray_classes, double *h_absorption, int64_t *h_coord_list, int64_t len_coord_list, double *h_rotated_s1, double *h_x_ray, double *voxel_size, double *coefficients, int8_t *h_label_list_1d, int64_t *shape)
{
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if (cudaError != cudaSuccess || devCount == 0)
	{
		printf("ERROR: CUDA capable device not found!\n");
		return (1);
	}

	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];

	//---------> Checking memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG)
		printf("--> DEBUG: Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
	int64_t diagonal = x_max * sqrt(3);
	int64_t cube_size = x_max * y_max * z_max * sizeof(int8_t);
	int64_t face_size = len_coord_list * 2 * sizeof(int);
	int64_t absorption_size = len_coord_list * 2 * sizeof(double);
	int64_t angle_size = 4 * sizeof(double);
	int64_t increments_size = 36 * sizeof(double);
	int64_t ray_classes_size = diagonal * len_coord_list * 2 * sizeof(int);
	int64_t coord_list_size = len_coord_list * 3 * sizeof(int64_t);
	int64_t ray_directions_size = 3 * sizeof(double);
	size_t total_memory_required_bytes = face_size + angle_size + increments_size + absorption_size + cube_size + ray_classes_size + coord_list_size + ray_directions_size;
	if (DEBUG)
		printf("--> DEBUG: Total memory required %0.3f MB.\n", (float)total_memory_required_bytes / (1024.0 * 1024.0));
	if (total_memory_required_bytes > free_mem)
	{
		printf("ERROR: Not enough memory! Input data is too big for the device.\n");
		return (1);
	}

	int nCUDAErrors = 0;
	//----------> Memory allocation
	int *d_face;
	double *d_angles;
	double *d_increments;
	int *d_ray_classes;
	double *d_absorption;
	int64_t *d_coord_list;
	double *d_rotated_s1;
	double *d_xray;
	int8_t *d_label_list;
	double *h_increments = (double *)malloc(36 * sizeof(double));
	cudaError = cudaMalloc((void **)&d_face, face_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_face\n");
		d_face = NULL;
	}
	cudaError = cudaMalloc((void **)&d_angles, angle_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_angles\n");
		d_angles = NULL;
	}
	cudaError = cudaMalloc((void **)&d_increments, increments_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_increments\n");
		d_increments = NULL;
	}
	cudaError = cudaMalloc((void **)&d_ray_classes, ray_classes_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_ray_classes\n");
		d_ray_classes = NULL;
	}
	cudaError = cudaMalloc((void **)&d_absorption, absorption_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_absorption\n");
		d_absorption = NULL;
	}
	cudaError = cudaMalloc((void **)&d_coord_list, coord_list_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_coord_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_rotated_s1, ray_directions_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_rotated_s1\n");
		d_rotated_s1 = NULL;
	}
	cudaError = cudaMalloc((void **)&d_xray, ray_directions_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_xray\n");
		d_xray = NULL;
	}
	cudaError = cudaMalloc((void **)&d_label_list, cube_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_label_list\n");
		d_label_list = NULL;
	}

	//---------> Memory copy and preparation
	GpuTimer timer;
	double memory_time = 0;
	timer.Start();
	cudaError = cudaMemcpy(d_label_list, h_label_list_1d, cube_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_label_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_coord_list, h_coord_list, coord_list_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_coord_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_rotated_s1, h_rotated_s1, ray_directions_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_rotated.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_xray, h_x_ray, ray_directions_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_xray.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemset(d_ray_classes, 0, ray_classes_size);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memset d_ray_classes.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	timer.Stop();
	memory_time = timer.Elapsed();

	cudaDeviceSynchronize();

	double compute_time = 0;
	if (nCUDAErrors == 0)
	{
		timer.Start();

		//**************** Calculate faces **************
		{
			int nThreads = 128;
			int nBlocks = ((len_coord_list * 2) + nThreads - 1) / nThreads;
			int x_max = shape[2];
			int y_max = shape[1];
			int z_max = shape[0];
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_get_face<<<gridSize_face, blockSize_face>>>(
				d_face,
				d_coord_list,
				d_rotated_s1,
				d_xray,
				x_max, y_max, z_max, (int)len_coord_list);
		}

		{
			int nBatches = 1;
			int nThreads = 128;
			int nBlocks = ((nBatches * 2) + nThreads - 1) / nThreads;
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_angles<<<gridSize_face, blockSize_face>>>(
				d_angles,
				d_rotated_s1,
				d_xray,
				nBatches);
		}

		{
			int nBatches = 1;
			int nThreads = 12;
			dim3 gridSize_face(nBatches, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_increments<<<gridSize_face, blockSize_face>>>(
				d_increments,
				d_angles);
		}
		cudaError = cudaMemcpy(h_increments, d_increments, increments_size, cudaMemcpyDeviceToHost);
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! Copy of d_face has failed.\n");
			nCUDAErrors++;
		}
		// printf("-------> GPU Increment test:\n");
		// double ix, iy, iz;

		// printf("rotated_s1:\n");
		// for(int f=0; f<6; f++){
		// 	ix = h_increments[f*3+0];
		// 	iy = h_increments[f*3+1];
		// 	iz = h_increments[f*3+2];
		// 	printf("==> GPU: face=%d; i=[%f; %f; %f];\n", f+1, ix, iy, iz);
		// }
		// printf("Xray:\n");
		// for(int f=6; f<12; f++){
		// 	ix = h_increments[f*3+0];
		// 	iy = h_increments[f*3+1];
		// 	iz = h_increments[f*3+2];
		// 	printf("==> GPU: face=%d; i=[%f; %f; %f];\n", f+1, ix, iy, iz);
		// }
		// printf("-------------------------------<\n");

		//---------> error check
		cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! GPU Kernel error.\n");
			print_cuda_error(cudaError);
			nCUDAErrors++;
		}
		else
		{
			printf("No CUDA error.\n");
		}

		{
			/*
			int nBlocks = len_coord_list*2;
			int nThreads = 256;
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_ray_classes<<<gridSize_face , blockSize_face>>>(
				d_ray_classes,
				d_label_list,
				d_coord_list,
				d_face,
				d_angles,
				d_increments,
				x_max, y_max, z_max,
				diagonal
			);
			*/
		}

		//---------> error check
		cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! GPU Kernel error.\n");
			print_cuda_error(cudaError);
			nCUDAErrors++;
		}
		else
		{
			printf("No CUDA error.\n");
		}

		{
			double voxel_length_z = voxel_size[0];
			double voxel_length_y = voxel_size[1];
			double voxel_length_x = voxel_size[2];
			double coeff_li = coefficients[0];
			double coeff_lo = coefficients[1];
			double coeff_cr = coefficients[2];
			double coeff_bu = coefficients[3];

			int nBlocks = len_coord_list * 2;
			int nThreads = 256;
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_absorption<<<gridSize_face, blockSize_face>>>(
				d_ray_classes,
				d_absorption,
				d_label_list,
				d_coord_list,
				d_face,
				d_angles,
				d_increments,
				x_max, y_max, z_max,
				voxel_length_x, voxel_length_y, voxel_length_z,
				coeff_li, coeff_lo, coeff_cr, coeff_bu,
				diagonal);
		}

		timer.Stop();
		compute_time = timer.Elapsed();
	}

	//---------> error check
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! GPU Kernel error.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	else
	{
		printf("No CUDA error.\n");
	}

	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_face, d_face, face_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_face has failed.\n");
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(h_angles, d_angles, angle_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_angles has failed.\n");
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(h_ray_classes, d_ray_classes, ray_classes_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_ray_classes has failed.\n");
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(h_absorption, d_absorption, absorption_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_absorption has failed.\n");
		nCUDAErrors++;
	}

	//---------> error check
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
	}

	if (nCUDAErrors != 0)
	{
		printf("Number of CUDA errors = %d;\n", nCUDAErrors);
	}
	if (DEBUG)
		printf("--> DEBUG: Memory preparation time: %fms; Compute time: %fms;\n", memory_time, compute_time);

	//---------> Freeing allocated resources
	if (d_face != NULL)
		cudaFree(d_face);
	if (d_angles != NULL)
		cudaFree(d_angles);
	if (d_increments != NULL)
		cudaFree(d_increments);
	if (d_ray_classes != NULL)
		cudaFree(d_ray_classes);
	if (d_absorption != NULL)
		cudaFree(d_absorption);
	if (d_coord_list != NULL)
		cudaFree(d_coord_list);
	if (d_rotated_s1 != NULL)
		cudaFree(d_rotated_s1);
	if (d_xray != NULL)
		cudaFree(d_xray);
	if (d_label_list != NULL)
		cudaFree(d_label_list);

	cudaDeviceSynchronize();

	return (0);
}

int verificaton_ray_tracing_path(double *h_angle_list, int32_t h_len_result, int64_t *h_coord_list, int32_t len_coord_list, int8_t *h_label_list_1d, double *voxel_size, double *coefficients, int64_t *shape, double *h_result_list)
{
	int64_t absorption_size = len_coord_list * 2 * sizeof(double);
	double *h_absorption = (double *)malloc(absorption_size);
	double phi = 0 / 180 * M_PI;
	double theta_1 = 180 / 180 * M_PI, phi_1 = 0 / 180 * M_PI;
	double theta;
	thetaphi_2_numpy_vector(h_x_ray, theta_1, phi_1);
	printf("h_x_ray: %f %f %f\n", h_x_ray[0], h_x_ray[1], h_x_ray[2]);
	double h_x_ray= (double *)malloc(ray_directions_size);
	double *h_rotated_s1 = (double *)malloc(ray_directions_size);
	theta = h_angle_list[a] / 180 * M_PI;
	thetaphi_2_numpy_vector(h_rotated_s1, theta, phi);
	printf("h_rotated_s1: %f %f %f\n", h_rotated_s1[0], h_rotated_s1[1], h_rotated_s1[2]);
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if (cudaError != cudaSuccess || devCount == 0)
	{
		printf("ERROR: CUDA capable device not found!\n");
		return (1);
	}

	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];

	//---------> Checking memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG)
		printf("--> DEBUG: Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
	int64_t diagonal = x_max * sqrt(3);
	int64_t cube_size = x_max * y_max * z_max * sizeof(int8_t);

	int64_t face_size = len_coord_list * 2 * sizeof(int);
	// int64_t absorption_size = len_coord_list * 2 * sizeof(double);
	int64_t angle_size = 4 * sizeof(double);
	int64_t increments_size = 36 * sizeof(double);
	int64_t ray_classes_size = diagonal * len_coord_list * 2 * sizeof(int);
	int64_t coord_list_size = len_coord_list * 3 * sizeof(int64_t);
	int64_t ray_directions_size = 3 * sizeof(double);
	size_t total_memory_required_bytes = face_size + angle_size + increments_size + absorption_size + cube_size + ray_classes_size + coord_list_size + ray_directions_size;
	if (DEBUG)
		printf("--> DEBUG: Total memory required %0.3f MB.\n", (float)total_memory_required_bytes / (1024.0 * 1024.0));
	if (total_memory_required_bytes > free_mem)
	{
		printf("--> ERROR:  Total memory required %0.3f MB.\n", (float)total_memory_required_bytes / (1024.0 * 1024.0));
		printf("ERROR: Not enough memory! Input data is too big for the device.\n");
		return (1);
	}

	int nCUDAErrors = 0;
	//----------> Memory allocation
	int *d_face;
	double *d_angles;
	double *d_increments;
	int *d_ray_classes;
	double *d_absorption;
	int64_t *d_coord_list;
	double *d_rotated_s1;
	double *d_xray;
	int8_t *d_label_list;
	double *h_increments = (double *)malloc(36 * sizeof(double));
	cudaError = cudaMalloc((void **)&d_face, face_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_face\n");
		d_face = NULL;
	}
	cudaError = cudaMalloc((void **)&d_angles, angle_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_angles\n");
		d_angles = NULL;
	}
	cudaError = cudaMalloc((void **)&d_increments, increments_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_increments\n");
		d_increments = NULL;
	}
	cudaError = cudaMalloc((void **)&d_ray_classes, ray_classes_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_ray_classes\n");
		d_ray_classes = NULL;
	}
	cudaError = cudaMalloc((void **)&d_absorption, absorption_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_absorption\n");
		d_absorption = NULL;
	}
	cudaError = cudaMalloc((void **)&d_coord_list, coord_list_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_coord_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_rotated_s1, ray_directions_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_rotated_s1\n");
		d_rotated_s1 = NULL;
	}
	cudaError = cudaMalloc((void **)&d_xray, ray_directions_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_xray\n");
		d_xray = NULL;
	}
	cudaError = cudaMalloc((void **)&d_label_list, cube_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_label_list\n");
		d_label_list = NULL;
	}




	cudaError = cudaMemcpy(d_label_list, h_label_list_1d, cube_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_label_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_coord_list, h_coord_list, coord_list_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_coord_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}

	cudaError = cudaMemcpy(d_xray, h_x_ray, ray_directions_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_xray.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemset(d_ray_classes, 0, ray_classes_size);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memset d_ray_classes.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}

	for (int a = 0; a < h_len_result; a++)
	{
		double *h_rotated_s1 = (double *)malloc(ray_directions_size);
		theta = h_angle_list[a] / 180 * M_PI;
		thetaphi_2_numpy_vector(h_rotated_s1, theta, phi);
		printf("h_rotated_s1: %f %f %f\n", h_rotated_s1[0], h_rotated_s1[1], h_rotated_s1[2]);
		//---------> Memory copy and preparation
		cudaError = cudaMemcpy(d_rotated_s1, h_rotated_s1, ray_directions_size, cudaMemcpyHostToDevice);


		if (cudaError != cudaSuccess)
		{
			printf("ERROR! Memcopy d_rotated.\n");
			print_cuda_error(cudaError);
			nCUDAErrors++;
		}
		cudaDeviceSynchronize();

		double compute_time = 0;
		if (nCUDAErrors == 0)
		{
			// timer.Start();

			//**************** Calculate faces **************
			{
				int nThreads = 128;
				int nBlocks = ((len_coord_list * 2) + nThreads - 1) / nThreads;
				int x_max = shape[2];
				int y_max = shape[1];
				int z_max = shape[0];
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_get_face<<<gridSize_face, blockSize_face>>>(
					d_face,
					d_coord_list,
					d_rotated_s1,
					d_xray,
					x_max, y_max, z_max, (int)len_coord_list);
			}

			{
				int nBatches = 1;
				int nThreads = 128;
				int nBlocks = ((nBatches * 2) + nThreads - 1) / nThreads;
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_angles<<<gridSize_face, blockSize_face>>>(
					d_angles,
					d_rotated_s1,
					d_xray,
					nBatches);
			}

			{
				int nBatches = 1;
				int nThreads = 12;
				dim3 gridSize_face(nBatches, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_increments<<<gridSize_face, blockSize_face>>>(
					d_increments,
					d_angles);
			}
			cudaError = cudaMemcpy(h_increments, d_increments, increments_size, cudaMemcpyDeviceToHost);
			if (cudaError != cudaSuccess)
			{
				printf("ERROR! Copy of d_face has failed.\n");
				nCUDAErrors++;
			}
			// printf("-------> GPU Increment test:\n");
			// double ix, iy, iz;

			// printf("rotated_s1:\n");
			// for(int f=0; f<6; f++){
			// 	ix = h_increments[f*3+0];
			// 	iy = h_increments[f*3+1];
			// 	iz = h_increments[f*3+2];
			// 	printf("==> GPU: face=%d; i=[%f; %f; %f];\n", f+1, ix, iy, iz);
			// }
			// printf("Xray:\n");
			// for(int f=6; f<12; f++){
			// 	ix = h_increments[f*3+0];
			// 	iy = h_increments[f*3+1];
			// 	iz = h_increments[f*3+2];
			// 	printf("==> GPU: face=%d; i=[%f; %f; %f];\n", f+1, ix, iy, iz);
			// }
			// printf("-------------------------------<\n");

			//---------> error check
			cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess)
			{
				printf("ERROR! GPU Kernel error.\n");
				print_cuda_error(cudaError);
				nCUDAErrors++;
			}
			else
			{
				printf("No CUDA error.\n");
			}

			{
				/*
				int nBlocks = len_coord_list*2;
				int nThreads = 256;
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_ray_classes<<<gridSize_face , blockSize_face>>>(
					d_ray_classes,
					d_label_list,
					d_coord_list,
					d_face,
					d_angles,
					d_increments,
					x_max, y_max, z_max,
					diagonal
				);
				*/
			}

			//---------> error check
			cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess)
			{
				printf("ERROR! GPU Kernel error.\n");
				print_cuda_error(cudaError);
				nCUDAErrors++;
			}
			else
			{
				printf("No CUDA error.\n");
			}

			{
				double voxel_length_z = voxel_size[0];
				double voxel_length_y = voxel_size[1];
				double voxel_length_x = voxel_size[2];
				double coeff_li = coefficients[0];
				double coeff_lo = coefficients[1];
				double coeff_cr = coefficients[2];
				double coeff_bu = coefficients[3];

				int nBlocks = len_coord_list * 2;
				int nThreads = 256;
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_absorption<<<gridSize_face, blockSize_face>>>(
					d_ray_classes,
					d_absorption,
					d_label_list,
					d_coord_list,
					d_face,
					d_angles,
					d_increments,
					x_max, y_max, z_max,
					voxel_length_x, voxel_length_y, voxel_length_z,
					coeff_li, coeff_lo, coeff_cr, coeff_bu,
					diagonal);
			}

			// timer.Stop();
			// compute_time = timer.Elapsed();
		}
		//---------> error check
		cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! GPU Kernel error.\n");
			print_cuda_error(cudaError);
			nCUDAErrors++;
		}
		else
		{
			printf("No CUDA error.\n");
		}

		//-----> Copy chunk of output data to host
		// cudaError = cudaMemcpy(h_face, d_face, face_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_face has failed.\n");
		// 	nCUDAErrors++;
		// }
		// cudaError = cudaMemcpy(h_angles, d_angles, angle_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_angles has failed.\n");
		// 	nCUDAErrors++;
		// }
		// cudaError = cudaMemcpy(h_ray_classes, d_ray_classes, ray_classes_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_ray_classes has failed.\n");
		// 	nCUDAErrors++;
		// }
		cudaError = cudaMemcpy(h_absorption, d_absorption, absorption_size, cudaMemcpyDeviceToHost);
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! Copy of d_absorption has failed.\n");
			nCUDAErrors++;
		}

		//---------> error check
		cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess)
		{
			nCUDAErrors++;
		}

		if (nCUDAErrors != 0)
		{
			printf("Number of CUDA errors = %d;\n", nCUDAErrors);
		}
		// if (DEBUG)
		// printf("--> DEBUG: Memory preparation time: %fms; Compute time: %fms;\n", memory_time, compute_time);

		//---------> Freeing allocated resources
		if (d_face != NULL)
			cudaFree(d_face);
		if (d_angles != NULL)
			cudaFree(d_angles);
		if (d_increments != NULL)
			cudaFree(d_increments);
		if (d_ray_classes != NULL)
			cudaFree(d_ray_classes);
		if (d_absorption != NULL)
			cudaFree(d_absorption);
		if (d_coord_list != NULL)
			cudaFree(d_coord_list);
		if (d_rotated_s1 != NULL)
			cudaFree(d_rotated_s1);
		if (d_xray != NULL)
			cudaFree(d_xray);
		if (d_label_list != NULL)
			cudaFree(d_label_list);

		cudaDeviceSynchronize();
		double result = 0;
		for (int k = 0; k < h_len_result; k++)
		{
			result += h_absorption[k];
		}
		result = result / h_len_result;
		h_result_list[a] = result;
		printf("Result: %f\n", result);
		
	}
	return (0);
}
