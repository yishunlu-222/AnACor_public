#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <cublas_v2.h>
#include "GPU_reduction.cuh"

#define DEBUG 0

#include "timer.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)



void print_cuda_error(cudaError_t code)
{
	printf("CUDA error code: %d; string: %s;\n", (int)code, cudaGetErrorString(code));
}


__global__ void rt_gpu_python_results(int64_t len_coord_list, double * d_result_list, double * d_python_result_list, int len_result){
	size_t id =blockDim.x * blockIdx.x + threadIdx.x;
	if (id < len_result){
	
	double gpu_absorption = 0;
	for (int64_t j = 0; j < len_coord_list; j++)
	{
		gpu_absorption += exp(-(d_result_list[id*len_coord_list*2+2 * j + 0] + d_result_list[id*len_coord_list*2+2 * j + 1]));

	}
	double gpu_absorption_mean = gpu_absorption / ((double)len_coord_list);
	d_python_result_list[id] = gpu_absorption_mean;
	}
}

__inline__ __device__ void transpose_device(double *input, int rows, int cols, double *output)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output[j * rows + i] = input[i * cols + j];
		}
	}
}

__inline__ __device__ void dot_product_device(const double *A, const double *B, double *C, int m, int n, int p)
{
	//     In the provided example, the dimensions m, n, and p of the matrices are as follows:

	// Matrix A: m x n = 2 x 3 (2 rows, 3 columns)
	// Matrix B: n x p = 3 x 2 (3 rows, 2 columns)
	// Matrix C: m x p = 2 x 2 (2 rows, 2 columns)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < p; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < n; k++)
			{
				sum += A[i * n + k] * B[k * p + j];
			}
			C[i * p + j] = sum;
		}
	}
}

__inline__ __device__ void kp_rotation_device(const double *axis, double theta, double *result)
{
	double x = axis[0];
	double y = axis[1];
	double z = axis[2];
	double c = cos(theta);
	double s = sin(theta);

	result[0] = c + (x * x) * (1 - c);
	result[1] = x * y * (1 - c) - z * s;
	result[2] = y * s + x * z * (1 - c);

	result[3] = z * s + x * y * (1 - c);
	result[4] = c + (y * y) * (1 - c);
	result[5] = -x * s + y * z * (1 - c);

	result[6] = -y * s + x * z * (1 - c);
	result[7] = x * s + y * z * (1 - c);
	result[8] = c + (z * z) * (1 - c);
}

__global__ void ray_tracing_rotation(const double *d_omega_axis, double *d_omega_list, double *d_kp_rotation_matrix, double *d_raw_xray, double *d_scattering_vector_list, double *d_rotated_xray_list, double *d_rotated_s1_list)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	double rotation_matrix_frame_omega[9];
	double rotation_matrix_overall[9];
	double total_rotation_matrix[9];
	double rotated_xray[3];
	double rotated_s1[3];

	kp_rotation_device(d_omega_axis, d_omega_list[id], rotation_matrix_frame_omega);
	dot_product_device((double *)rotation_matrix_frame_omega, d_kp_rotation_matrix, (double *)rotation_matrix_overall, 3, 3, 3);
	transpose_device((double *)rotation_matrix_overall, 3, 3, (double *)total_rotation_matrix);

	dot_product_device((double *)total_rotation_matrix, d_raw_xray, (double *)rotated_xray, 3, 3, 1);
	d_rotated_xray_list[3 * id] = rotated_xray[0];
	d_rotated_xray_list[3 * id + 1] = rotated_xray[1];
	d_rotated_xray_list[3 * id + 2] = rotated_xray[2];

	double scattering_vector[3] = {d_scattering_vector_list[id * 3],
								   d_scattering_vector_list[id * 3 + 1],
								   d_scattering_vector_list[id * 3 + 2]};
	dot_product_device((double *)total_rotation_matrix, (double *)scattering_vector, (double *)rotated_s1, 3, 3, 1);
	d_rotated_s1_list[3 * id] = rotated_s1[0];
	d_rotated_s1_list[3 * id + 1] = rotated_s1[1];
	d_rotated_s1_list[3 * id + 2] = rotated_s1[2];
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

// __global__ void rt_gpu_get_face(int *d_face, int64_t *d_coord_list, double *d_rotated_s1, double *d_xray, int x_max, int y_max, int z_max, int len_coord_list, int)
__global__ void rt_gpu_get_face(int *d_face, int64_t *d_coord_list, double *d_rotated_s1_list, double *d_rotated_xray_list, int x_max, int y_max, int z_max, int len_coord_list, int batch_number)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1);

	int64_t coord[3];
	double ray_direction[3];

	if (pos < len_coord_list)
	{
		coord[0] = d_coord_list[3 * pos + 0];
		coord[1] = d_coord_list[3 * pos + 1];
		coord[2] = d_coord_list[3 * pos + 2];

		if (is_ray_incomming == 1)
		{
			ray_direction[0] = d_rotated_xray_list[batch_number * 3 + 0];
			ray_direction[1] = d_rotated_xray_list[batch_number * 3 + 2];
			ray_direction[2] = d_rotated_xray_list[batch_number * 3 + 1];
		}
		else
		{
			ray_direction[0] = d_rotated_s1_list[batch_number * 3 + 0];
			ray_direction[1] = d_rotated_s1_list[batch_number * 3 + 2];
			ray_direction[2] = d_rotated_s1_list[batch_number * 3 + 1];
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

__global__ void rt_gpu_angles(double *d_angles, double *d_rotated_s1_list, double *d_rotated_xray_list, int nBatches, int batch_number)
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
			ray_direction[0] = d_rotated_xray_list[batch_number * 3 + 0];
			ray_direction[1] = d_rotated_xray_list[batch_number * 3 + 1];
			ray_direction[2] = d_rotated_xray_list[batch_number * 3 + 2];
		}
		else
		{
			ray_direction[0] = d_rotated_s1_list[batch_number * 3 + 0];
			ray_direction[1] = d_rotated_s1_list[batch_number * 3 + 1];
			ray_direction[2] = d_rotated_s1_list[batch_number * 3 + 2];
		}

		get_theta_phi(&theta, &phi, ray_direction, is_ray_incomming);

		// printf("pos=[%d; %d] theta=%f; phi=%f;\n", (int) (2*id + 0), (int) (2*id + 1), theta, phi);

		d_angles[2 * id + 0] = theta;
		d_angles[2 * id + 1] = phi;
	}
	// printf("d_angles =[%f,%f,%f,%f]  ", d_angles[0], d_angles[1], d_angles[2], d_angles[3]);
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

__inline__ __device__ void get_distance_2(double *total_length, double s_sum, double voxel_length_x, double voxel_length_y, double voxel_length_z, double increment_ratio_x, double increment_ratio_y, double increment_ratio_z, int face)
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

__global__ void rt_gpu_absorption(int *d_ray_classes, double *d_absorption, int8_t *d_label_list, int64_t *d_coord_list, int *d_face, double *d_angles, double *d_increments, int x_max, int y_max, int z_max, double voxel_length_x, double voxel_length_y, double voxel_length_z, double coeff_li, double coeff_lo, double coeff_cr, double coeff_bu, int diagonal,double * d_result_list, int index,int len_coord_list)
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
		// if (DEBUG)
		// {
		// 	if (threadIdx.x == 0 && lpos == 0)
		// 	{
		// 		printf("coord is %d %d %d\n x: %d, y: %d, z: %d; increment_z: %f, increment_y: %f , increment_x: %f , \n lpos %d \n", (int)coord[2], (int)coord[1], (int)coord[0], (int)x, (int)y, (int)z, increments[2], increments[1], increments[0], lpos);
		// 	}
		// }

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
	// get_distance_2(&total_length, s_sum, voxel_length_x, voxel_length_y, voxel_length_z, increments[0], increments[1], increments[2], face, id);
	get_distance_2(&total_length, diagonal, voxel_length_x, voxel_length_y, voxel_length_z, increments[0], increments[1], increments[2], face);
	// if(blockIdx.x <2 && threadIdx.x == 0) {

	// 	printf("bl:%d; coord is [%d %d %d] total_length=%f; sum = %f;\n",blockIdx.x, (int)coord[0],(int)coord[1],(int)coord[2],total_length, s_sum);
	// }

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
	// s_li_absorption[threadIdx.x] = coeff_li * li_l;
	// s_lo_absorption[threadIdx.x] = coeff_lo * lo_l;
	// s_cr_absorption[threadIdx.x] = coeff_cr * cr_l;
	// s_bu_absorption[threadIdx.x] = coeff_bu * bu_l;
	__syncthreads();
	absorption = Reduce_SM(s_absorption);
	// li_absorption = Reduce_SM(s_li_absorption);
	// lo_absorption = Reduce_SM(s_lo_absorption);
	// cr_absorption = Reduce_SM(s_cr_absorption);
	// bu_absorption = Reduce_SM(s_bu_absorption);
	Reduce_WARP(&absorption);
	// Reduce_WARP(&li_absorption);
	// Reduce_WARP(&lo_absorption);
	// Reduce_WARP(&cr_absorption);
	// Reduce_WARP(&bu_absorption);
	__syncthreads();

	// calculation of the absorption for given ray
	if (threadIdx.x == 0)
	{
		d_absorption[id] = absorption;
		d_result_list[index * len_coord_list*2+id] = absorption;
		// if (id<2){
		// printf("id: %d, absorption: %f, li_absorption: %f, lo_absorption: %f, cr_absorption: %f, bu_absorption: %f\n",id, absorption, li_absorption, lo_absorption, cr_absorption, bu_absorption);
		// printf("id: %d,total_length: %f\n",id, total_length);
		// }
	}
}



// ******************************************************************
// ******************************************************************
// ******************************************************************



int ray_tracing_path(int *h_face, double *h_angles, int *h_ray_classes, double *h_absorption, int64_t *h_coord_list, int64_t len_coord_list, double *h_rotated_s1, double *h_xray, double *voxel_size, double *coefficients, int8_t *h_label_list_1d, int64_t *shape)
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
	cudaError = cudaMemcpy(d_xray, h_xray, ray_directions_size, cudaMemcpyHostToDevice);
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
	int i = 0;
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
				x_max, y_max, z_max, (int)len_coord_list, i);
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
				nBatches, i);
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
				diagonal,d_absorption,0,len_coord_list);
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
	free(h_increments);
	cudaDeviceSynchronize();

	return (0);
}

void transpose(double *input, int rows, int cols, double *output);
void dot_product(const double *A, const double *B, double *C, int m, int n, int p);
void kp_rotation(const double *axis, double theta, double *result);

int ray_tracing_gpu_overall_kernel(int32_t low, int32_t up,
								   int64_t *coord_list,
								   int64_t len_coord_list,
								   const double *scattering_vector_list, const double *omega_list,
								   const double *raw_xray,
								   const double *omega_axis, const double *kp_rotation_matrix,
								   int64_t len_result,
								   double *voxel_size, double *coefficients,
								   int8_t *label_list_1d, int64_t *shape, int32_t full_iteration,
								   int32_t store_paths, double *h_result_list,int *h_face,double *h_angles,double *h_python_result_list)
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
	printf("--> GPU info:" );
	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];

	//---------> Checking memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);

	printf("--> GPU info: Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));

	int64_t diagonal = x_max * sqrt(3);
	int64_t cube_size = x_max * y_max * z_max * sizeof(int8_t);
	int64_t face_size = len_coord_list * 2 * sizeof(int);
	int64_t absorption_size = len_coord_list * 2 * sizeof(double);
	int64_t angle_size = 4 * sizeof(double);
	int64_t increments_size = 36 * sizeof(double);
	int64_t ray_classes_size = diagonal * len_coord_list * 2 * sizeof(int);
	int64_t coord_list_size = len_coord_list * 3 * sizeof(int64_t);
	int64_t ray_directions_size = 3 * sizeof(double);


	int64_t h_len_result_coord = (int64_t)len_result * (int64_t)len_coord_list;
	size_t result_size = h_len_result_coord * 2 * sizeof(double);
	int64_t python_result_size = len_result * sizeof(double); // my desktop doesnt have enough memory to store the whole result list, so take a half of it to test
	int64_t scattering_vector_list_size = len_result * sizeof(double) * 3;
	int64_t omega_list_size = len_result * sizeof(double);
	int64_t raw_xray_size = 3 * sizeof(double);
	int64_t omega_axis_size = 3 * sizeof(double);
	int64_t kp_rotation_matrix_size = 9 * sizeof(double);
	int64_t rotated_s1_size = len_result * sizeof(double) * 3;
	int64_t rotated_xray_size = len_result * sizeof(double) * 3;

	size_t total_memory_required_bytes = face_size + angle_size + increments_size + absorption_size + cube_size + ray_classes_size + coord_list_size + ray_directions_size+result_size+scattering_vector_list_size+omega_list_size+raw_xray_size+omega_axis_size+kp_rotation_matrix_size+rotated_s1_size+rotated_xray_size;

	printf("--> DEBUG: Total memory required %0.3f MB.\n", (double)total_memory_required_bytes / (1024.0 * 1024.0));

	if (total_memory_required_bytes > free_mem)
	{
		printf("--> DEBUG: Total memory required %0.3f MB.\n", (double)
																	   total_memory_required_bytes /
																   (1024.0 * 1024.0));
		printf("ERROR: Not enough memory! Input data is too big for the device.\n");
		return (1);
	}

	int nCUDAErrors = 0;
	//----------> Memory allocation
	//---------> Allocating memory on the device
	// global memory
	double *d_result_list;  // contains i reflections, each with j rays
	double *d_python_result_list; // contains i reflections
	double *d_scattering_vector_list;
	double *d_omega_list;
	double *d_raw_xray;
	double *d_omega_axis;
	double *d_kp_rotation_matrix;
	double *d_rotated_s1_list;
	double *d_rotated_xray_list;
	int64_t *d_coord_list;
	int8_t *d_label_list;

	// individual memory for each relfection
	int *d_face;
	double *d_angles;
	double *d_increments;
	int *d_ray_classes;
	double *d_absorption_lengths;

	// output memory
	double *h_rotated_s1_list = (double *)malloc(rotated_s1_size);
	double *h_rotated_xray_list = (double *)malloc(rotated_xray_size);
	// double *h_result_list = (double *)malloc(result_size);

	cudaError = cudaMalloc((void **)&d_result_list, result_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_result_list\n");
		d_result_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_python_result_list, python_result_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_python_result_list\n");
		d_result_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_scattering_vector_list, scattering_vector_list_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_scattering_vector_list\n");
		d_scattering_vector_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_omega_list, omega_list_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_omega_list\n");
		d_omega_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_raw_xray, raw_xray_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_raw_xray\n");
		d_raw_xray = NULL;
	}
	cudaError = cudaMalloc((void **)&d_omega_axis, omega_axis_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_omega_axis\n");
		d_omega_axis = NULL;
	}
	cudaError = cudaMalloc((void **)&d_kp_rotation_matrix, kp_rotation_matrix_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_kp_rotation_matrix\n");
		d_kp_rotation_matrix = NULL;
	}

	cudaError = cudaMalloc((void **)&d_rotated_xray_list, rotated_xray_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_rotated_xray_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_rotated_s1_list, rotated_s1_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_rotated_s1_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_coord_list, coord_list_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_coord_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_label_list, cube_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_label_list\n");
		d_label_list = NULL;
	}

	cudaError = cudaMalloc((void **)&d_face, face_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_face\n");
		d_label_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_angles, angle_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_angles\n");
		d_label_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_increments, increments_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_increments\n");
		d_label_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_ray_classes, ray_classes_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_ray_classes\n");
		d_label_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_absorption_lengths, absorption_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_absorption_lengths \n");
		d_label_list = NULL;
	}
	cudaError = cudaMalloc((void **)&d_absorption_lengths, absorption_size);
	if (cudaError != cudaSuccess)
	{
		nCUDAErrors++;
		printf("ERROR: memory allocation d_absorption_lengths \n");
		d_label_list = NULL;
	}
	//---------> Memory copy and preparation
	GpuTimer timer;
	double memory_time = 0;
	timer.Start();
	cudaError = cudaMemcpy(d_result_list, h_result_list, result_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_result_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_scattering_vector_list, scattering_vector_list, scattering_vector_list_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_scattering_vector_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_omega_list, omega_list, omega_list_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_omega_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_raw_xray, raw_xray, raw_xray_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_raw_xray.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_omega_axis, omega_axis, omega_axis_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_omega_axis.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_kp_rotation_matrix, kp_rotation_matrix, kp_rotation_matrix_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_kp_rotation_matrix.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_coord_list, coord_list, coord_list_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_coord_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(d_label_list, label_list_1d, cube_size, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Memcopy d_label_list.\n");
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

	//---------> Kernel execution
	double compute_time = 0;
	printf("len result: %d\n", len_result);
	if (nCUDAErrors == 0)
	{
		timer.Start();
		int nThreads = 256;
		int nBlocks = (len_result + nThreads - 1) / nThreads;
		dim3 gridSize_rotation(nBlocks, 1, 1);
		dim3 blockSize_rotation(nThreads, 1, 1);

		ray_tracing_rotation<<<gridSize_rotation, blockSize_rotation>>>(d_omega_axis, d_omega_list, d_kp_rotation_matrix, d_raw_xray, d_scattering_vector_list, d_rotated_xray_list, d_rotated_s1_list);

		timer.Stop();
		compute_time = timer.Elapsed();
	}
	cudaDeviceSynchronize();
	//---------> error check
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! GPU Kernel error.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++;
	}
	else{
	if (DEBUG)
	{
		printf("No CUDA error.\n");
		printf("Rotation matrices are calculated\n");
	}
	}
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy( h_rotated_xray_list, d_rotated_xray_list, rotated_xray_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_rotated_xray_list has failed.\n");
		nCUDAErrors++;
	}
	cudaError = cudaMemcpy(h_rotated_s1_list,d_rotated_s1_list,  rotated_s1_size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf("ERROR! Copy of d_rotated_s1_list has failed.\n");
		nCUDAErrors++;
	}

	double total_time = 0;
	timer.Start();
	// for (int i = 0; i < 5; i++)
	// {

	// 	printf("[%d]",i);
	// 	printf("h_rotated_xray_list =[%f,%f,%f]  ", h_rotated_xray_list[i*3], h_rotated_xray_list[i*3+1], h_rotated_xray_list[i*3+2]);
	// 	printf("h_rotated_s1_list =[%f,%f,%f] \n",h_rotated_s1_list[i*3], h_rotated_s1_list[i*3+1], h_rotated_s1_list[i*3+2]);
	// }
	for (int i = 0; i < (int)len_result; i++)
	{

		if (nCUDAErrors == 0)

		{


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
					d_rotated_s1_list,
					d_rotated_xray_list,
					x_max, y_max, z_max, (int)len_coord_list, i);
			}

			{
				int nBatches = 1;
				int nThreads = 128;
				int nBlocks = ((nBatches * 2) + nThreads - 1) / nThreads;
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_angles<<<gridSize_face, blockSize_face>>>(
					d_angles,
					d_rotated_s1_list,
					d_rotated_xray_list,
					nBatches, i);
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
		
	
			//---------> error check
			cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess)
			{
				printf("ERROR! GPU Kernel error.\n");
				print_cuda_error(cudaError);
				nCUDAErrors++;
			}
			// else
			// {
			// 	printf("No CUDA error.\n");
			// }

			{
				double voxel_length_z = voxel_size[0];
				double voxel_length_y = voxel_size[1];
				double voxel_length_x = voxel_size[2];
				double coeff_li = coefficients[0];
				double coeff_lo = coefficients[1];
				double coeff_cr = coefficients[2];
				double coeff_bu = coefficients[3];

				int nBlocks = len_coord_list * 2;
				int nThreads = 256; //256:49s ,128:49s 32:52s,512:fail,320:55s
				dim3 gridSize_face(nBlocks, 1, 1);
				dim3 blockSize_face(nThreads, 1, 1);
				rt_gpu_absorption<<<gridSize_face, blockSize_face>>>(
					d_ray_classes,
					d_absorption_lengths,
					d_label_list,
					d_coord_list,
					d_face,
					d_angles,
					d_increments,
					x_max, y_max, z_max,
					voxel_length_x, voxel_length_y, voxel_length_z,
					coeff_li, coeff_lo, coeff_cr, coeff_bu,
					diagonal,d_result_list,i,len_coord_list);
			}


		}
		//---------> error check for the first result-----> passed
		// printf("i is %d\n", i);
		// cudaError = cudaMemcpy(h_result_list,d_result_list,  result_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_rotated_s1_list has failed.\n");
		// 	nCUDAErrors++;
		// }
		// double gpu_absorption = 0;
		// for (int64_t j = 0; j < len_coord_list; j++)
		// {
		// 	gpu_absorption += exp(-(h_result_list[i*len_coord_list*2+2 * j + 0] + h_result_list[i*len_coord_list*2+2 * j + 1]));



		// }

		// double gpu_absorption_mean = gpu_absorption / ((double)len_coord_list);

		// printf("GPU mean absorption in cuda code: %f;\n", gpu_absorption_mean);

		cudaDeviceSynchronize();


		// double result;
		// double rotation_matrix_frame_omega[9];
		// double rotation_matrix[9];
		// double total_rotation_matrix[9];
		// double xray[3];
		// double rotated_s1[3];
		// kp_rotation(omega_axis, omega_list[i], (double *)rotation_matrix_frame_omega);
		// dot_product((double *)rotation_matrix_frame_omega, kp_rotation_matrix, (double *)rotation_matrix, 3, 3, 3);
		// transpose((double *)rotation_matrix, 3, 3, (double *)total_rotation_matrix);
		// dot_product((double *)total_rotation_matrix, raw_xray, (double *)xray, 3, 3, 1);
		// // printf("xray is \n");
		// // print_matrix(xray,1,3);
		// double scattering_vector[3] = {scattering_vector_list[i * 3],
		//                                scattering_vector_list[i * 3 + 1],
		//                                scattering_vector_list[i * 3 + 2]};
		// dot_product((double *)total_rotation_matrix, (double *)scattering_vector, (double *)rotated_s1, 3, 3, 1);
		// if (i < 1){
		// 	printf("rotated xray in host is [%f,%f,%f]\n", xray[0], xray[1], xray[2]);
		// 	printf("rotated xray in device is [%f,%f,%f]\n", h_rotated_xray_list[i*3+0], h_rotated_xray_list[i*3+1], h_rotated_xray_list[i*3+2]);
		// 	printf("rotated s1 in host is [%f,%f,%f]\n", rotated_s1[0], rotated_s1[1], rotated_s1[2]);
		// 	printf("rotated s1 in device is [%f,%f,%f]\n", h_rotated_s1_list[i*3+0], h_rotated_s1_list[i*3+1], h_rotated_s1_list[i*3+2]);
		// }
		// else{
		// 	break;
		// }
		if (i % 1000 == 0)
		{
			timer.Stop();
			compute_time = timer.Elapsed();
			total_time += compute_time;
			printf("--> Batch [%d]: Memory preparation time: %fms; Compute time: %fms;\n", i,memory_time, compute_time);
			timer.Start();
		}
	}
	//---------> summing the results and output the final array

		int nThreads = 256;
		int nBlocks = (len_result + nThreads - 1) / nThreads;
		dim3 gridSize_face(nBlocks, 1, 1);
		dim3 blockSize_face(nThreads, 1, 1);
		rt_gpu_python_results<<<gridSize_face,blockSize_face>>>(len_coord_list,  d_result_list, d_python_result_list,len_result);

		cudaError = cudaMemcpy(h_python_result_list,d_python_result_list,  python_result_size, cudaMemcpyDeviceToHost);
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! Copy of d_python_result_list has failed.\n");
			nCUDAErrors++;
		}
		printf("Copying from device to host:\n");
printf("  Size: %zu\n", python_result_size);
printf("  Device pointer: %p\n", d_python_result_list);
printf("  Host pointer: %p\n", h_python_result_list);

	printf("Total time spent is: %fms\n", total_time);
	//-----> Free memory
	cudaFree(d_result_list);
	cudaFree(d_python_result_list);
	cudaFree(d_scattering_vector_list);
	cudaFree(d_omega_list);
	cudaFree(d_raw_xray);
	cudaFree(d_omega_axis);
	cudaFree(d_kp_rotation_matrix);
	cudaFree(d_rotated_s1_list);
	cudaFree(d_rotated_xray_list);
	cudaFree(d_coord_list);
	cudaFree(d_label_list);
	cudaFree(d_face);
	cudaFree(d_angles);
	cudaFree(d_increments);
	cudaFree(d_ray_classes);
	cudaFree(d_absorption_lengths);
	free(h_rotated_s1_list);
	free(h_rotated_xray_list);

	return (0);
}