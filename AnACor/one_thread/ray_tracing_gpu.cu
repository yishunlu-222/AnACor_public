#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "GPU_reduction.cuh"

#define DEBUG 0

#include "timer.h"

#define warpSize 32
#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)

__device__ __constant__ int x_max, y_max, z_max, diagonal, len_coord_list, len_result;
__device__ __constant__ float coeff_cr, coeff_bu, coeff_lo, coeff_li, voxel_length_x, voxel_length_y, voxel_length_z;

void print_cuda_error(cudaError_t code)
{
	printf("CUDA error code: %d; string: %s;\n", (int)code, cudaGetErrorString(code));
}

__global__ void rt_gpu_python_results(float *d_result_list, float *d_python_result_list, int h_len_result)
{
	size_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < h_len_result)
	{
		float gpu_absorption = 0;
		for (int j = 0; j < len_coord_list; j++)
		{
			gpu_absorption += exp(-(d_result_list[id * len_coord_list * 2 + 2 * j + 0] + d_result_list[id * len_coord_list * 2 + 2 * j + 1]));
		}
		float gpu_absorption_mean = gpu_absorption / ((float)len_coord_list);
		d_python_result_list[id] = gpu_absorption_mean;
	}
}

__inline__ __device__ void transpose_device(float *input, int rows, int cols, float *output)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output[j * rows + i] = input[i * cols + j];
		}
	}
}

__inline__ __device__ void dot_product_device(const float *A, const float *B, float *C, int m, int n, int p)
{
	//     In the provided example, the dimensions m, n, and p of the matrices are as follows:

	// Matrix A: m x n = 2 x 3 (2 rows, 3 columns)
	// Matrix B: n x p = 3 x 2 (3 rows, 2 columns)
	// Matrix C: m x p = 2 x 2 (2 rows, 2 columns)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < p; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < n; k++)
			{
				sum += A[i * n + k] * B[k * p + j];
			}
			C[i * p + j] = sum;
		}
	}
}

__inline__ __device__ void kp_rotation_device(const float *axis, float theta, float *result)
{
	float x = axis[0];
	float y = axis[1];
	float z = axis[2];
	float c = cosf(theta);
	float s = sinf(theta);

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

__global__ void ray_tracing_rotation(const float *d_omega_axis, float *d_omega_list, float *d_kp_rotation_matrix, float *d_raw_xray, float *d_scattering_vector_list,  float *d_rotated_xray_list, float *d_rotated_s1_list)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	float rotation_matrix_frame_omega[9];
	float rotation_matrix_overall[9];
	float total_rotation_matrix[9];
	float rotated_xray[3];
	float rotated_s1[3];
	if (id < len_result)
	{
		kp_rotation_device(d_omega_axis, d_omega_list[id], rotation_matrix_frame_omega);
		dot_product_device((float *)rotation_matrix_frame_omega, d_kp_rotation_matrix, (float *)rotation_matrix_overall, 3, 3, 3);
		transpose_device((float *)rotation_matrix_overall, 3, 3, (float *)total_rotation_matrix);

		dot_product_device((float *)total_rotation_matrix, d_raw_xray, (float *)rotated_xray, 3, 3, 1);
		d_rotated_xray_list[3 * id] = rotated_xray[0];
		d_rotated_xray_list[3 * id + 1] = rotated_xray[1];
		d_rotated_xray_list[3 * id + 2] = rotated_xray[2];

		float scattering_vector[3] = {d_scattering_vector_list[id * 3],
									  d_scattering_vector_list[id * 3 + 1],
									  d_scattering_vector_list[id * 3 + 2]};
		dot_product_device((float *)total_rotation_matrix, (float *)scattering_vector, (float *)rotated_s1, 3, 3, 1);
		d_rotated_s1_list[3 * id] = rotated_s1[0];
		d_rotated_s1_list[3 * id + 1] = rotated_s1[1];
		d_rotated_s1_list[3 * id + 2] = rotated_s1[2];
	}
}

__inline__ __device__ int cube_face(int *ray_origin, float *ray_direction, int L1)
{
	float t_min = x_max * y_max * z_max, dtemp = 0;
	int face_id = 0;

	// float tx_min = (min_x - ray_origin[2]) / ray_direction[2];
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

	// float tx_max = (max_x - ray_origin[2]) / ray_direction[2];
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

	// float ty_min = (min_y - ray_origin[1]) / ray_direction[1];
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

	// float ty_max = (max_y - ray_origin[1]) / ray_direction[1];
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
	// float tz_min = (min_z - ray_origin[0]) / ray_direction[0];
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

	// float tz_max = (max_z - ray_origin[0]) / ray_direction[0];
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

__global__ void rt_gpu_get_face_overall(int *d_face, int *d_coord_list, float *d_rotated_s1_list, float *d_rotated_xray_list)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t batch_number = blockIdx.y * blockDim.y + threadIdx.y;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1);
	// if (threadIdx.x==3){
	// printf("batch_number=%d, id=%ld, blockIdx.x=%d ,blockDim.x=%d, threadIdx.x=%d, blockIdx.y=%d, blockDim.y=%d, threadIdx.y=%d, is_ray_incomming=%d, pos=%d\n", batch_number, id, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y, is_ray_incomming, pos);
	// }
	int coord[3];
	float ray_direction[3];
	// printf("batch number=%d, len_result=%d\n", batch_number, len_result);
	if (batch_number < len_result)
	{
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
			int face = cube_face(coord, ray_direction, is_ray_incomming);
			// printf("face=%d\n", face);

			d_face[batch_number * len_coord_list * 2 + id] = face;
		}
	}
}


__global__ void rt_gpu_get_face(int *d_face, int *d_coord_list, float *d_rotated_s1_list, float *d_rotated_xray_list,  int batch_number)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1);

	int coord[3];
	float ray_direction[3];

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
		int face = cube_face(coord, ray_direction,is_ray_incomming);
		d_face[id] = face;
	}
}



__inline__ __device__ void get_theta_phi(float *theta, float *phi, float *ray_direction, int L1)
{
	if (L1 == 1)
	{
		ray_direction[0] = -ray_direction[0];
		ray_direction[1] = -ray_direction[1];
		ray_direction[2] = -ray_direction[2];
	}

	if (ray_direction[1] == 0)
	{
		(*theta) = atanf(-ray_direction[2] / (-sqrtf(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]) + 0.001));
		(*phi) = atanf(-ray_direction[0] / (ray_direction[1] + 0.001));
	}
	else
	{
		if (ray_direction[1] < 0)
		{
			(*theta) = atanf(-ray_direction[2] / sqrtf(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			(*phi) = atanf(-ray_direction[0] / (ray_direction[1]));
		}
		else
		{
			if (ray_direction[2] < 0)
			{
				(*theta) = M_PI - atanf(-ray_direction[2] / sqrtf(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			}
			else
			{
				(*theta) = -M_PI - atanf(-ray_direction[2] / sqrtf(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]));
			}
			(*phi) = -atanf(-ray_direction[0] / (-ray_direction[1]));
		}
	}
}

__global__ void rt_gpu_angles(float *d_angles, float *d_rotated_s1_list, float *d_rotated_xray_list, int nBatches, int batch_number)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t batch = (id >> 1);
	int is_ray_incomming = id & 1;

	float theta = 0, phi = 0;
	float ray_direction[3];

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
	float *increment_ratio_x,
	float *increment_ratio_y,
	float *increment_ratio_z,
	float theta,
	float phi,
	int face)
{
	if (face == 1)
	{
		*increment_ratio_x = -1;
		*increment_ratio_y = tanf(M_PI - theta) / cosf(fabs(phi));
		*increment_ratio_z = tanf(phi);
	}
	else if (face == 2)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = 1 / tanf(fabs(phi));
			*increment_ratio_y = tanf(theta) / sinf(fabs(phi));
			*increment_ratio_z = -1;
		}
		else
		{
			*increment_ratio_x = 1 / tanf(fabs(phi));
			*increment_ratio_y = tanf(M_PI - theta) / sinf(fabs(phi));
			*increment_ratio_z = -1;
		}
	}
	else if (face == 3)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = 1 / tanf(fabs(phi));
			*increment_ratio_y = tanf(theta) / sinf(fabs(phi));
			*increment_ratio_z = 1;
		}
		else
		{
			*increment_ratio_x = 1 / (tanf(fabs(phi)));
			*increment_ratio_y = tanf(M_PI - theta) / sinf(fabs(phi));
			*increment_ratio_z = 1;
		}
	}
	else if (face == 4)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = cosf(fabs(phi)) / tanf(fabs(theta));
			*increment_ratio_y = 1;
			*increment_ratio_z = sinf(phi) / tanf(fabs(theta));
		}
		else
		{
			*increment_ratio_x = cosf(fabs(phi)) / (tanf((M_PI - fabs(theta))));
			*increment_ratio_y = 1;
			*increment_ratio_z = sinf(-phi) / (tanf((M_PI - fabs(theta))));
		}
	}
	else if (face == 5)
	{
		if (fabs(theta) < M_PI / 2)
		{
			*increment_ratio_x = cosf(fabs(phi)) / (tanf(fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sinf(phi) / (tanf(fabs(theta)));
		}
		else
		{
			*increment_ratio_x = cosf(fabs(phi)) / (tanf(M_PI - fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sinf(phi) / (tanf(M_PI - fabs(theta)));
		}
	}
	else if (face == 6)
	{
		*increment_ratio_x = -1;
		*increment_ratio_y = tanf(theta) / cosf(phi);
		*increment_ratio_z = tanf(phi);
	}
}

__global__ void rt_gpu_increments(float *d_increments, float *d_angles)
{
	// store increments according to different faces and different thetas
	// and for one single reflection, the increments are the same
	// so we only need to store the increments for one single reflection with
	// different crystal voxel positions
	size_t id = threadIdx.x;
	size_t batch = blockIdx.x;
	int face = id % 6;
	int is_ray_incomming = id / 6.0f;

	float theta = 0, phi = 0;
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

	float ix = 0, iy = 0, iz = 0;
	get_increment_ratio(&ix, &iy, &iz, theta, phi, face + 1);

	d_increments[36 * batch + 3 * threadIdx.x + 0] = ix;
	d_increments[36 * batch + 3 * threadIdx.x + 1] = iy;
	d_increments[36 * batch + 3 * threadIdx.x + 2] = iz;
}

__inline__ __device__ void get_new_coordinates(
	int *new_x, int *new_y, int *new_z,
	int x, int y, int z,
	float increment_ratio_x, float increment_ratio_y, float increment_ratio_z,
	int increment, float theta, int face)
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

__inline__ __device__ void get_distance_2(float *total_length, float s_sum, float increment_ratio_x, float increment_ratio_y, float increment_ratio_z, int face)
{
	float dist_x, dist_y, dist_z;
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
	*total_length = sqrtf(
		(dist_x * voxel_length_x) * (dist_x * voxel_length_x) +
		(dist_y * voxel_length_y) * (dist_y * voxel_length_y) +
		(dist_z * voxel_length_z) * (dist_z * voxel_length_z));
}

__global__ void rt_gpu_absorption(int8_t *d_label_list, int *d_coord_list, int *d_face, float *d_angles, float *d_increments, float *d_result_list, int index)
{
	size_t id = blockIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1); /* the right shift operation effectively divided the value of id by 2 (since shifting the bits to the right by 1 is equivalent to integer division by 2).*/
	float increments[3];
	int face = 0;
	int coord[3];
	float theta, phi;
	__shared__ float s_absorption[1024];
	// __shared__ int s_ray_classes[512];
	int cr_l_2_int = 0;
	int li_l_2_int = 0;
	int bu_l_2_int = 0;
	int lo_l_2_int = 0;

	// Load coordinates
	coord[0] = d_coord_list[3 * pos + 0]; // z
	coord[1] = d_coord_list[3 * pos + 1]; // y
	coord[2] = d_coord_list[3 * pos + 2]; // x

	// Load face
	//face = d_face[index * len_coord_list * 2 + id];
	face = d_face[id];
	// printf("index= %ld face=%d\n", index*len_coord_list*2 + id,face);

	// Load angle
	theta = d_angles[4 * blockIdx.y + 2 * is_ray_incomming];
	// phi = d_angles[4*blockIdx.y + 2*is_ray_incomming + 1];

	// Load Increment
	size_t incr_pos = 36 * blockIdx.y + 18 * is_ray_incomming + 3 * (face - 1);
	// get_increment_ratio(&increments[0], &increments[1], &increments[2], theta, phi, face);
	increments[0] = d_increments[incr_pos + 0];
	increments[1] = d_increments[incr_pos + 1];
	increments[2] = d_increments[incr_pos + 2];

	// Calculate number of iterations of blocks
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
		// if (lpos < diagonal)
		// {
		// 	size_t gpos = blockIdx.x * diagonal + lpos;
		// 	d_ray_classes[gpos] = label;
		// }
	}

	float total_length;
	get_distance_2(&total_length, diagonal, increments[0], increments[1], increments[2], face);

	float cr_l = (total_length * cr_l_2_int) / ((float)diagonal);
	float li_l = (total_length * li_l_2_int) / ((float)diagonal);
	float lo_l = (total_length * lo_l_2_int) / ((float)diagonal);
	float bu_l = (total_length * bu_l_2_int) / ((float)diagonal);

	float absorption = 0;
	float li_absorption = 0;
	float lo_absorption = 0;
	float cr_absorption = 0;
	float bu_absorption = 0;
	s_absorption[threadIdx.x] = coeff_li * li_l + coeff_lo * lo_l + coeff_cr * cr_l + coeff_bu * bu_l;

	__syncthreads();
	absorption = Reduce_SM(s_absorption);

	Reduce_WARP(&absorption);

	__syncthreads();

	// calculation of the absorption for given ray
	if (threadIdx.x == 0)
	{
		// d_absorption[id] = absorption;
		d_result_list[index * len_coord_list * 2 + id] = absorption;
	}
}

__device__ void determine_boundaries(int *s_ray_classes, int offset, int *boundaries, int *class_values, int *boundary_count)
{
	int tid = threadIdx.x % warpSize; // Thread id within the warp

	int val = s_ray_classes[offset + tid];
	int prev_val;

	if (tid == 0)
	{
		prev_val = (threadIdx.x > 0) ? s_ray_classes[offset + tid - 1] : 3; // Fetch directly from shared memory
	}
	else
	{
		prev_val = __shfl_down_sync(0xFFFFFFFF, val, 1); // Get the value of the previous thread in the warp
	}

	if (threadIdx.x != 0 && val != prev_val)
	{

		int local_count = atomicAdd(boundary_count, 1);
		boundaries[local_count] = offset + tid + 1;
		class_values[local_count] = val;
	}
}

__device__ void calculate_distances(int *boundaries, int *class_values, int count, int *distances)
{
	for (int i = 0; i < count - 1; i++)
	{
		distances[i] = boundaries[i + 1] - boundaries[i];
	}
	distances[count - 1] = warpSize - boundaries[count - 1];
}

__global__ void rt_gpu_absorption_shuffle(int8_t *d_label_list, int *d_coord_list, int *d_face, float *d_angles, float *d_increments, float *d_result_list, int index)
{
	size_t id = blockIdx.x;
	int is_ray_incomming = id & 1;
	size_t pos = (id >> 1); /* the right shift operation effectively divided the value of id by 2 (since shifting the bits to the right by 1 is equivalent to integer division by 2).*/
	float increments[3];
	int face = 0;
	int coord[3];
	float theta, phi;
	// __shared__ float s_absorption[1024];
	// extern __shared__ int s_ray_classes[];
	// extern __shared__ int DynamicsharedMemory[];
	// int * s_ray_classes = DynamicsharedMemory;
	// int * boundaries = &DynamicsharedMemory[diagonal];
	// int * class_values = &DynamicsharedMemory[diagonal*2];

	// extern __shared__ int s_ray_classes[];

	__shared__ int s_ray_classes[2048];
	__shared__ int boundaries[128];
	__shared__ int class_values[128];

	__shared__ int boundary_count;
	// __shared__ float cr_l;
	// __shared__ float li_l;
	// __shared__ float bu_l;
	// __shared__ float lo_l;
	int cr_l = 0;
	int li_l = 0;
	int bu_l = 0;
	int lo_l = 0;
	int total_length;
	float absorption;

	if (threadIdx.x == 0)
	{
		boundary_count = 0;
		// cr_l = 0;
		// li_l = 0;
		// bu_l = 0;
		// lo_l = 0;
	}
	// extern __shared__ float s_absorption[];

	// Load coordinates
	coord[0] = d_coord_list[3 * pos + 0]; // z
	coord[1] = d_coord_list[3 * pos + 1]; // y
	coord[2] = d_coord_list[3 * pos + 2]; // x

	// Load face
	face = d_face[index * len_coord_list * 2 + id];
	// printf("index= %ld face=%d\n", index*len_coord_list*2 + id,face);

	// Load angle
	theta = d_angles[4 * blockIdx.y + 2 * is_ray_incomming];
	// phi = d_angles[4*blockIdx.y + 2*is_ray_incomming + 1];

	// Load Increment
	size_t incr_pos = 36 * blockIdx.y + 18 * is_ray_incomming + 3 * (face - 1);
	// get_increment_ratio(&increments[0], &increments[1], &increments[2], theta, phi, face);
	increments[0] = d_increments[incr_pos + 0];
	increments[1] = d_increments[incr_pos + 1];
	increments[2] = d_increments[incr_pos + 2];

	// Calculate number of iterations of blocks
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

		if (
			x < x_max && y < y_max && z < z_max &&
			x >= 0 && y >= 0 && z >= 0)
		{
			size_t cube_pos = INDEX_3D(
				z_max, y_max, x_max,
				z, y, x);
			label = (int)d_label_list[cube_pos];

			if (lpos < diagonal)
			{
				s_ray_classes[lpos] = label;
			}
		}
	}

	// __shared__ int distances[32 * warpSize];
	// int count = 0;

	__syncthreads();
	for (int f = 0; f < nIter; f++)
	{
		int lpos = (f * blockDim.x + threadIdx.x);
		int warpId = lpos / warpSize;
		int laneId = lpos % warpSize;
		if (lpos < diagonal)
		{
			determine_boundaries(s_ray_classes, warpId * warpSize, boundaries, class_values, &boundary_count);
		}
	}
	__syncthreads();
	if (blockIdx.x == 100)
	{
		printf("boundary_count=%d\n", boundary_count);
	}
	if (threadIdx.x == 0)
	{
		for (int count; count < boundary_count; count++)
		{
			if (count == 0)
			{
				cr_l += boundaries[count];
			}
			else
			{
				int distance = boundaries[count] - boundaries[count - 1];
				if (class_values[count] == 3)
					cr_l += distance;
				else if (class_values[count] == 1)
					li_l += distance;
				else if (class_values[count] == 2)
					lo_l += distance;
				else if (class_values[count] == 4)
					bu_l += distance;
				else
				{
				}
			}
		}
		absorption = coeff_li * li_l + coeff_lo * lo_l + coeff_cr * cr_l + coeff_bu * bu_l;
		d_result_list[index * len_coord_list * 2 + id] = absorption;
	}
	// Only a fraction of the threads will calculate distances to reduce redundant work.
	// if (laneId < boundary_count) {
	//     calculate_distances(boundaries + warpId * 32, class_values + warpId * 32, boundary_count, distances + warpId * 32);
	// }
	// get_distance_2(&total_length, diagonal, increments[0], increments[1], increments[2], face);

	// float cr_l = (total_length * cr_l_2_int) / ((float)diagonal);
	// float li_l = (total_length * li_l_2_int) / ((float)diagonal);
	// float lo_l = (total_length * lo_l_2_int) / ((float)diagonal);
	// float bu_l = (total_length * bu_l_2_int) / ((float)diagonal);

	// float absorption = 0;
	// float li_absorption = 0;
	// float lo_absorption = 0;
	// float cr_absorption = 0;
	// float bu_absorption = 0;
	// s_absorption[threadIdx.x] = coeff_li * li_l + coeff_lo * lo_l + coeff_cr * cr_l + coeff_bu * bu_l;

	// __syncthreads();
	// absorption = Reduce_SM(s_absorption);

	// Reduce_WARP(&absorption);

	// __syncthreads();

	// calculation of the absorption for given ray

	// }
}

float *ray_tracing_gpu_single(int rotated_s1_size, int rotated_xray_size, int h_len_result,int h_x_max, int  h_y_max, int h_z_max, int h_diagonal, int h_len_coord_list, float *coefficients, float * voxel_size, size_t result_size, size_t python_result_size,size_t scattering_vector_list_size,size_t omega_list_size,size_t raw_xray_size,size_t omega_axis_size, size_t kp_rotation_matrix_size, size_t coord_list_size, size_t cube_size,size_t face_size,size_t angle_size,size_t increments_size,size_t ray_classes_size,size_t absorption_size,float* h_result_list, const float *scattering_vector_list,const float *omega_list,const float *raw_xray,const float *omega_axis,const float *kp_rotation_matrix,int *coord_list,int8_t *label_list_1d,float * h_python_result_list){
			//----------> Memory allocation
		//---------> Allocating memory on the device
		// global memory
		cudaError_t cudaError;
		int nCUDAErrors = 0;
		float *d_result_list;		 // contains i reflections, each with j rays
		float *d_python_result_list; // contains i reflections
		float *d_scattering_vector_list;
		float *d_omega_list;
		float *d_raw_xray;
		float *d_omega_axis;
		float *d_kp_rotation_matrix;
		float *d_rotated_s1_list;
		float *d_rotated_xray_list;
		int *d_coord_list;
		int8_t *d_label_list;

		// individual memory for each relfection
		int *d_face;
		float *d_angles;
		float *d_increments;
		int *d_ray_classes;
		float *d_absorption_lengths;

		// output memory
		float *h_rotated_s1_list = (float *)malloc(rotated_s1_size);
		float *h_rotated_xray_list = (float *)malloc(rotated_xray_size);
		// float *h_result_list = (float *)malloc(result_size);

		/* creating  global memory for constants */
		// int   x_max,y_max,z_max,diagonal,len_coord_list, len_result;
		// float coeff_cr, coeff_bu, coeff_lo, coeff_li,voxel_length_x,voxel_length_y,voxel_length_z;

		checkCudaErrors(cudaMemcpyToSymbol(len_result, &h_len_result, sizeof(h_len_result)));
		checkCudaErrors(cudaMemcpyToSymbol(x_max, &h_x_max, sizeof(h_x_max)));
		checkCudaErrors(cudaMemcpyToSymbol(y_max, &h_y_max, sizeof(h_y_max)));
		checkCudaErrors(cudaMemcpyToSymbol(z_max, &h_z_max, sizeof(h_z_max)));
		checkCudaErrors(cudaMemcpyToSymbol(diagonal, &h_diagonal, sizeof(h_diagonal)));
		checkCudaErrors(cudaMemcpyToSymbol(len_coord_list, &h_len_coord_list, sizeof(h_len_coord_list)));
		checkCudaErrors(cudaMemcpyToSymbol(coeff_li, &coefficients[0], sizeof(coefficients[0])));
		checkCudaErrors(cudaMemcpyToSymbol(coeff_lo, &coefficients[1], sizeof(coefficients[1])));
		checkCudaErrors(cudaMemcpyToSymbol(coeff_cr, &coefficients[2], sizeof(coefficients[2])));
		checkCudaErrors(cudaMemcpyToSymbol(coeff_bu, &coefficients[3], sizeof(coefficients[3])));
		checkCudaErrors(cudaMemcpyToSymbol(voxel_length_z, &voxel_size[0], sizeof(voxel_size[0])));
		checkCudaErrors(cudaMemcpyToSymbol(voxel_length_y, &voxel_size[1], sizeof(voxel_size[1])));
		checkCudaErrors(cudaMemcpyToSymbol(voxel_length_x, &voxel_size[2], sizeof(voxel_size[2])));
		printf("result_size: %ld\n", result_size);
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
		float memory_time = 0;
		timer.Start();

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
		// cudaError = cudaMemset(d_ray_classes, 0, ray_classes_size);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Memset d_ray_classes.\n");
		// 	print_cuda_error(cudaError);
		// 	nCUDAErrors++;
		// }
		timer.Stop();
		memory_time = timer.Elapsed();
		float single_time = memory_time;
		printf("Total time: %0.3f ms; memory_time : %0.3f ms\n", single_time, memory_time);
		cudaDeviceSynchronize();

		//---------> Kernel execution
		float precompute_time = 0;
		printf("len result: %d\n", h_len_result);
		timer.Start();
		if (nCUDAErrors == 0)
		{

			{
				int nThreads = 256;
				int nBlocks = (h_len_result + nThreads - 1) / nThreads;
				dim3 gridSize_rotation(nBlocks, 1, 1);
				dim3 blockSize_rotation(nThreads, 1, 1);

				ray_tracing_rotation<<<gridSize_rotation, blockSize_rotation>>>(d_omega_axis, d_omega_list, d_kp_rotation_matrix, d_raw_xray, d_scattering_vector_list, d_rotated_xray_list, d_rotated_s1_list);
			}

			// {
			// 	int nThreads_x = 256;
			// 	int nThreads_y = 4;
			// 	int nBlocks_x = ((h_len_coord_list * 2) + nThreads_x - 1) / nThreads_x;
			// 	int nBlocks_y = (h_len_result + nThreads_y - 1) / nThreads_y;
			// 	dim3 gridSize_face(nBlocks_x, nBlocks_y);
			// 	dim3 blockSize_face(nThreads_x, nThreads_y);
			// 	printf("nBlocks_x: %d\n", nBlocks_x);
			// 	printf("nBlocks_y: %d\n", nBlocks_y);
			// 	printf("nThreads_x: %d\n", nThreads_x);
			// 	printf("nThreads_y: %d\n", nThreads_y);
			// 	rt_gpu_get_face_overall<<<gridSize_face, blockSize_face>>>(
			// 		d_face,
			// 		d_coord_list,
			// 		d_rotated_s1_list,
			// 		d_rotated_xray_list);
			// 	// output_size = h_len_coord_list * 2 * h_len_result; 0101=>s1,s0,s1,s0
			// }
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
			if (DEBUG)
			{
				printf("No CUDA error.\n");
				printf("Rotation matrices are calculated\n");
			}
		}
		//-----> Copy chunk of output data to host
		// cudaError = cudaMemcpy(h_rotated_xray_list, d_rotated_xray_list, rotated_xray_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_rotated_xray_list has failed.\n");
		// 	nCUDAErrors++;
		// }
		// cudaError = cudaMemcpy(h_rotated_s1_list, d_rotated_s1_list, rotated_s1_size, cudaMemcpyDeviceToHost);
		// if (cudaError != cudaSuccess)
		// {
		// 	printf("ERROR! Copy of d_rotated_s1_list has failed.\n");
		// 	nCUDAErrors++;
		// }
		cudaDeviceSynchronize();
		timer.Stop();
		precompute_time = timer.Elapsed();
		single_time += precompute_time;
		printf("Total time: %0.3f ms; Precompute time: %0.3f ms\n", single_time, precompute_time);

		float compute_time = 0;
		timer.Start();

		for (int i = 0; i < (int)h_len_result; i++)
		{

			if (nCUDAErrors == 0)

			{

				//**************** Calculate faces **************
				{
					int nThreads = 128;
					int nBlocks = ((h_len_coord_list * 2) + nThreads - 1) / nThreads;

					dim3 gridSize_face(nBlocks, 1, 1);
					dim3 blockSize_face(nThreads, 1, 1);
					rt_gpu_get_face<<<gridSize_face, blockSize_face>>>(
						d_face,
						d_coord_list,
						d_rotated_s1_list,
						d_rotated_xray_list, i);
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
					// float voxel_length_z = voxel_size[0];
					// float voxel_length_y = voxel_size[1];
					// float voxel_length_x = voxel_size[2];
					// float coeff_li = coefficients[0];
					// float coeff_lo = coefficients[1];
					// float coeff_cr = coefficients[2];
					// float coeff_bu = coefficients[3];

					int nBlocks = h_len_coord_list * 2; // one block for one crystal voxel
					int nThreads = 256;					// 256:49s ,128:49s 32:52s,512:fail,320:55s
					dim3 gridSize_face(nBlocks, 1, 1);
					dim3 blockSize_face(nThreads, 1, 1);

					rt_gpu_absorption<<<gridSize_face, blockSize_face>>>(

						d_label_list,
						d_coord_list,
						d_face,
						d_angles,
						d_increments, d_result_list, i); //,sizeof(int)*h_diagonal*3

					// rt_gpu_absorption_shuffle<<<gridSize_face, blockSize_face>>>(
					// 	d_label_list,
					// 	d_coord_list,
					// 	d_face,
					// 	d_angles,
					// 	d_increments, d_result_list, i);
				}
			}

			cudaDeviceSynchronize();

			if (i % 1000 == 0)
			{
				timer.Stop();
				compute_time = timer.Elapsed();
				single_time += compute_time;
				printf("--> Batch [%d]: total time: %0.3fms; Compute time: %0.3fms;\n", i, single_time, compute_time);
				timer.Start();
			}
		}
		//---------> summing the results and output the final array

		int nThreads = 256;
		int nBlocks = (h_len_result + nThreads - 1) / nThreads;
		dim3 gridSize_face(nBlocks, 1, 1);
		dim3 blockSize_face(nThreads, 1, 1);
		rt_gpu_python_results<<<gridSize_face, blockSize_face>>>(d_result_list, d_python_result_list, h_len_result);

		cudaError = cudaMemcpy(h_python_result_list, d_python_result_list, python_result_size, cudaMemcpyDeviceToHost);
		if (cudaError != cudaSuccess)
		{
			printf("ERROR! Copy of d_python_result_list has failed.\n");
			nCUDAErrors++;
		}
		// printf("Copying from device to host:\n");
		// printf("  Size: %zu\n", python_result_size);
		// printf("  Device pointer: %p\n", d_python_result_list);
		// printf("  Host pointer: %p\n", h_python_result_list);

		printf("Total time spent is: %fms\n", single_time);
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

		return h_python_result_list;
}


void transpose(float *input, int rows, int cols, float *output);
void dot_product(const float *A, const float *B, float *C, int m, int n, int p);
void kp_rotation(const float *axis, float theta, float *result);

// size_t multiplier(int multiplier_1, int multiplier_2)
// {
// 	return multiplier_1 * multiplier_2;
// }

int ray_tracing_gpu_overall_kernel(int low, int up,
								   int *coord_list,
								   int h_len_coord_list,
								   const float *scattering_vector_list, const float *omega_list,
								   const float *raw_xray,
								   const float *omega_axis, const float *kp_rotation_matrix,
								   int h_len_result,
								   float *voxel_size, float *coefficients,
								   int8_t *label_list_1d, int *shape, int full_iteration,
								   int store_paths, float *h_result_list, int *h_face, float *h_angles, float *h_python_overall_result_list)
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

	printf("--> GPU info:");

	int deviceID = 0; // Replace with the desired device ID
	cudaSetDevice(deviceID);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	unsigned long sharedMemPerBlock = deviceProp.sharedMemPerBlock;
	if (deviceProp.major >= 1)
	{
		printf("Device %d: %s\n", deviceID, deviceProp.name);
		printf("Shared memory per block: %lu bytes\n", sharedMemPerBlock);
	}
	
	//---------> Checking memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	printf("--> GPU info: Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
	float total_time = 0;
	int n_chunks = 1;
	int h_z_max = shape[0], h_y_max = shape[1], h_x_max = shape[2];
	int64_t h_len_result_coord=(int64_t)h_len_result * (int64_t)h_len_coord_list;

	int h_diagonal = h_x_max * sqrtf(3);
	size_t cube_size = h_x_max * h_y_max * h_z_max * sizeof(int8_t);
	size_t face_size =  h_len_coord_list * 2 * sizeof(int);
	size_t absorption_size = h_len_coord_list * 2 * sizeof(float);
	size_t angle_size = 4 * sizeof(float);
	size_t increments_size = 36 * sizeof(float);
	size_t ray_classes_size = h_diagonal * h_len_coord_list * 2 * sizeof(int);
	size_t coord_list_size = h_len_coord_list * 3 * sizeof(int);
	size_t ray_directions_size = 3 * sizeof(float);

	// size_t   result_size = multiplier(h_len_result,h_len_coord_list) * h_len_coord_list * 2 * sizeof(float);
	size_t result_size = h_len_result_coord * 2 * sizeof(float);
	size_t python_result_size = h_len_result * sizeof(float); // my desktop doesnt have enough memory to store the whole result list, so take a half of it to test
	size_t scattering_vector_list_size = h_len_result * sizeof(float) * 3;
	size_t omega_list_size = h_len_result * sizeof(float);
	size_t raw_xray_size = 3 * sizeof(float);
	size_t omega_axis_size = 3 * sizeof(float);
	size_t kp_rotation_matrix_size = 9 * sizeof(float);
	size_t rotated_s1_size = h_len_result * sizeof(float) * 3;
	size_t rotated_xray_size = h_len_result * sizeof(float) * 3;
	// size_t face_size = h_len_result * h_len_coord_list * 2 * sizeof(int);
	printf("len_coord_list %d \n", h_len_coord_list);
	printf("h_len_result %d \n", h_len_result);
	size_t  total_memory_required_bytes = face_size + angle_size + increments_size + absorption_size + cube_size + ray_classes_size + coord_list_size + ray_directions_size + result_size + scattering_vector_list_size + omega_list_size + raw_xray_size + omega_axis_size + kp_rotation_matrix_size + rotated_s1_size + rotated_xray_size;

	printf("total_memory_required_bytes %f \n", total_memory_required_bytes);
	printf("--> DEBUG: Total memory required %0.3f MB.\n", (double)total_memory_required_bytes / (1024.0 * 1024.0));

	if (total_memory_required_bytes > free_mem)
	{
		printf("--> DEBUG: Total memory required %0.3f MB.\n", (double)
																	   total_memory_required_bytes /
																   (1024.0 * 1024.0));
		printf(" Not enough memory! Input data is splitted into chunks.\n");
		// return (1);
		n_chunks = (total_memory_required_bytes + free_mem - 1) / free_mem;
	}
	else
	{
	}
	
	float *h_python_result_list = (float *)malloc(python_result_size);
	for (int chunk = 0; chunk < n_chunks; chunk++)
	{
		h_python_overall_result_list=ray_tracing_gpu_single(rotated_s1_size, rotated_xray_size, h_len_result, h_x_max, h_y_max, h_z_max, h_diagonal, h_len_coord_list, coefficients, voxel_size, result_size, python_result_size, scattering_vector_list_size, omega_list_size, raw_xray_size, omega_axis_size, kp_rotation_matrix_size, coord_list_size, cube_size, face_size, angle_size, increments_size, ray_classes_size, absorption_size, h_result_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, coord_list, label_list_1d, h_python_result_list);

	}

	return (0);
}