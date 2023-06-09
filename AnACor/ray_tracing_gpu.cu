#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <stdio.h>

#define DEBUG 1

#include "timer.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)

void print_cuda_error(cudaError_t code){
	printf("CUDA error code: %d; string: %s;\n", (int) code, cudaGetErrorString(code));
}

__inline__ __device__ int cube_face(int64_t *ray_origin, double *ray_direction, int x_max, int y_max, int z_max, int L1){
	double t_min = x_max*y_max*z_max, dtemp = 0;
	int face_id = 0;
	
	//double tx_min = (min_x - ray_origin[2]) / ray_direction[2];
	dtemp = (0 - ray_origin[2]) / ray_direction[2];
	if(dtemp >= 0) {
		t_min = dtemp;
		face_id = 1;
	}
    //double tx_max = (max_x - ray_origin[2]) / ray_direction[2];
	dtemp = (x_max - ray_origin[2]) / ray_direction[2];
	if(dtemp >= 0 && dtemp < t_min) {
		t_min = dtemp;
		face_id = 2;
	}
    //double ty_min = (min_y - ray_origin[1]) / ray_direction[1];
	dtemp = (0 - ray_origin[1]) / ray_direction[1];
	if(dtemp >= 0 && dtemp < t_min) {
		t_min = dtemp;
		face_id = 3;
	}
    //double ty_max = (max_y - ray_origin[1]) / ray_direction[1];
	dtemp = (y_max - ray_origin[1]) / ray_direction[1];
	if(dtemp >= 0 && dtemp < t_min) {
		t_min = dtemp;
		face_id = 4;
	}
    //double tz_min = (min_z - ray_origin[0]) / ray_direction[0];
	dtemp = (0 - ray_origin[0]) / ray_direction[0];
	if(dtemp >= 0 && dtemp < t_min) {
		t_min = dtemp;
		face_id = 5;
	}
    //double tz_max = (max_z - ray_origin[0]) / ray_direction[0];
	dtemp = (z_max - ray_origin[0]) / ray_direction[0];
	if(dtemp >= 0 && dtemp < t_min) {
		t_min = dtemp;
		face_id = 6;
	}
	
    if (face_id == 1) { // tx_min
        return L1 ? 1 : 6;
    }
    else if (face_id == 2) { // tx_max
        return L1 ? 6 : 1;
    }
    else if (face_id == 3) { // 3 ty_min
        return L1 ? 5 : 4;
    }
    else if (face_id == 4) { // 4 ty_max
        return L1 ? 4 : 5;
    }
    else if (face_id == 5) { // 5 tz_min
        return L1 ? 3 : 2;
    }
    else if (face_id == 6) { // 6 tz_max
        return L1 ? 2 : 3;
    }
    else {
		return 0;
	}
}

__global__ void rt_gpu_get_face(int *d_face, int64_t *d_coord_list, double *d_rotated_s1, double *d_xray, int x_max, int y_max, int z_max, int len_coord_list) {
	size_t id = blockIdx.x*blockDim.x + threadIdx.x;
	int is_ray_incomming = id&1;
	size_t pos = (id>>1);
	
	int64_t coord[3];
	double ray_direction[3];
	
	if(pos<len_coord_list) {
		coord[0] = d_coord_list[3*pos + 0];
		coord[1] = d_coord_list[3*pos + 1];
		coord[2] = d_coord_list[3*pos + 2];
		
		if(is_ray_incomming==1) {
			ray_direction[0] = d_xray[0];
			ray_direction[1] = d_xray[2];
			ray_direction[2] = d_xray[1];
		}
		else {
			ray_direction[0] = d_rotated_s1[0];
			ray_direction[1] = d_rotated_s1[2];
			ray_direction[2] = d_rotated_s1[1];
		}
		
		int face = cube_face(coord, ray_direction, x_max, y_max, z_max, is_ray_incomming);
		d_face[id] = face;
	}
}

__inline__ __device__ void get_theta_phi(double *theta, double *phi, double *ray_direction, int L1){
	if(L1 == 1){
		ray_direction[0] = -ray_direction[0];
		ray_direction[1] = -ray_direction[1];
		ray_direction[2] = -ray_direction[2];
	}
	
    if (ray_direction[1] == 0) {
		(*theta) = atan(-ray_direction[2] / (-sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1]) + 0.001));
		(*phi) = atan(-ray_direction[0] / (ray_direction[1] + 0.001));
    }
    else {
        if (ray_direction[1] < 0) {
            (*theta) = atan(-ray_direction[2] / sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1]));
            (*phi) = atan(-ray_direction[0] / (ray_direction[1]));
        }
        else {
            if (ray_direction[2] < 0) {
                (*theta) = M_PI - atan(-ray_direction[2] / sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1]));
            }
            else {
                (*theta) = -M_PI - atan(-ray_direction[2] / sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1]));
            }
            (*phi) = -atan(-ray_direction[0] / (-ray_direction[1]));
        }
    }
}

__global__ void rt_gpu_angles(double *d_angles, double *d_rotated_s1, double *d_xray, int nBatches){
	size_t id = blockIdx.x*blockDim.x + threadIdx.x;
	size_t batch = (id>>1);
	int is_ray_incomming = id&1;
	
	double theta = 0, phi = 0;
	double ray_direction[3];
	
	if(batch < nBatches) {
		if(is_ray_incomming==1) {
			ray_direction[0] = d_xray[0];
			ray_direction[1] = d_xray[1];
			ray_direction[2] = d_xray[2];
		}
		else {
			ray_direction[0] = d_rotated_s1[0];
			ray_direction[1] = d_rotated_s1[1];
			ray_direction[2] = d_rotated_s1[2];
		}
		
		get_theta_phi(&theta, &phi, ray_direction, is_ray_incomming);
		
		//printf("pos=[%d; %d] theta=%f; phi=%f;\n", (int) (2*id + 0), (int) (2*id + 1), theta, phi);
		
		d_angles[2*id + 0] = theta;
		d_angles[2*id + 1] = phi;
	}
}

__inline__ __device__ void get_increment_ratio(
	double *increment_ratio_x, 
	double *increment_ratio_y, 
	double *increment_ratio_z,
	double theta,
	double phi,
	int face
){
	if (face == 1) {
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
		*increment_ratio_z = tan(phi);
	}
	else if (face == 2) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
		else {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
	}
	else if (face == 3) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
		else {
			*increment_ratio_x = 1 / (tan(fabs(phi)));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
	}
	else if (face == 4) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(phi) / tan(fabs(theta));
		}
		else {
			*increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
		}
	}
	else if (face == 5) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(fabs(theta)));
		}
		else {
			*increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(M_PI - fabs(theta)));
		}
	}
	else if (face == 6) {
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(theta) / cos(phi);
		*increment_ratio_z = tan(phi);
	}
}

__global__ void rt_gpu_increments(double *d_increments, double *d_angles){
	size_t id = threadIdx.x;
	size_t batch = blockIdx.x;
	int face = id%6;
	int is_ray_incomming = id/6;
	
	double theta = 0, phi = 0;
	if(is_ray_incomming==1) {
		theta = d_angles[4*batch + 2 + 0];
		phi = d_angles[4*batch + 2 + 1];
	}
	else {
		theta = d_angles[4*batch + 0];
		phi = d_angles[4*batch + 1];
	}
	
	double ix = 0, iy = 0, iz=0;
	get_increment_ratio(&ix, &iy, &iz, theta, phi, face + 1);
	
	d_increments[36*batch + 3*threadIdx.x + 0] = ix;
	d_increments[36*batch + 3*threadIdx.x + 1] = iy;
	d_increments[36*batch + 3*threadIdx.x + 2] = iz;
}

__inline__ __device__ void get_new_coordinates(
	int *new_x, int *new_y, int *new_z,
	int64_t x, int64_t y, int64_t z,
	double increment_ratio_x, double increment_ratio_y, double increment_ratio_z,
	int increment, double theta, int face
){
	if (face == 1) {
		if (theta > 0) {
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = (int) (x - increment * increment_ratio_x); 
			*new_y = (int) (y - increment * increment_ratio_y);
			*new_z = (int) (z - increment * increment_ratio_z);
		}
		else {
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = (int) (x - increment * increment_ratio_x + 0.5);
			*new_y = (int) (y - increment * increment_ratio_y + 0.5);
			*new_z = (int) (z - increment * increment_ratio_z + 0.5);
		}
	}
	else if (face == 2) {
		if (fabs(theta) < M_PI / 2) {
			if (theta > 0) {
				*new_x = (int) (x + -1 * increment * increment_ratio_x);
				*new_y = (int) (y - increment * increment_ratio_y);
				*new_z = (int) (z + increment * increment_ratio_z);
			}
			else {
				*new_x = (int) (x + -1 * increment * increment_ratio_x + 0.5);
				*new_y = (int) (y - increment * increment_ratio_y + 0.5);
				*new_z = (int) (z + increment * increment_ratio_z + 0.5);
			}
		}
		else {
			if (theta > 0) {
				*new_x = (int) (x + 1 * increment * increment_ratio_x);
				*new_y = (int) (y - increment * increment_ratio_y);
				*new_z = (int) (z + increment * increment_ratio_z);
			}
			else {
				*new_x = (int) (x + 1 * increment * increment_ratio_x + 0.5);
				*new_y = (int) (y - increment * increment_ratio_y + 0.5);
				*new_z = (int) (z + increment * increment_ratio_z + 0.5);
			}
		}
	}
	else if (face == 3) {
		if (fabs(theta) < M_PI / 2) {
			if (theta > 0) {
				*new_x = (int) (x + -1 * increment * increment_ratio_x);
				*new_y = (int) (y - increment * increment_ratio_y);
				*new_z = (int) (z + increment * increment_ratio_z);
			}
			else {
				*new_x = (int) (x + -1 * increment * increment_ratio_x + 0.5);
				*new_y = (int) (y - increment * increment_ratio_y + 0.5);
				*new_z = (int) (z + increment * increment_ratio_z + 0.5);
			}
		}
		else {
			if (theta > 0) {
				*new_x = (int) (x + 1 * increment * increment_ratio_x);
				*new_y = (int) (y - increment * increment_ratio_y);
				*new_z = (int) (z + increment * 1);
			}
			else {
				*new_x = (int) (x + 1 * increment * increment_ratio_x + 0.5);
				*new_y = (int) (y - increment * increment_ratio_y + 0.5);
				*new_z = (int) (z + increment * 1 + 0.5);
			}
		}
	}
	else if (face == 4) {
		if (fabs(theta) < M_PI / 2) {
			*new_x = (int) (x + -1 * increment * increment_ratio_x);
			*new_y = (int) (y - increment * increment_ratio_y);
			*new_z = (int) (z + increment * increment_ratio_z);
		}
		else {
			*new_x = (int) (x + 1 * increment * increment_ratio_x);
			*new_y = (int) (y - increment * increment_ratio_y);
			*new_z = (int) (z + increment * increment_ratio_z);
		}
	}
	else if (face == 5) {
		if (fabs(theta) < M_PI / 2) {
			*new_x = (int) (x + -1 * increment * increment_ratio_x + 0.5);
			*new_y = (int) (y - increment * increment_ratio_y + 0.5);
			*new_z = (int) (z + increment * increment_ratio_z + 0.5);
		}
		else {
			*new_x = (int) (x + 1 * increment * increment_ratio_x + 0.5);
			*new_y = (int) (y - increment * increment_ratio_y + 0.5);
			*new_z = (int) (z - increment * increment_ratio_z + 0.5);
		}
	}
	else if (face == 6) {
		if (theta > 0) {
			*new_x = (int) (x + increment * increment_ratio_x);
			*new_y = (int) (y - increment * increment_ratio_y);
			*new_z = (int) (z + increment * increment_ratio_z);
		}
		else {
			*new_x = (int) (x + increment * increment_ratio_x + 0.5);
			*new_y = (int) (y - increment * increment_ratio_y + 0.5);
			*new_z = (int) (z + increment * increment_ratio_z + 0.5);
		}
	}
}


__global__ void rt_gpu_ray_classes(int *d_ray_classes, int8_t *d_label_list, int64_t *d_coord_list, int *d_face, double *d_angles, double *d_increments, int x_max, int y_max, int z_max, int diagonal){
	size_t id = blockIdx.x;
	int is_ray_incomming = id&1;
	size_t pos = (id>>1);
	double increments[3];
	int face = 0;
	int64_t coord[3];
	double theta, phi;
	
	int cr_l_2_int = 0;
	int li_l_2_int = 0;
	int bu_l_2_int = 0;
	int lo_l_2_int = 0;
	
	// Load coordinates
	coord[0] = d_coord_list[3*pos + 0];
	coord[1] = d_coord_list[3*pos + 1];
	coord[2] = d_coord_list[3*pos + 2];
	
	
	// Load face
	face = d_face[id];
	
	// Load angle
	theta = d_angles[4*blockIdx.y + 2*is_ray_incomming];
	//phi = d_angles[4*blockIdx.y + 2*is_ray_incomming + 1];
	
	// Load Increment
	size_t incr_pos = 36*blockIdx.y + 18*is_ray_incomming + 3*(face - 1);
	//get_increment_ratio(&increments[0], &increments[1], &increments[2], theta, phi, face);
	increments[0] = d_increments[incr_pos + 0];
	increments[1] = d_increments[incr_pos + 1];
	increments[2] = d_increments[incr_pos + 2];
	
	// Calculate number of iterations
	int nIter = (int) ((diagonal + blockDim.x - 1)/blockDim.x);
	for(int f = 0; f < nIter; f++){
		// calculate position based on threads id
		// check if the position is within a cube_face
		// write into ray_direction
		int lpos = (f*blockDim.x + threadIdx.x);
		int x, y, z;
		get_new_coordinates(
			&x, &y, &z,
			coord[2], coord[1], coord[0],
			increments[0], increments[1], increments[2],
			lpos, theta, face
		);
		int label = 0;
		if(
			x < x_max && y < y_max && z < z_max && 
			x >= 0 && y >= 0 && z >= 0
		) {
			size_t cube_pos = INDEX_3D(
				z_max, y_max, x_max,
				z, y, x
			);
			label = (int) d_label_list[cube_pos];
		}
		if(lpos<diagonal){
			size_t gpos = blockIdx.x*diagonal + lpos;
			
			d_ray_classes[gpos] = label;
			
			if (label == 3) cr_l_2_int++;
			else if (label == 1) li_l_2_int++;
			else if (label == 2) lo_l_2_int++;
			else if (label == 4) bu_l_2_int++;
			else {
			}
		}
	}
	
	// calculation of the absorption for given ray
	
}




// ******************************************************************
// ******************************************************************
// ******************************************************************

int ray_tracing_path(int *h_face, double *h_angles, int *h_ray_classes, double *h_absorption, int64_t *h_coord_list, int64_t len_coord_list, double *h_rotated_s1, double *h_xray, double *voxel_size, double *coefficients, int8_t *h_label_list_1d, int64_t *shape){
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess || devCount==0) {
		printf("ERROR: CUDA capable device not found!\n");
		return (1);
	}
	
	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
	
	//---------> Checking memory
	size_t free_mem,total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("--> DEBUG: Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float) total_mem)/(1024.0*1024.0), (float) free_mem/(1024.0*1024.0));
	int64_t diagonal = x_max*sqrt(3);
	int64_t cube_size = x_max*y_max*z_max*sizeof(int8_t);
	int64_t face_size = len_coord_list*2*sizeof(int);
	int64_t absorbtion_size = len_coord_list*2*sizeof(double);
	int64_t angle_size = 4*sizeof(double);
	int64_t increments_size = 36*sizeof(double);
	int64_t ray_classes_size = diagonal*len_coord_list*2*sizeof(int);
	int64_t coord_list_size = len_coord_list*3*sizeof(int64_t);
	int64_t ray_directions_size = 3*sizeof(double);
	size_t total_memory_required_bytes = face_size + angle_size + increments_size + absorbtion_size + cube_size + ray_classes_size + coord_list_size + ray_directions_size;
	if(DEBUG) printf("--> DEBUG: Total memory required %0.3f MB.\n", (float) total_memory_required_bytes/(1024.0*1024.0));
	if(total_memory_required_bytes>free_mem) {
		printf("ERROR: Not enough memory! Input data is too big for the device.\n");
		return(1);
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
	
	cudaError = cudaMalloc((void **) &d_face, face_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_face\n");
		d_face = NULL;
	}
	cudaError = cudaMalloc((void **) &d_angles, angle_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_angles\n");
		d_angles = NULL;
	}
	cudaError = cudaMalloc((void **) &d_increments, increments_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_increments\n");
		d_increments = NULL;
	}
	cudaError = cudaMalloc((void **) &d_ray_classes, ray_classes_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_ray_classes\n");
		d_ray_classes = NULL;
	}
	cudaError = cudaMalloc((void **) &d_absorption, absorbtion_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_absorption\n");
		d_absorption = NULL;
	}
	cudaError = cudaMalloc((void **) &d_coord_list, coord_list_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_coord_list\n");
		d_coord_list = NULL;
	}
	cudaError = cudaMalloc((void **) &d_rotated_s1, ray_directions_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_rotated_s1\n");
		d_rotated_s1 = NULL;
	}
	cudaError = cudaMalloc((void **) &d_xray, ray_directions_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_xray\n");
		d_xray = NULL;
	}
	cudaError = cudaMalloc((void **) &d_label_list, cube_size);
	if(cudaError != cudaSuccess) { 
		nCUDAErrors++; 
		printf("ERROR: memory allocation d_label_list\n");
		d_label_list = NULL;
	}
	
	//---------> Memory copy and preparation
	GpuTimer timer;
	double memory_time = 0;
	timer.Start();
	cudaError = cudaMemcpy(d_label_list, h_label_list_1d, cube_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Memcopy d_label_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(d_coord_list, h_coord_list, coord_list_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Memcopy d_coord_list.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(d_rotated_s1, h_rotated_s1, ray_directions_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Memcopy d_rotated.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(d_xray, h_xray, ray_directions_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Memcopy d_xray.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	cudaError = cudaMemset(d_ray_classes, 0, ray_classes_size);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Memset d_ray_classes.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	timer.Stop();
	memory_time = timer.Elapsed();
	
	cudaDeviceSynchronize();
	
	double compute_time = 0;
	if(nCUDAErrors==0){
		timer.Start();
		
		//**************** Calculate faces **************
		{
			int nThreads = 128;
			int nBlocks = ((len_coord_list*2) + nThreads - 1)/nThreads;
			int x_max = shape[2];
			int y_max = shape[1];
			int z_max = shape[0];
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_get_face<<<gridSize_face , blockSize_face>>>(
				d_face, 
				d_coord_list, 
				d_rotated_s1, 
				d_xray, 
				x_max, y_max, z_max, (int) len_coord_list
			);
		}
		
		{
			int nBatches = 1;
			int nThreads = 128;
			int nBlocks = ((nBatches*2) + nThreads - 1)/nThreads;
			dim3 gridSize_face(nBlocks, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_angles<<<gridSize_face , blockSize_face>>>(
				d_angles, 
				d_rotated_s1, 
				d_xray, 
				nBatches
			);
		}
		
		{
			int nBatches = 1;
			int nThreads = 12;
			dim3 gridSize_face(nBatches, 1, 1);
			dim3 blockSize_face(nThreads, 1, 1);
			rt_gpu_increments<<<gridSize_face , blockSize_face>>>(
				d_increments,
				d_angles
			);
		}
		
		//---------> error check
		cudaError = cudaGetLastError();
		if(cudaError != cudaSuccess){
			printf("ERROR! GPU Kernel error.\n");
			print_cuda_error(cudaError);
			nCUDAErrors++; 
		}
		else {
			printf("No CUDA error.\n");
		}
		
		
		{
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
		}
		
		
		timer.Stop();
		compute_time = timer.Elapsed();
	}
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel error.\n");
		print_cuda_error(cudaError);
		nCUDAErrors++; 
	}
	else {
		printf("No CUDA error.\n");
	}
	
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_face, d_face, face_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Copy of d_face has failed.\n");
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(h_angles, d_angles, angle_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Copy of d_angles has failed.\n");
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(h_ray_classes, d_ray_classes, ray_classes_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Copy of d_ray_classes has failed.\n");
		nCUDAErrors++; 
	}
	cudaError = cudaMemcpy(h_absorption, d_absorption, absorbtion_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		printf("ERROR! Copy of d_absorption has failed.\n");
		nCUDAErrors++; 
	}
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		nCUDAErrors++; 
	}
	
	if(nCUDAErrors!=0) {
		printf("Number of CUDA errors = %d;\n", nCUDAErrors);
	}
	if(DEBUG) printf("--> DEBUG: Memory preparation time: %fms; Compute time: %fms;\n", memory_time, compute_time);
	
	//---------> Freeing allocated resources
	if(d_face!=NULL) cudaFree(d_face);
	if(d_angles!=NULL) cudaFree(d_angles);
	if(d_increments!=NULL) cudaFree(d_increments);
	if(d_ray_classes!=NULL) cudaFree(d_ray_classes);
	if(d_absorption!=NULL) cudaFree(d_absorption);
	if(d_coord_list!=NULL) cudaFree(d_coord_list);
	if(d_rotated_s1!=NULL) cudaFree(d_rotated_s1);
	if(d_xray!=NULL) cudaFree(d_xray);
	if(d_label_list!=NULL) cudaFree(d_label_list);
	
	cudaDeviceSynchronize();
	
	return(0);
}

