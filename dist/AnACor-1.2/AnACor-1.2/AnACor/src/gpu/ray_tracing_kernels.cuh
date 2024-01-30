
#ifndef RAY_TRACING_KERNELS_H
#define RAY_TRACING_KERNELS_H



extern __device__  size_t x_max, y_max, z_max, diagonal, len_coord_list, len_result;
extern __device__  float coeff_cr, coeff_bu, coeff_lo, coeff_li, voxel_length_x, voxel_length_y, voxel_length_z;

__global__ void rt_gpu_python_results(float *d_result_list, float *d_python_result_list, size_t h_len_result);

__inline__ __device__ void transpose_device(float *input, int rows, int cols, float *output);

__inline__ __device__ void dot_product_device(const float *A, const float *B, float *C, int m, int n, int p);

__inline__ __device__ void kp_rotation_device(const float *axis, float theta, float *result);

__global__ void ray_tracing_rotation(const float *d_omega_axis, float *d_omega_list, float *d_kp_rotation_matrix, float *d_raw_xray, float *d_scattering_vector_list, float *d_rotated_xray_list, float *d_rotated_s1_list);

__inline__ __device__ int cube_face(int *ray_origin, float *ray_direction, int L1);

__global__ void rt_gpu_get_face_overall(int *d_face, int *d_coord_list, float *d_rotated_s1_list, float *d_rotated_xray_list);

__global__ void rt_gpu_get_face(int *d_face, int *d_coord_list, float *d_rotated_s1_list, float *d_rotated_xray_list, int batch_number);

__inline__ __device__ void get_theta_phi(float *theta, float *phi, float *ray_direction, int L1);

__global__ void rt_gpu_angles(float *d_angles, float *d_rotated_s1_list, float *d_rotated_xray_list, int nBatches, int batch_number);

__global__ void rt_gpu_angles_overall(float *d_angles, float *d_rotated_s1_list, float *d_rotated_xray_list);

__inline__ __device__ void get_increment_ratio(
	float *increment_ratio_x,
	float *increment_ratio_y,
	float *increment_ratio_z,
	float theta,
	float phi,
	int face);

__global__ void rt_gpu_increments_overall(float *d_increments, float *d_angles);

__global__ void rt_gpu_increments(float *d_increments, float *d_angles);

__inline__ __device__ void get_new_coordinates(
	int *new_x, int *new_y, int *new_z,
	int x, int y, int z,
	float increment_ratio_x, float increment_ratio_y, float increment_ratio_z,
	int increment, float theta, int face);


	__inline__ __device__ void get_distance_2(float *total_length, float s_sum, float increment_ratio_x, float increment_ratio_y, float increment_ratio_z, int face);

__global__ void rt_gpu_absorption(int8_t *d_label_list, int *d_coord_list, int *d_face, float *d_angles, float *d_increments, float *d_result_list, size_t index);

__device__ void determine_boundaries_v2(int *s_ray_classes, int offset, int *boundaries, int *class_values, int *boundary_count);

__device__ void determine_boundaries_nowarp(int *s_ray_classes, int *boundaries, int *class_values, int *boundary_count, int len_ray_classes, size_t lpos);

__device__ void determine_boundaries(int *s_ray_classes, int offset, int *boundaries, int *class_values, int *boundary_count);

__device__ void calculate_distances(int *boundaries, int *class_values, int count, int *distances);


__global__ void rt_gpu_absorption_shuffle_v2(int8_t *d_label_list, int *d_coord_list, int *d_face, float *d_angles, float *d_increments, float *d_result_list, size_t index);

__global__ void rt_gpu_absorption_shuffle(int8_t *d_label_list, int *d_coord_list, int *d_face, float *d_angles, float *d_increments, float *d_result_list, size_t index);

#endif