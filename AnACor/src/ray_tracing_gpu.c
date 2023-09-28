// #define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/types.h>
#define M_PI 3.14159265358979323846
#define test_mod 0


int ray_tracing_gpu_overall_kernel(size_t low, size_t up,
								   int *coord_list,
								   size_t h_len_coord_list,
								   const float *scattering_vector_list, const float *omega_list,
								   const float *raw_xray,
								   const float *omega_axis, const float *kp_rotation_matrix,
								   size_t h_len_result,
								   float *voxel_size, float *coefficients,
								   int8_t *label_list_1d, int *shape, int full_iteration,
								   int store_paths, float *h_result_list, int *h_face, float *h_angles, float *h_python_overall_result_list,int gpumethod);

#ifdef __cplusplus
extern "C"
{
#endif



float *ray_tracing_gpu_overall(size_t low, size_t up,
                                    int *coord_list,
                                    size_t len_coord_list,
                                    const float *scattering_vector_list, const float *omega_list,
                                    const float *raw_xray,
                                    const float *omega_axis, const float *kp_rotation_matrix,
                                    size_t len_result,
                                    float *voxel_size, float *coefficients, int8_t ***label_list,
                                    int8_t *label_list_1d, int *shape, int full_iteration,
                                    int store_paths,int gpumethod)
    {

        // float *result_list = malloc( len_result* sizeof(float));
        // size_t len_result_float = (int32_t) len_result* sizeof(float);
        // int32_t len_result_float = (int32_t) len_result;
        printf("low is %d \n", low);
        printf("up is %d \n", up);
        float factor = 1;
        // len_result = (int)(len_result*factor);
        printf("len_result is %d \n", len_result);
        float *h_result_list = (float *)malloc(len_result * len_coord_list * 2 * sizeof(float));
        float *h_python_result_list = (float *)malloc(len_result * sizeof(float));
        int *h_face = (int *)malloc(len_coord_list * 2 * sizeof(int));
        float *h_angles = (float *)malloc(4 * sizeof(float));

        ray_tracing_gpu_overall_kernel(low, up, coord_list, len_coord_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, len_result, voxel_size, coefficients, label_list_1d, shape, full_iteration, store_paths, h_result_list, h_face, h_angles, h_python_result_list,gpumethod);
        
        free(h_result_list);
        return h_python_result_list;
    }
#ifdef __cplusplus
}
#endif