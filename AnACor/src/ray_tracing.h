#ifndef RAY_TRACING_H
#define RAY_TRACING_H
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#define M_PI 3.14159265358979323846
#include <stdint.h>


//gpu kernel
// int ray_tracing_gpu_overall_kernel(size_t low, size_t up,
// 								   int *coord_list,
// 								   size_t h_len_coord_list,
// 								   const float *scattering_vector_list, const float *omega_list,
// 								   const float *raw_xray,
// 								   const float *omega_axis, const float *kp_rotation_matrix,
// 								   size_t h_len_result,
// 								   float *voxel_size, float *coefficients,
// 								   int8_t *label_list_1d, int *shape, int full_iteration,
// 								   int store_paths, float *h_result_list, int *h_face, float *h_angles, float *h_python_overall_result_list,int gpumethod);



typedef struct
{
    int64_t x, y, z;
} Vector3D;

typedef struct
{
    double li;
    double lo;
    double cr;
    double bu;
} classes_lengths;

typedef struct
{
    int64_t *ray;
    int64_t *posi;
    int64_t *classes;
    int64_t len_path_2;
    int64_t len_classes_posi;
    int64_t len_classes;
} Path2_c;

typedef struct
{
    double theta;
    double phi;
} ThetaPhi;

typedef struct
{
    char key[100];
    char value[100];
} DictionaryEntry;


ThetaPhi dials_2_thetaphi_22(double rotated_s1[3], int64_t L1);
void dials_2_numpy(double vector[3], double result[3]);
int64_t cube_face(int64_t ray_origin[3], double ray_direction[3], int64_t cube_size[3], int L1);
int64_t which_face(int64_t coord[3], int64_t shape[3], double theta, double phi);
void appending(int64_t increment, int64_t *path_2,
               int64_t *classes, int64_t *classes_posi,
               int64_t *potential_coord,
               int64_t label, int64_t previous_label,
               int64_t *len_classes, int64_t *len_classes_posi,
               int64_t *len_path_2);
Path2_c cal_coord(double theta, double phi, int64_t *coord, int64_t face,
                  int64_t *shape, int8_t ***label_list, int64_t full_iteration);
int64_t *ray_tracing(int64_t *coord, int64_t *shape, int8_t ***label_list,
                     double *scattering_vector_list, double *omega_list,
                     double *raw_xray,
                     double *omega_axis, double *kp_rotation_matrix,
                     int64_t *voxel_size, double *coefficients,
                     int64_t full_iteration, int64_t store_paths, int64_t *len_result_list,
                     int64_t *face, double *angles, int64_t *python_overall_result_list);

double *cal_path2_plus(Path2_c path_2_cal_result, double *voxel_size);
double cal_rate(double *numbers_1, double *numbers_2, double *coefficients,
                char Isexp);


#endif  // RAY_H
