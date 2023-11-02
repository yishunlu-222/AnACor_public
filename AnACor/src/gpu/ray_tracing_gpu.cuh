#ifndef RAY_TRACING_GPU
#define RAY_TRACING_GPU

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
#include "ray_tracing_kernels.cuh"
#define DEBUG 0
#include "timer.h"
#define warpSize 32
#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)

// extern  __device__ __constant__ size_t x_max, y_max, z_max, diagonal, len_coord_list, len_result;
// extern  __device__ __constant__ float coeff_cr, coeff_bu, coeff_lo, coeff_li, voxel_length_x, voxel_length_y, voxel_length_z;




#endif // RAY_TRACING_GPU