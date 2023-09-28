#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define HALF_WARP 16

#ifndef GPU_REDUCTION
#define GPU_REDUCTION

template<typename Type>
__device__ __inline__ Type Reduce_SM(Type *s_data){
	Type l_A = s_data[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			l_A = s_data[threadIdx.x] + s_data[i + threadIdx.x];			
			s_data[threadIdx.x] = l_A;
		}
		__syncthreads();
	}
	
	return(l_A);
}

template<typename Type>
__device__ __inline__ void Reduce_WARP(Type *A){
	Type l_A;
	
	for (int q = HALF_WARP; q > 0; q = q >> 1) {
		l_A = __shfl_down_sync(0xFFFFFFFF, (*A), q);
		__syncwarp();
		(*A) = (*A) + l_A;
	}
}

#endif