



pth := ./
gpu_pth := ./gpu/

INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda
LDFLAGS=-fopenmp
GCC = gcc
NVCC = ${CUDA_HOME}/bin/nvcc




# Determine the architecture flag based on the GPU model
ifeq ($(ARCH),h100)
  SM_ARCH = sm_90
else ifeq ($(ARCH),a100)
  SM_ARCH = sm_80
else ifeq ($(ARCH),v100)
  SM_ARCH = sm_70
else ifeq ($(ARCH),titanx)
  SM_ARCH = sm_61
else
  # Default architecture
  SM_ARCH = sm_61
endif

NVCCFLAGS = -O3 -arch=$(SM_ARCH) --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo
NVCCFLAGS_D = -O0 -arch=$(SM_ARCH) --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -G -g


#NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  #V100
GCC_OPTS =-O3 -fPIC -Wall -Wextra  $(LDFLAGS)  $(INC)
GCC_OPTS_D =-O0 -fPIC -Wall -Wextra $(INC)  $(LDFLAGS)  -g
ANALYZE = FFT.exe

all: clean sharedlibrary_cpu sharedlibrary_gpu sharedlibrary_interpolation
debug: clean sharedlibrary_cpu_d sharedlibrary_gpu_d sharedlibrary_interpolation_d
cpu: clean sharedlibrary_cpu
gpu: clean sharedlibrary_gpu



sharedlibrary_cpu: ray_tracing.o ray_tracing_basic.o bisection.o testkit.o Makefile
	$(GCC) -shared -o ray_tracing_cpu.so  -fPIC -std=c99 $(GCC_OPTS) $(LDFLAGS) ray_tracing.o ray_tracing_basic.o bisection.o testkit.o 

ray_tracing.o: $(pth)ray_tracing.c
	$(GCC) -c $(pth)ray_tracing.c $(GCC_OPTS) -o ray_tracing.o

ray_tracing_basic.o: $(pth)ray_tracing_basic.c
	$(GCC) -c $(pth)ray_tracing_basic.c $(GCC_OPTS) -o ray_tracing_basic.o

bisection.o: $(pth)bisection.c
	$(GCC) -c $(pth)bisection.c $(GCC_OPTS) -o bisection.o

testkit.o: $(pth)testkit.c
	$(GCC) -c $(pth)testkit.c $(GCC_OPTS) -o testkit.o





sharedlibrary_cpu_d: ray_tracing_d.o ray_tracing_basic_d.o bisection_d.o testkit_d.o 
	$(GCC) -shared -o ray_tracing_cpu.so  -fPIC -std=c99 $(GCC_OPTS_D) $(LDFLAGS) ray_tracing_d.o ray_tracing_basic_d.o bisection_d.o testkit_d.o 

ray_tracing_d.o: $(pth)ray_tracing.c
	$(GCC) -c $(pth)ray_tracing.c $(GCC_OPTS_D) -o ray_tracing_d.o

ray_tracing_basic_d.o: $(pth)ray_tracing_basic.c
	$(GCC) -c $(pth)ray_tracing_basic.c $(GCC_OPTS_D) -o ray_tracing_basic_d.o

bisection_d.o: $(pth)bisection.c
	$(GCC) -c $(pth)bisection.c $(GCC_OPTS_D) -o bisection_d.o

testkit_d.o: $(pth)testkit.c
	$(GCC) -c $(pth)testkit.c $(GCC_OPTS_D) -o testkit_d.o




sharedlibrary_gpu: ray_tracing_c.o  ray_tracing_gpu.o 
	$(NVCC) $(NVCCFLAGS) -shared -o ray_tracing_gpu_${ARCH}.so  ray_tracing_c.o  ray_tracing_gpu.o  

ray_tracing_c.o: ray_tracing_gpu.c 
		g++  -c ray_tracing_gpu.c $(GCC_OPTS) -o  ray_tracing_c.o  

ray_tracing_gpu.o:  $(gpu_pth)ray_tracing_gpu.cu
	$(NVCC) -c  $(gpu_pth)ray_tracing_gpu.cu $(NVCCFLAGS) -o ray_tracing_gpu.o

# ray_tracing_kernels.o: $(gpu_pth)ray_tracing_kernels.cu
# 	$(NVCC) -c  $(gpu_pth)ray_tracing_kernels.cu $(NVCCFLAGS) -o ray_tracing_kernels.o


sharedlibrary_gpu_d: ray_tracing_c_d.o  ray_tracing_gpu_d.o 
	$(NVCC) $(NVCCFLAGS_D) -shared -o ray_tracing_gpu_${ARCH}.so  ray_tracing_c_d.o  ray_tracing_gpu_d.o 

ray_tracing_c_d.o: ray_tracing_gpu.c 
		g++  -c ray_tracing_gpu.c $(GCC_OPTS_D) -o  ray_tracing_c_d.o  

ray_tracing_gpu_d.o:  $(gpu_pth)ray_tracing_gpu.cu
	$(NVCC) -c  $(gpu_pth)ray_tracing_gpu.cu $(NVCCFLAGS_D) -o ray_tracing_gpu_d.o

# ray_tracing_kernels_d.o: $(gpu_pth)ray_tracing_kernels.cu
# 	$(NVCC) -c  $(gpu_pth)ray_tracing_kernels.cu $(NVCCFLAGS_D) -o ray_tracing_kernels_d.o


sharedlibrary_interpolation: gridding_interpolation.c ray_tracing_basic.o
	$(GCC) -shared -o gridding_interpolation.so  -fPIC -std=c99 $(GCC_OPTS) $(LDFLAGS) -lgsl -lgslcblas gridding_interpolation.c ray_tracing_basic.o

sharedlibrary_interpolation_d: gridding_interpolation.c ray_tracing_basic.o
	$(GCC) -shared -o gridding_interpolation.so  -fPIC -std=c99 $(GCC_OPTS_D) $(LDFLAGS) -lgsl -lgslcblas gridding_interpolation.c ray_tracing_basic.o




clean:	
	rm -f *.o *.~ $(ANALYZE)
