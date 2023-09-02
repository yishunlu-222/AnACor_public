INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo
NVCCFLAGS_D = -O0  -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC  -G -g
#NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  #V100
NVCCFLAGS = -O3 -arch=sm_90 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  #V100
GCC_OPTS =-O3 -fPIC -Wall -Wextra $(INC)
GCC_OPTS_D =-O0 -fPIC -Wall -Wextra $(INC)
ANALYZE = FFT.exe

all: clean sharedlibrary sharedlibrary_f
debug: clean de



de: ray_tracing_d.o ray_tracing_gpu_d.o
	$(NVCC) $(NVCCFLAGS_D) -shared -o ray_tracing.so ray_tracing_d.o ray_tracing_gpu_d.o

ray_tracing_gpu_d.o: ray_tracing_gpu.cu
	$(NVCC) -c ray_tracing_gpu.cu $(NVCCFLAGS_D) -o ray_tracing_gpu_d.o

ray_tracing_d.o: ray_tracing.c
	$(GCC) -c ray_tracing.c $(GCC_OPTS_D) -o ray_tracing_d.o


sharedlibrary_f: ray_tracing_f.o ray_tracing_gpu_f.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing_f.so ray_tracing_f.o ray_tracing_gpu_f.o

ray_tracing_gpu_f.o: ray_tracing_gpu_f.cu
	$(NVCC) -c ray_tracing_gpu_f.cu $(NVCCFLAGS)

ray_tracing_f.o: ray_tracing_f.c
	$(GCC) -c ray_tracing_f.c $(GCC_OPTS)


sharedlibrary: ray_tracing.o ray_tracing_gpu.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing.so ray_tracing.o ray_tracing_gpu.o

ray_tracing_gpu.o: ray_tracing_gpu.cu
	$(NVCC) -c ray_tracing_gpu.cu $(NVCCFLAGS)

ray_tracing.o: ray_tracing.c
	$(GCC) -c ray_tracing.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


