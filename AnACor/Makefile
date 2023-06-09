INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_86 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo

GCC_OPTS =-O3 -fPIC -Wall -Wextra $(INC)

ANALYZE = FFT.exe

all: clean sharedlibrary

sharedlibrary: ray_tracing.o ray_tracing_gpu.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing.so ray_tracing.o ray_tracing_gpu.o

ray_tracing_gpu.o: ray_tracing_gpu.cu
	$(NVCC) -c ray_tracing_gpu.cu $(NVCCFLAGS)

ray_tracing.o: ray_tracing.c
	$(GCC) -c ray_tracing.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


