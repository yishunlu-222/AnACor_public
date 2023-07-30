INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_86 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  # 3090
NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  # V100
NVCCFLAGS = -O3 -arch=sm_80 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  # A100
#NVCCFLAGS = -O3 -arch=sm_90 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  # H100

GCC_OPTS =-O3 -fPIC -Wall -Wextra $(INC)

ANALYZE = FFT.exe

all: clean sharedlibrary

sharedlibrary: ray_tracing_f.o ray_tracing_gpu_f.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing_f.so ray_tracing_f.o ray_tracing_gpu_f.o

ray_tracing_gpu_f.o: ray_tracing_gpu_f.cu
	$(NVCC) -c ray_tracing_gpu_f.cu $(NVCCFLAGS)

ray_tracing_f.o: ray_tracing_f.c
	$(GCC) -c ray_tracing_f.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


