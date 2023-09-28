# CC=g++
# CFLAGS=-Wall -Wextra -g -fopenmp -O3 -march=native
# GCC_OPTS =-O3 -fPIC -Wall -Wextra -march=native $(INC)  $(LDFLAGS)
# GCC_OPTS_D =-O0 -fPIC -Wall -Wextra -march=native $(INC)  $(LDFLAGS)
# LDFLAGS=-fopenmp
# pth=./c/
# gpu_pth=./gpu/

# NVCC = ${CUDA_HOME}/bin/nvcc

# NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo
# NVCCFLAGS_D = -O0  -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC  -G -g
# #NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  #V100



# all: sharedlibrary

# sharedlibrary: ray_tracing.o 
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing.so ray_tracing.o 


# ray_tracing.o: $(pth)ray_tracing.c
# 	$(GCC) -c $(pth)ray_tracing.c $(GCC_OPTS) $(LDFLAGS) -o ray_tracing.o

# # program:
# # 	$(CC) -shared -o ray_tracing.so ${pth}*.c -fPIC -std=c99 $(GCC_OPTS) $(LDFLAGS)

# clean:
# 	rm -f program *.o ray_tracing.so



pth := ./
gpu_pth := ./gpu/

INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda
LDFLAGS=-fopenmp
GCC = gcc
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo
NVCCFLAGS_D = -O0  -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC  -G -g
#NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo  #V100
GCC_OPTS =-O3 -fPIC -Wall -Wextra  $(LDFLAGS)  $(INC)
GCC_OPTS_D =-O0 -fPIC -Wall -Wextra $(INC)  $(LDFLAGS)
ANALYZE = FFT.exe

all: clean sharedlibrary_cpu
debug: clean de
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




sharedlibrary_gpu: ray_tracing_c.o  ray_tracing_gpu.o
	$(NVCC) $(NVCCFLAGS) -shared -o ray_tracing_gpu.so  ray_tracing_c.o  ray_tracing_gpu.o

ray_tracing_c.o: ray_tracing_gpu.c 
		g++  -c ray_tracing_gpu.c $(GCC_OPTS) -o  ray_tracing_c.o  

ray_tracing_gpu.o:  $(gpu_pth)ray_tracing_gpu.cu
	$(NVCC) -c  $(gpu_pth)ray_tracing_gpu.cu $(NVCCFLAGS) -o ray_tracing_gpu.o


# bisection_d.o: $(pth)bisection.c
# 	$(GCC) -c $(pth)bisection.c $(GCC_OPTS_D) -o bisection_d.o

# # Modify the 'de' target to include bisection_d.o in the linking process
# de: ray_tracing_d.o ray_tracing_gpu_d.o bisection_d.o testkit_d.o  ray_tracing_basic_d.o 
# 	$(NVCC) $(NVCCFLAGS_D) -shared -o ray_tracing.so ray_tracing_d.o ray_tracing_gpu_d.o bisection_d.o testkit_d.o ray_tracing_basic_d.o


# ray_tracing_gpu_d.o:  $(gpu_pth)ray_tracing_gpu.cu
# 	$(NVCC) -c  $(gpu_pth)ray_tracing_gpu.cu $(NVCCFLAGS_D) -o ray_tracing_gpu_d.o

# ray_tracing_d.o: $(pth)ray_tracing.c
# 	$(GCC) -c $(pth)ray_tracing.c $(GCC_OPTS_D) -o ray_tracing_d.o

# ray_tracing_basic_d.o: $(pth)ray_tracing_basic.c
# 	$(GCC) -c $(pth)ray_tracing_basic.c $(GCC_OPTS) -o ray_tracing_basic_d.o

# bisection_d.o: $(pth)bisection.c
# 	$(GCC) -c $(pth)bisection.c $(GCC_OPTS) -o bisection_d.o

# testkit_d.o: $(pth)testkit.c
# 	$(GCC) -c $(pth)testkit.c $(GCC_OPTS) -o testkit_d.o

# # nnormal

# sharedlibrary: ray_tracing.o ray_tracing_gpu.o bisection.o testkit.o  ray_tracing_basic.o Makefile
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o ray_tracing.so ray_tracing.o ray_tracing_gpu.o bisection.o testkit.o ray_tracing_basic.o


# sharedlibrary_cpu: ray_tracing.o bisection.o testkit.o  ray_tracing_basic.o Makefile
# 	$(CC)  -fPIC -std=c99 $(GCC_OPTS) -shared -o ray_tracing.so ray_tracing.o bisection.o testkit.o  ray_tracing_basic.o



# ray_tracing_gpu.o:  $(gpu_pth)ray_tracing_gpu.cu
# 	$(NVCC) -c  $(gpu_pth)ray_tracing_gpu.cu $(NVCCFLAGS) -o ray_tracing_gpu.o

# ray_tracing.o: $(pth)ray_tracing.c
# 	$(GCC) -c $(pth)ray_tracing.c $(GCC_OPTS) -o ray_tracing.o

# ray_tracing_basic.o: $(pth)ray_tracing_basic.c
# 	$(GCC) -c $(pth)ray_tracing_basic.c $(GCC_OPTS) -o ray_tracing_basic.o

# bisection.o: $(pth)bisection.c
# 	$(GCC) -c $(pth)bisection.c $(GCC_OPTS) -o bisection.o

# testkit.o: $(pth)testkit.c
# 	$(GCC) -c $(pth)testkit.c $(GCC_OPTS) -o testkit.o

clean:	
	rm -f *.o *.~ $(ANALYZE)
