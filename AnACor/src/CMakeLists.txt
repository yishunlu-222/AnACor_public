
set(${CMAKE_PROJECT_NAME}_CUDA_SRC

    src/gpu/ray_tracing_gpu.cu
    PARENT_SCOPE
)

set(${CMAKE_PROJECT_NAME}_SRC
    src/ray_tracing.c
    src/bisection.c
    src/ray_tracing_basic.c
    src/testkit.c
    PARENT_SCOPE
)