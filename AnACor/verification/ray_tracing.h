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


typedef struct
{
    int64_t x, y, z;
} Vector3D;

typedef struct
{
    Vector3D *ray;
    int64_t *posi;
    char *classes;
} Path2;

typedef struct
{
    int64_t *ray;
    int64_t *ray_classes;
    int64_t *posi;
    int64_t *classes;
    int64_t len_path_2;
    int64_t len_classes_posi;
    int64_t len_classes;
} Path2_c;


#endif  // RAY_TRACING_H