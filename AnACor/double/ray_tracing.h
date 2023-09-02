#ifndef RAY_TRACING_F_H
#define RAY_TRACING_F_H
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
    int x, y, z;
} Vector3D;

typedef struct
{
    Vector3D *ray;
    int *posi;
    char *classes;
} Path2;

typedef struct
{
    int *ray;
    int *ray_classes;
    int *posi;
    int *classes;
    int len_path_2;
    int len_classes_posi;
    int len_classes;
} Path2_c;


#endif  // RAY_TRACING_H