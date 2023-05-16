#ifndef RAY_TRACING_H
#define RAY_TRACING_H

#include <stdint.h>

typedef struct
{
    int64_t x, y, z;
} Vector3D;

typedef struct
{
    double li;
    double lo;
    double cr;
    double bu;
} classes_lengths;

#endif  // RAY_H
