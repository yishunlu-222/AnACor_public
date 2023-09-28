// bisection.h
#ifndef BISECTION_H
#define BISECTION_H


typedef struct
{
    int64_t *path;
    int8_t *classes;
    int64_t *boundary_list;
    int64_t length;
} Path_iterative_bisection;

Path_iterative_bisection iterative_bisection(double theta, double phi,
                                             int64_t *coord, int face, 
                                             int8_t ***label_list, int64_t *shape, double resolution,int num_cls);

double *cal_path_bisection(Path_iterative_bisection Path, double *voxel_size);




#endif
