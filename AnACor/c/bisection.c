#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
// #include "ray_tracing.h"
#include "testkit.h"
#include <assert.h>
#include "bisection.h"
#define M_PI 3.14159265358979323846
#define test_mode 0
typedef struct
{
    double ratio_x, ratio_y, ratio_z;
    double AirLongest;
} Increment3D;

Increment3D increments(int face, double theta, double phi,
                       int64_t z_max, int64_t y_max, int64_t x_max,
                       int64_t z, int64_t y, int64_t x)
{
    /*  'FRONTZY' = 1;
    *   'LEYX' = 2 ;
    *   'RIYX' = 3;
        'TOPZX' = 4;
        'BOTZX' = 5;
        "BACKZY" = 6 ;

    */
    Increment3D result;
    // printf("face: %d\n", face);
    // printf("theta: %lf\n", theta);
    // printf("phi: %lf\n", phi);
    // printf("z_max: %d y_max: %d x_max: %d\n", z_max, y_max, x_max);
    // printf("z: %d y: %d x: %d\n", z, y, x);
    if (face == 4)
    {
        assert(theta > 0);
        if (fabs(theta) < M_PI / 2)
        {
            double increment_ratio_x = -cos(fabs(phi)) / (tan(fabs(theta)));
            double increment_ratio_y = 1;
            double increment_ratio_z = sin(phi) / (tan(fabs(theta)));
            result.ratio_x = increment_ratio_x;
            result.ratio_y = increment_ratio_y;
            result.ratio_z = increment_ratio_z;
        }
        else
        {
            double increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
            double increment_ratio_y = 1;
            double increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
            result.ratio_x = increment_ratio_x;
            result.ratio_y = increment_ratio_y;
            result.ratio_z = increment_ratio_z;
        }
        result.AirLongest = (double)y;
    }
    else if (face == 5)
    {

        if (fabs(theta) < M_PI / 2)
        {
            double increment_ratio_x = -cos(fabs(phi)) / (tan(fabs(theta)));
            double increment_ratio_y = -1;
            double increment_ratio_z = sin(phi) / (tan(fabs(theta)));
            result.ratio_x = increment_ratio_x;
            result.ratio_y = increment_ratio_y;
            result.ratio_z = increment_ratio_z;
        }
        else
        {
            double increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
            double increment_ratio_y = -1;
            double increment_ratio_z = -sin(phi) / (tan(M_PI - fabs(theta)));
            result.ratio_x = increment_ratio_x;
            result.ratio_y = increment_ratio_y;
            result.ratio_z = increment_ratio_z;
        }
        result.AirLongest = (double)(y_max - y);
    }
    else if (face == 6)
    {
        // assert(fabs(theta) < M_PI / 2);
        double increment_ratio_x = -1;
        double increment_ratio_y = tan(theta) / cos(phi);
        double increment_ratio_z = tan(phi);
        result.ratio_x = increment_ratio_x;
        result.ratio_y = increment_ratio_y;
        result.ratio_z = increment_ratio_z;
        result.AirLongest = (double)(x);
    }
    else if (face == 1)
    {
        double increment_ratio_x = 1;
        double increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
        double increment_ratio_z = -tan(phi);
        result.ratio_x = increment_ratio_x;
        result.ratio_y = increment_ratio_y;
        result.ratio_z = increment_ratio_z;
        result.AirLongest = (double)(x_max - x);
    }
    else if (face == 2)
    {
        if (fabs(theta) < M_PI / 2)
        {
            result.ratio_x = -1 / (tan(fabs(phi)));
            result.ratio_y = tan(theta) / sin(fabs(phi));
            result.ratio_z = -1;
        }
        else
        {
            result.ratio_x = 1 / (tan(fabs(phi)));
            result.ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            result.ratio_z = -1;
        }
        result.AirLongest = (double)(z);
    }
    else if (face == 3)
    {
        if (fabs(theta) < M_PI / 2)
        {
            result.ratio_x = -1 / (tan(fabs(phi)));
            result.ratio_y = tan(theta) / sin(fabs(phi));
            result.ratio_z = 1;
        }
        else
        {
            result.ratio_x = 1 / (tan(fabs(phi)));
            result.ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            result.ratio_z = 1;
        }
        result.AirLongest = (double)(z_max - z);
    }
    else
    {
        fprintf(stderr, "Error: Unexpected ray out face.\n");
    }
    return result;
}

typedef struct
{
    int64_t x, y, z;
    double MiddlePoint;
} bisection_result;

bisection_result bisection(double Longest, double Shortest,
                           double resolution, int8_t ***label_list, int64_t *shape,
                           Increment3D increment_ratios, int64_t *coord,
                           int8_t boundary, int8_t cls)
{
    // write the doc of the variables here
    // Longest is the longest distance between the current class and the boundary
    // Shortest is the shortest distance between the current class and the boundary
    // resolution is the maximum resolution to stop the  iteration
    // label_list is the list of labels
    // shape is the shape of the label_list
    // increment_ratios is the ratio of the increment of the x, y, z axis
    // coord is the coordinate of the current voxel
    // boundary is the label of the boundary, 0: inside boudary, 1: outside boundary
    bisection_result result;
    double Difference = Longest - Shortest;

    double increment_ratio_z = increment_ratios.ratio_z;
    double increment_ratio_y = increment_ratios.ratio_y;
    double increment_ratio_x = increment_ratios.ratio_x;
    int64_t z = coord[0];
    int64_t y = coord[1];
    int64_t x = coord[2];
    double Middle = (Longest + Shortest) / 2.0;

    int64_t z_max = shape[0];
    int64_t y_max = shape[1];
    int64_t x_max = shape[2];

    x_max -= 1;
    y_max -= 1;
    z_max -= 1;

    while (Difference > resolution)
    {
        // (*counter)++;
        Middle = (Longest + Shortest) / 2.0;

        int64_t new_x = (int64_t)(floor(x + Middle * increment_ratio_x));
        int64_t new_y = (int64_t)(floor(y - Middle * increment_ratio_y));
        int64_t new_z = (int64_t)(floor(z + Middle * increment_ratio_z));

        int8_t label = label_list[new_z][new_y][new_x];

        if (boundary == 0)
        {
            if (label == cls)
            {
                Longest = Middle;
            }
            else
            {
                Shortest = Middle;
            }
        }
        else
        {
            if (label == cls)
            {
                Shortest = Middle;
            }
            else
            {
                Longest = Middle;
            }
        }
        Difference = Longest - Shortest;
    }

    int64_t new_x = (int64_t)(floor(x + floor(Middle) * increment_ratio_x));
    int64_t new_y = (int64_t)(floor(y - floor(Middle) * increment_ratio_y));
    int64_t new_z = (int64_t)(floor(z + floor(Middle) * increment_ratio_z));

    result.x = new_x;
    result.y = new_y;
    result.z = new_z;
    result.MiddlePoint = Middle;

    return result;
}

void coord_append(int64_t *path_2, int64_t path_2_size, int64_t z, int64_t y, int64_t x)
{
    path_2[path_2_size * 3] = z;
    path_2[path_2_size * 3 + 1] = y;
    path_2[path_2_size * 3 + 2] = x;
}

double difference_length(int64_t *start, int64_t *end, double *voxel_size)
{
    double voxel_length_z = voxel_size[0];
    double voxel_length_y = voxel_size[1];
    double voxel_length_x = voxel_size[2];
    double result = sqrt(pow((start[0] - end[0] ) * voxel_length_z, 2) +
                         pow((start[1] - end[1] ) * voxel_length_y, 2) +
                         pow((start[2] - end[2] ) * voxel_length_x, 2));
    return result;
}

double *cal_path_bisection(Path_iterative_bisection Path, double *voxel_size)
{
    double *result = malloc(4 * sizeof(double));
    double voxel_length_z = voxel_size[0];
    double voxel_length_y = voxel_size[1];
    double voxel_length_x = voxel_size[2];
    int64_t *ray = Path.path;
    int8_t *clses = Path.classes;
    int8_t *bl = Path.boundary_list;
    int64_t l = Path.length;

    int64_t cr_inner[3] = {ray[0], ray[1], ray[2]};
    int64_t air_inner[3] = {ray[1 * 3 + 0], ray[1 * 3 + 1], ray[1 * 3 + 2]};
    int64_t cr_outer[3] = {ray[2 * 3 + 0], ray[2 * 3 + 1], ray[2 * 3 + 2]};

    double total_LineLength = difference_length(cr_inner, air_inner, voxel_size);
    double cr_l = difference_length(cr_inner, cr_outer, voxel_size);
    double li_l = 0;
    double bu_l = 0;
    double lo_l = 0;
    double air_l = 0;
    if (l == 5)
    {

        int64_t lo_outer[3] = {ray[4 * 3 + 0], ray[4 * 3 + 1], ray[4 * 3 + 2]};
        int64_t lo_inner[3] = {ray[3 * 3 + 0], ray[3 * 3 + 1], ray[3 * 3 + 2]};
        lo_l = difference_length(lo_inner, lo_outer, voxel_size);
    }
    else if (l == 7)
    {
        int64_t bu_inner[3] = {ray[5 * 3 + 0], ray[5 * 3 + 1], ray[5 * 3 + 2]};
        int64_t bu_outer[3] = {ray[6 * 3 + 0], ray[6 * 3 + 1], ray[6 * 3 + 2]};
        bu_l = difference_length(bu_inner, bu_outer, voxel_size);
    }

    li_l = total_LineLength - lo_l - bu_l - cr_l - air_l;
    result[2] = cr_l;
    result[0] = li_l;
    result[1] = lo_l;
    result[3] = bu_l;
    return result;
}

Path_iterative_bisection iterative_bisection(double theta, double phi,
                                             int64_t *coord, int face, int8_t ***label_list, int64_t *shape, double resolution, int num_cls)
{
    Path_iterative_bisection result;
    int64_t counter = 0;
    double AirShortest = 0;
    int cr_cls = 3, air_cls = 0, lo_cls = 2, bu_cls = 4;
    int64_t z = coord[0], y = coord[1], x = coord[2];
    int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int64_t diagonal = x_max * sqrt(3);
    int64_t *path_2 = malloc(diagonal * 3 * sizeof(int64_t));
    int64_t path_2_size = 0;
    int8_t *classes = malloc(diagonal * sizeof(int8_t));
    int8_t *boundary_list = malloc(diagonal * sizeof(int8_t));
    classes[path_2_size] = 3;
    boundary_list[path_2_size] = 0;
    coord_append(path_2, path_2_size, z, y, x);
    x_max -= 1;
    y_max -= 1;
    z_max -= 1;

    Increment3D increment_ratios = increments(face, theta, phi, z_max, y_max, x_max, z, y, x);

    // first of all, detereminate the boundary between the air and the whole sample

    double AirLongest = increment_ratios.AirLongest;

    bisection_result Air_outermost_result = bisection(AirLongest, AirShortest,
                                                      resolution, label_list, shape,
                                                      increment_ratios, coord,
                                                      0, air_cls);
    path_2_size++;

    coord_append(path_2, path_2_size,
                 Air_outermost_result.z, Air_outermost_result.y, Air_outermost_result.x);
    classes[path_2_size] = air_cls;
    boundary_list[path_2_size] = 0;

    // then,  finding the boundary between outer boudary of the crystal

    double CrystalLongest = Air_outermost_result.MiddlePoint;
    double CrystalShortest = 0;
    int boundary_cr_outermost = 1;

    bisection_result Crystal_outer_result = bisection(CrystalLongest, CrystalShortest,
                                                      resolution, label_list, shape,
                                                      increment_ratios, coord,
                                                      1, cr_cls);
    path_2_size++;
    classes[path_2_size] = cr_cls;
    boundary_list[path_2_size] = 1;
    coord_append(path_2, path_2_size,
                 Crystal_outer_result.z, Crystal_outer_result.y, Crystal_outer_result.x);

    // then, finding the inner boundaries (flag=0) and outer boundaries (flag=1) of the loop
    while (1)
    {
        double LoopLongest = Air_outermost_result.MiddlePoint;
        double LoopShortest = Crystal_outer_result.MiddlePoint;
        bisection_result Loop_inner_result = bisection(LoopLongest, LoopShortest,
                                                       resolution, label_list, shape,
                                                       increment_ratios, coord,
                                                       0, lo_cls);
        // printf("Loop_inner_result: %f, %f, %f\n", Loop_inner_result.z, Loop_inner_result.y, Loop_inner_result.x);
        if ((fabs(Loop_inner_result.z - Air_outermost_result.z) +
             fabs(Loop_inner_result.y - Air_outermost_result.y) +
             fabs(Loop_inner_result.x - Air_outermost_result.x)) /
                3 >
            1)
        {

            LoopLongest = Air_outermost_result.MiddlePoint;
            LoopShortest = Loop_inner_result.MiddlePoint;
            bisection_result Loop_outer_result = bisection(LoopLongest, LoopShortest,
                                                           resolution, label_list, shape,
                                                           increment_ratios, coord,
                                                           1, lo_cls);

            // if the Loop_inner_result is very close to the Loop_outer_result, then we consider this is the artefact of the segmentation.
            if ((fabs(Loop_inner_result.z - Loop_outer_result.z) +
                 fabs(Loop_inner_result.y - Loop_outer_result.y) +
                 fabs(Loop_inner_result.x - Loop_outer_result.x)) /
                    3 <=
                2)
            {
                break;
            }
            else
            {

                path_2_size++;
                classes[path_2_size] = lo_cls;
                boundary_list[path_2_size] = 0;
                coord_append(path_2, path_2_size,
                             Loop_inner_result.z, Loop_inner_result.y, Loop_inner_result.x);
                path_2_size++;
                classes[path_2_size] = lo_cls;
                boundary_list[path_2_size] = 1;
                coord_append(path_2, path_2_size,
                             Loop_outer_result.z, Loop_outer_result.y, Loop_outer_result.x);
            }
        }
        // then, finding the inner boundaries (flag=0) and outer boundaries (flag=1) of the bubble if there is one (num_cls ==4)

        if (num_cls == 3)
        { // this is the most frequent case
            result.path = path_2;
            result.classes = classes;
            result.length = path_2_size;
            result.boundary_list = boundary_list;
            return result;
        }

        else
        {
            double BubbleLongest = Air_outermost_result.MiddlePoint;
            double BubbleShortest = Crystal_outer_result.MiddlePoint;
            bisection_result Bubble_inner_result = bisection(BubbleLongest, BubbleShortest,
                                                             resolution, label_list, shape,
                                                             increment_ratios, coord,
                                                             0, bu_cls);

            if (fabs(Bubble_inner_result.z - Air_outermost_result.z) +
                    fabs(Bubble_inner_result.y - Air_outermost_result.y) +
                    fabs(Bubble_inner_result.x - Air_outermost_result.x) >
                1)
            {

                BubbleLongest = Air_outermost_result.MiddlePoint;
                BubbleShortest = Bubble_inner_result.MiddlePoint;
                bisection_result Bubble_outer_result = bisection(BubbleLongest, BubbleShortest,
                                                                 resolution, label_list, shape,
                                                                 increment_ratios, coord,
                                                                 1, bu_cls);

                if ((fabs(Bubble_inner_result.z - Bubble_outer_result.z) +
                     fabs(Bubble_inner_result.y - Bubble_outer_result.y) +
                     fabs(Bubble_inner_result.x - Bubble_outer_result.x)) /
                        3 <=
                    2)
                {
                    break;
                }

                path_2_size++;
                classes[path_2_size] = bu_cls;
                boundary_list[path_2_size] = 0;
                coord_append(path_2, path_2_size,
                             Bubble_inner_result.z, Bubble_inner_result.y, Bubble_inner_result.x);
                path_2_size++;
                classes[path_2_size] = bu_cls;
                boundary_list[path_2_size] = 1;
                coord_append(path_2, path_2_size,
                             Bubble_outer_result.z, Bubble_outer_result.y, Bubble_outer_result.x);
            }
        }

        break;
    }
    /***
     maybe need to consider the intermediate boundaries between the bubble and the loop, like air in the future
     ***/

    if (test_mode)
    {
        printArray(path_2, (path_2_size + 1) * 3);
        printArrayshort(classes, path_2_size + 1);
        printArrayshort(boundary_list, path_2_size + 1);
    }
    path_2_size++;

    result.path = realloc(path_2, path_2_size * 3 * sizeof(int64_t));
    result.classes = realloc(classes, path_2_size * sizeof(int8_t));
    result.boundary_list = realloc(boundary_list, path_2_size * sizeof(int8_t));
    result.length = path_2_size;
    return result;
}