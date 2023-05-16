#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ray_tracing.h"
#include "testkit.h"
#define M_PI 3.14159265358979323846

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

    if (face == 4)
    {

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
        result.AirLongest = double(y);
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
        result.AirLongest = double(y_max - y);
    }
    else if (face == 6)
    {
        assert(fabs(theta) < M_PI / 2);
        double increment_ratio_x = -1;
        double increment_ratio_y = tan(theta) / cos(phi);
        double increment_ratio_z = tan(phi);
        result.ratio_x = increment_ratio_x;
        result.ratio_y = increment_ratio_y;
        result.ratio_z = increment_ratio_z;
        result.AirLongest = double(x);
    }
    else if (face == 1)
    {
        double increment_ratio_x = 1;
        double increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
        double increment_ratio_z = -tan(phi);
        result.ratio_x = increment_ratio_x;
        result.ratio_y = increment_ratio_y;
        result.ratio_z = increment_ratio_z;
        result.AirLongest = double(x_max - x);
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
        result.AirLongest = double(z);
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
        result.AirLongest = double(z_max - z);
    }
    else
    {
        fprintf(stderr, "Error: Unexpected ray out face.\n");
    }
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

    bisection_result.x = new_x;
    bisection_result.y = new_y;
    bisection_result.z = new_z;
    bisection_result.MiddlePoint = Middle;

    return bisection_result;
}

typedef struct
{
    int64_t *path;
    int64_t *classes;
} Path_iterative_bisection;

void coord_append(int64_t *path_2, int64_t path_2_size, int64_t z, int64_t y, int64_t x)
{
    path_2[path_2_size * 3] = z;
    path_2[path_2_size * 3 + 1] = y;
    path_2[path_2_size * 3 + 2] = x;
}

Path_iterative_bisection iterative_bisection(double theta, double phi,
                                             int64_t *coord, int face, int64_t ***label_list, int64_t *shape, double resolution)
{
    Path_iterative_bisection result;
    int64_t counter = 0;
    double AirShortest = 0;


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
    int boundary_air_outermost = 0;
    int air_cls = 0;
    bisection_result Air_outermost_result = bisection(AirLongest, AirShortest,
                                                      resolution, label_list, shape,
                                                      increment_ratios, coord,
                                                      boundary_air_outermost, air_cls);
    path_2_size++;
    coord_append(path_2, path_2_size,
                 Air_outermost_result.z, Air_outermost_result.y, Air_outermost_result.x);
    classes[path_2_size] = 0;
    boundary_list[path_2_size] = boundary_air_outermost;
    printArray(path_2, path_2_size*3);
    printArray(classes, path_2_size);
    printArray(boundary_list, path_2_size);

    // double CrystalLongest = AirMiddle_outer;
    // double CrystalShortest = 0.0;
    // Coordinate cr_outer_potential_coord = bisection(&counter, CrystalLongest, CrystalShortest, resolution, label_list, increment_ratios, coord, "outer", 3);
    // path_2 = realloc(path_2, (path_2_size + 1) * sizeof(Coordinate));
    // path_2[path_2_size] = cr_outer_potential_coord;
    // classes = realloc(classes, (path_2_size + 1) * sizeof(char *));
    // classes[path_2_size] = "cr_outer";
    // path_2_size++;

    // double LoopLongest = AirMiddle_outer;
    // double LoopShortest = CrystalMiddle;
    // Coordinate potential_coord = bisection(&counter, LoopLongest, LoopShortest, resolution, label_list, increment_ratios, coord, "inner", 2);

    // if (fabs(potential_coord.x - air_outermost_potential_coord.x) + fabs(potential_coord.y - air_outermost_potential_coord.y) + fabs(potential_coord.z - air_outermost_potential_coord.z) < 1.0)
    // {
    //     // pass
    // }
    // else
    // {
    //     path_2 = realloc(path_2, (path_2_size + 1) * sizeof(Coordinate));
    //     path_2[path_2_size] = potential_coord;
    //     classes = realloc(classes, (path_2_size + 1) * sizeof(char *));
    //     classes[path_2_size] = "lo_inner";
    //     path_2_size++;

    //     LoopLongest = AirMiddle_outer;
    //     LoopShortest = LoopMiddle;
    //     potential_coord = bisection(&counter, LoopLongest, LoopShortest, resolution, label_list, increment_ratios, coord, "outer", 2);
    //     path_2 = realloc(path_2, (path_2_size + 1) * sizeof(Coordinate));
    //     path_2[path_2_size] = potential_coord;
    //     classes = realloc(classes, (path_2_size + 1) * sizeof(char *));
    //     classes[path_2_size] = "lo_outer";
    //     path_2_size++;
    // }
    // double BubbleLongest = AirMiddle_outer;
    // double BubbleShortest = CrystalMiddle;
    // Coordinate potential_coord = bisection(&counter, BubbleLongest, BubbleShortest, resolution, label_list, increment_ratios, coord, "inner", 4);
    // if (fabs(potential_coord.x - air_outermost_potential_coord.x) + fabs(potential_coord.y - air_outermost_potential_coord.y) + fabs(potential_coord.z - air_outermost_potential_coord.z) < 1.0)
    // {
    //     // pass
    // }
    // else
    // {
    //     path_2 = realloc(path_2, (path_2_size + 1) * sizeof(Coordinate));
    //     path_2[path_2_size] = potential_coord;
    //     classes = realloc(classes, (path_2_size + 1) * sizeof(char *));
    //     classes[path_2_size] = "bu_inner";
    //     path_2_size++;

    //     BubbleLongest = AirMiddle_outer;
    //     BubbleShortest = BubbleMiddle;
    //     potential_coord = bisection(&counter, BubbleLongest, BubbleShortest, resolution, label_list, increment_ratios, coord, "outer", 4);
    //     path_2 = realloc(path_2, (path_2_size + 1) * sizeof(Coordinate));
    //     path_2[path_2_size] = potential_coord;
    //     classes = realloc(classes, (path_2_size + 1) * sizeof(char *));
    //     classes[path_2_size] = "bu_outer";
    //     path_2_size++;
    // }

    // PathClasses result;
    // result.path = path_2;
    // result.classes = classes;
    // result.size = path_2_size;

    return result;
}