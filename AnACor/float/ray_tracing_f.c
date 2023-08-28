// #define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "unit_test.h"
#include "ray_tracing_f.h"
#include <sys/resource.h>
#include "matrices.h"
#define M_PI static_cast<float>(3.14159265358979323846)
#define test_mod 0
#define DEBUG 0
#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)

// TODO:
// Use 1d data in ray calculation
//

int compare_Path2s(Path2_c *path, Path2_c *path_ref)
{
    int total_errors = 0;
    if (path->len_path_2 != path_ref->len_path_2)
    {
        printf("--> Comparing Path2_c: Wrong len_path_2 C:%d; R:%d;\n", path->len_path_2, path_ref->len_path_2);
        total_errors++;
    }
    if (path->len_classes_posi != path_ref->len_classes_posi)
    {
        printf("--> Comparing Path2_c: Wrong len_classes_posi C:%d; R:%d;\n", path->len_classes_posi, path_ref->len_classes_posi);
        total_errors++;
    }
    if (path->len_classes != path_ref->len_classes)
    {
        printf("--> Comparing Path2_c: Wrong len_classes C:%d; R:%d;\n", path->len_classes, path_ref->len_classes);
        total_errors++;
    }

    // Comparing ray coordinates
    int ray_errors = 0;
    for (int f = 0; f < (path->len_path_2) * 3; f++)
    {
        if (path->ray[f] != path_ref->ray[f])
            ray_errors++;
    }
    if (ray_errors > 0)
    {
        printf("--> Comparing Path2_c: path->ray do not agree!\n");
        total_errors++;
    }

    // Comparing ray classes
    int ray_classes_errors = 0;
    for (int f = 0; f < (path->len_path_2); f++)
    {
        if (path->ray_classes[f] != path_ref->ray_classes[f])
            ray_classes_errors++;
        // printf("%d, ", path->ray_classes[f]);
        // if(f==(path->len_path_2)-1) printf("\n=\n");
    }
    if (ray_classes_errors > 0)
    {
        printf("--> Comparing Path2_c: path->ray_classes do not agree!\n");
        total_errors++;
    }

    // Comparing position of the borders
    int posi_errors = 0;
    for (int f = 0; f < (path->len_classes_posi); f++)
    {
        if (path->posi[f] != path_ref->posi[f])
            posi_errors++;
    }
    if (posi_errors > 0)
    {
        printf("--> Comparing Path2_c: path->posi do not agree!\n");
        total_errors++;
    }

    // Comparing border labels
    int classes_errors = 0;
    for (int f = 0; f < (path->len_classes); f++)
    {
        if (path->classes[f] != path_ref->classes[f])
            classes_errors++;
    }
    if (classes_errors > 0)
    {
        printf("--> Comparing Path2_c: path->classes do not agree!\n");
        total_errors++;
    }

    return (total_errors);
}

int compare_classes_lengths(float *lengths, float *lengths_ref)
{
    float max_difference = 1.0e-6;
    float total_difference = 0;
    total_difference += abs(lengths[0] - lengths_ref[0]);
    total_difference += abs(lengths[1] - lengths_ref[1]);
    total_difference += abs(lengths[2] - lengths_ref[2]);
    total_difference += abs(lengths[3] - lengths_ref[3]);
    if (total_difference > max_difference)
    {
        printf("--> Comparing classes lenghts: Error!\n");
        return 1;
    }
    else
        return 0;
}

int compare_voxels(int8_t ***label_list, int8_t *label_list_1d, int *shape)
{
    int z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int nErrors = 0;
    for (int z = 0; z < z_max; z++)
    {
        for (int y = 0; y < y_max; y++)
        {
            for (int x = 0; x < x_max; x++)
            {
                int pos = INDEX_3D(z_max, y_max, x_max, z, y, x);
                if (label_list_1d[pos] != label_list[z][y][x])
                {
                    printf("--> Comparing label lists: Error! %d!=%d;\n", (int)label_list_1d[pos], (int)label_list[z][y][x]);
                    nErrors++;
                }
            }
        }
    }
    return nErrors;
}

typedef struct
{
    float theta;
    float phi;
} ThetaPhi;

typedef struct
{
    float li;
    float lo;
    float cr;
    float bu;
} classes_lengths;

// typedef struct {
//     Point *path_ray;
//     int *posi;
//     int *classes;
// } Path2;

int count_len(int *arr)
{
    int count = 0;
    while (*arr != '\0')
    {
        count++;
        arr++;
    }
    printf("Length of array: %d\n", count);

    return count;
}

void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayshort(int arr[], char n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayD(float arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%.15lf ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

ThetaPhi dials_2_thetaphi_22(float rotated_s1[3], int L1)
{
    ThetaPhi result;
    if (L1 == 1)
    {
        rotated_s1[0] = -rotated_s1[0];
        rotated_s1[1] = -rotated_s1[1];
        rotated_s1[2] = -rotated_s1[2];
    }

    if (rotated_s1[1] == 0)
    {
        result.theta = atan(-rotated_s1[2] / (-sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[1], 2)) + 0.001));
        result.phi = atan(-rotated_s1[0] / (rotated_s1[1] + 0.001));
    }
    else
    {
        if (rotated_s1[1] < 0)
        {
            result.theta = atan(-rotated_s1[2] / sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[1], 2)));
            result.phi = atan(-rotated_s1[0] / (rotated_s1[1]));
        }
        else
        {
            if (rotated_s1[2] < 0)
            {
                result.theta = M_PI - atan(-rotated_s1[2] / sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[1], 2)));
            }
            else
            {
                result.theta = -M_PI - atan(-rotated_s1[2] / sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[1], 2)));
            }
            result.phi = -atan(-rotated_s1[0] / (-rotated_s1[1]));
        }
    }

    return result;
}

int which_face(int coord[3], int shape[3], float theta, float phi)
{
    // deciding which plane to go out, to see which direction (xyz) has increment of 1
    /*  'FRONTZY' = 1;
*   'LEYX' = 2 ;
*   'RIYX' = 3;
    'TOPZX' = 4;
    'BOTZX' = 5;
    "BACKZY" = 6 ;

*/
    /*
     * coord: the point which was calculated the ray length
     * shape: shape of the tomography matrix
     * theta: calculated theta angle to the point on the detector, positive means rotate clockwisely, vice versa
     * phi: calculated phi angle to the point on the detector, positive means rotate clockwisely
     * return: which face of the ray to exit, that represents the which (x,y,z) increment is 1
     *
     * top front left is the origin, not bottom front left
     */
    // the detector and the x-ray anti-clockwise rotation is positive
    float z_max = shape[0] - 1;
    float y_max = shape[1] - 1;
    float x_max = shape[2] - 1;
    float x = coord[2];
    float y = coord[1];
    float z = coord[0];
    if (test_mod)
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        printf("Memory usage: %d KB\n", usage.ru_maxrss);
    }
    if (fabs(theta) < M_PI / 2)
    {
        float theta_up = atan((y - 0) / (x - 0 + 0.001));
        float theta_down = -atan((y_max - y) / (x - 0 + 0.001)); // negative
        float phi_right = atan((z_max - z) / (x - 0 + 0.001));
        float phi_left = -atan((z - 0) / (x - 0 + 0.001)); // negative
        float omega = atan(tan(theta) * cos(phi));

        if (omega > theta_up)
        {
            // at this case, theta is positive,
            // normally the most cases for theta > theta_up, the ray passes the top ZX plane
            // if the phis are smaller than both edge limits
            // the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            float side = (y - 0) * sin(fabs(phi)) / tan(theta); // the length of rotation is the projected length on x
            if (side > (z - 0) && phi < phi_left)
            {
                return 2;
            }
            else if (side > (z_max - z) && phi > phi_right)
            {
                return 3;
            }
            else
            {
                return 4;
            }
        }
        else if (omega < theta_down)
        {
            float side = (y_max - y) * sin(fabs(phi)) / tan(-theta);
            if (side > (z - 0) && phi < phi_left)
            {
                return 2;
            }
            else if (side > (z_max - z) && phi > phi_right)
            {
                return 3;
            }
            else
            {
                return 5;
            }
        }
        else if (phi > phi_right)
        {
            // when the code goes to this line, it means the theta is within the limits
            return 3;
        }
        else if (phi < phi_left)
        {
            return 2;
        }
        else
        {
            // ray passes through the back plane
            return 6;
        }
    }
    else
    {
        // theta is larger than 90 degree or smaller than -90
        float theta_up = atan((y - 0) / (x_max - x + 0.001));
        float theta_down = atan((y_max - y) / (x_max - x + 0.001)); // negative
        float phi_left = atan((z_max - z) / (x_max - x + 0.001));   // it is the reverse of the top phi_left
        float phi_right = -atan((z - 0) / (x_max - x + 0.001));     // negative
        //
        //
        if ((M_PI - theta) > theta_up && theta > 0)
        {
            // at this case, theta is positive,
            // normally the most cases for theta > theta_up, the ray passes the top ZX plane
            // if the phis are smaller than both edge limits
            // the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            float side = (y - 0) * sin(fabs(phi)) / fabs(tan(theta));
            if (side > (z - 0) && -phi < phi_right)
            {
                return 2;
            }
            else if (side > (z_max - z) && -phi > phi_left)
            {
                return 3;
            }
            else
            {
                return 4;
            }
            //
        }
        else if (theta > theta_down - M_PI && theta <= 0)
        {
            float side = (y_max - y) * sin(fabs(phi)) / fabs(tan(-theta));
            if (side > (z - 0) && -phi < phi_right)
            {
                return 2;
            }
            else if (side > (z_max - z) && -phi > phi_left)
            {
                return 3;
            }
            else
            {
                return 5;
            }
        }
        else if (-phi < phi_right)
        {
            // when the code goes to this line, it means the theta is within the limits
            return 2;
        }
        else if (-phi > phi_left)
        {
            return 3;
        }
        else
        {
            // ray passes through the back plane
            return 1;
        }
    }
}

void dials_2_numpy(float input[3], float output[3])
{
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
}

void dials_2_numpy_matrix(float vector[3], float result[3])
{
    float numpy_2_dials_1[3][3] = {
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0}};

    for (int i = 0; i < 3; i++)
    {
        result[i] = 0.0;
        for (int j = 0; j < 3; j++)
        {
            result[i] += numpy_2_dials_1[i][j] * vector[j];
        }
    }
}

int cube_face(int ray_origin[3], float ray_direction[3], int cube_size[3], int L1)
{
    // deciding which plane to go out, to see which direction (xyz) has increment of 1
    /*  'FRONTZY' = 1;
*   'LEYX' = 2 ;
*   'RIYX' = 3;
    'TOPZX' = 4;
    'BOTZX' = 5;
    "BACKZY" = 6 ;

*/
    int min_x = 0;
    int max_x = cube_size[2];
    int min_y = 0;
    int max_y = cube_size[1];
    int min_z = 0;
    int max_z = cube_size[0];

    float tx_min = (min_x - ray_origin[2]) / ray_direction[2];
    float tx_max = (max_x - ray_origin[2]) / ray_direction[2];
    float ty_min = (min_y - ray_origin[1]) / ray_direction[1];
    float ty_max = (max_y - ray_origin[1]) / ray_direction[1];
    float tz_min = (min_z - ray_origin[0]) / ray_direction[0];
    float tz_max = (max_z - ray_origin[0]) / ray_direction[0];

    if (L1)
    {
        tx_min = -tx_min;
        tx_max = -tx_max;
        ty_min = -ty_min;
        ty_max = -ty_max;
        tz_min = -tz_min;
        tz_max = -tz_max;
    }

    float t_numbers[6] = {tx_min, ty_min, tz_min, tx_max, ty_max, tz_max};
    int t_numbers_len = sizeof(t_numbers) / sizeof(t_numbers[0]);

    float non_negative_numbers[t_numbers_len];
    int non_negative_len = 0;
    for (int i = 0; i < t_numbers_len; i++)
    {
        if (t_numbers[i] >= 0)
        {
            non_negative_numbers[non_negative_len++] = t_numbers[i];
        }
    }

    float t_min = non_negative_numbers[0];
    for (int i = 1; i < non_negative_len; i++)
    {
        if (non_negative_numbers[i] < t_min)
        {
            t_min = non_negative_numbers[i];
        }
    }
    // printf("t_min: %f\n", t_min);
    if (t_min == tx_min)
    {
        return 6;
    }
    else if (t_min == tx_max)
    {
        return 1;
    }
    else if (t_min == ty_min)
    {
        return 4;
    }
    else if (t_min == ty_max)
    {
        return 5;
    }
    else if (t_min == tz_min)
    {
        return 2;
    }
    else if (t_min == tz_max)
    {
        return 3;
    }
    else
    {
        fprintf(stderr, "face determination has a problem with direction %f, %f, %f and position %f, %f, %f\n", ray_direction[0], ray_direction[1],
                ray_direction[2], ray_origin[0], ray_origin[1], ray_origin[2]);
        exit(EXIT_FAILURE);
    }
    // if (t_min == tx_min)
    // {
    //     return L1 ? 1 : 6;
    // }
    // else if (t_min == tx_max)
    // {
    //     return L1 ? 6 : 1;
    // }
    // else if (t_min == ty_min)
    // {
    //     return L1 ? 5 : 4;
    // }
    // else if (t_min == ty_max)
    // {
    //     return L1 ? 4 : 5;
    // }
    // else if (t_min == tz_min)
    // {
    //     return L1 ? 3 : 2;
    // }
    // else if (t_min == tz_max)
    // {
    //     return L1 ? 2 : 3;
    // }
    // else
    // {
    //     fprintf(stderr, "face determination has a problem with direction %f, %f, %f and position %f, %f, %f\n", ray_direction[0], ray_direction[1],
    //             ray_direction[2], ray_origin[0], ray_origin[1], ray_origin[2]);
    //     exit(EXIT_FAILURE);
    // }
}

void appending(int increment, int *path_2,
               int *classes, int *classes_posi,
               int *potential_coord,
               int label, int previous_label,
               int *len_classes, int *len_classes_posi,
               int *len_path_2)
{
    if (label != previous_label)
    {

        if (label == 1)
        {
            classes[*len_classes] = 1;
            classes_posi[*len_classes_posi] = increment;
            (*len_classes_posi)++;
            (*len_classes)++;
        }
        else if (label == 2)
        {
            classes[*len_classes] = 2;
            classes_posi[*len_classes_posi] = increment;
            (*len_classes_posi)++;
            (*len_classes)++;
        }
        else if (label == 3)
        {
            classes[*len_classes] = 3;
            classes_posi[*len_classes_posi] = increment;
            (*len_classes_posi)++;
            (*len_classes)++;
        }
        else if (label == 4)
        {
            classes[*len_classes] = 4;
            classes_posi[*len_classes_posi] = increment;
            (*len_classes_posi)++;
            (*len_classes)++;
        }
        else if (label == 0)
        {
            classes[*len_classes] = 0;
            classes_posi[*len_classes_posi] = increment;
            (*len_classes_posi)++;
            (*len_classes)++;
        }
    }

    path_2[increment * 3] = potential_coord[0];
    path_2[increment * 3 + 1] = potential_coord[1];
    path_2[increment * 3 + 2] = potential_coord[2];
    (*len_path_2)++;
}

void get_increment_ratio(
    float *increment_ratio_x,
    float *increment_ratio_y,
    float *increment_ratio_z,
    float theta,
    float phi,
    float face)
{
    if (face == 1)
    {
        *increment_ratio_x = -1;
        *increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
        *increment_ratio_z = tan(phi);
    }
    else if (face == 2)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *increment_ratio_x = 1 / tan(fabs(phi));
            *increment_ratio_y = tan(theta) / sin(fabs(phi));
            *increment_ratio_z = -1;
        }
        else
        {
            *increment_ratio_x = 1 / tan(fabs(phi));
            *increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            *increment_ratio_z = -1;
        }
    }
    else if (face == 3)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *increment_ratio_x = 1 / tan(fabs(phi));
            *increment_ratio_y = tan(theta) / sin(fabs(phi));
            *increment_ratio_z = 1;
        }
        else
        {
            *increment_ratio_x = 1 / (tan(fabs(phi)));
            *increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            *increment_ratio_z = 1;
        }
    }
    else if (face == 4)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
            *increment_ratio_y = 1;
            *increment_ratio_z = sin(phi) / tan(fabs(theta));
        }
        else
        {
            *increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
            *increment_ratio_y = 1;
            *increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
        }
    }
    else if (face == 5)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
            *increment_ratio_y = -1;
            *increment_ratio_z = sin(phi) / (tan(fabs(theta)));
        }
        else
        {
            *increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
            *increment_ratio_y = -1;
            *increment_ratio_z = sin(phi) / (tan(M_PI - fabs(theta)));
        }
    }
    else if (face == 6)
    {
        *increment_ratio_x = -1;
        *increment_ratio_y = tan(theta) / cos(phi);
        *increment_ratio_z = tan(phi);
    }
    else
    {
        printf("ERROR! Unrecognised value of face.\n");
        *increment_ratio_x = 0;
        *increment_ratio_y = 0;
        *increment_ratio_z = 0;
    }
}

int get_maximum_increment(
    int x, int y, int z,
    int x_max, int y_max, int z_max,
    float theta, int face)
{
    if (face == 1)
    { // FRONTZY
        return (x_max - x);
    }
    else if (face == 2)
    { // LEYX
        return (z + 1);
    }
    else if (face == 3)
    { // RIYX
        if (fabs(theta) < M_PI / 2)
        {
            return (z_max - z);
        }
        else
        {
            return (z_max - z + 1);
        }
    }
    else if (face == 4)
    { // TOPZX
        return (y + 1);
    }
    else if (face == 5)
    { // BOTZX
        if (fabs(theta) < M_PI / 2)
        {
            return (y_max - y);
        }
        else
        {
            return (y_max - y + 1);
        }
    }
    else if (face == 6)
    { // BACKZY
        return (x + 1);
    }
    else
    {
        printf("ERROR! Unrecognised value of face.\n");
        return (0);
    }
}

void get_new_coordinates(
    int *new_x, int *new_y, int *new_z,
    int x, int y, int z,
    float increment_ratio_x, float increment_ratio_y, float increment_ratio_z,
    int increment, float theta, int face)
{
    if (face == 1)
    {
        if (theta > 0)
        {
            // this -1 represents that the opposition of direction
            // between the lab x-axis and the wavevector
            *new_x = floor(x - increment * increment_ratio_x);
            *new_y = floor(y - increment * increment_ratio_y);
            *new_z = floor(z - increment * increment_ratio_z);
        }
        else
        {
            // this -1 represents that the opposition of direction
            // between the lab x-axis and the wavevector
            *new_x = round(x - increment * increment_ratio_x);
            *new_y = round(y - increment * increment_ratio_y);
            *new_z = round(z - increment * increment_ratio_z);
        }
    }
    else if (face == 2)
    {
        if (fabs(theta) < M_PI / 2)
        {
            if (theta > 0)
            {
                *new_x = floor(x + -1 * increment * increment_ratio_x);
                *new_y = floor(y - increment * increment_ratio_y);
                *new_z = floor(z + increment * increment_ratio_z);
            }
            else
            {
                *new_x = round(x + -1 * increment * increment_ratio_x);
                *new_y = round(y - increment * increment_ratio_y);
                *new_z = round(z + increment * increment_ratio_z);
            }
        }
        else
        {
            if (theta > 0)
            {
                *new_x = floor(x + 1 * increment * increment_ratio_x);
                *new_y = floor(y - increment * increment_ratio_y);
                *new_z = floor(z + increment * increment_ratio_z);
            }
            else
            {
                *new_x = round(x + 1 * increment * increment_ratio_x);
                *new_y = round(y - increment * increment_ratio_y);
                *new_z = round(z + increment * increment_ratio_z);
            }
        }
    }
    else if (face == 3)
    {
        if (fabs(theta) < M_PI / 2)
        {
            if (theta > 0)
            {
                *new_x = floor(x + -1 * increment * increment_ratio_x);
                *new_y = floor(y - increment * increment_ratio_y);
                *new_z = floor(z + increment * increment_ratio_z);
            }
            else
            {
                *new_x = round(x + -1 * increment * increment_ratio_x);
                *new_y = round(y - increment * increment_ratio_y);
                *new_z = round(z + increment * increment_ratio_z);
            }
        }
        else
        {
            if (theta > 0)
            {
                *new_x = floor(x + 1 * increment * increment_ratio_x);
                *new_y = floor(y - increment * increment_ratio_y);
                *new_z = floor(z + increment * 1);
            }
            else
            {
                *new_x = round(x + 1 * increment * increment_ratio_x);
                *new_y = round(y - increment * increment_ratio_y);
                *new_z = round(z + increment * 1);
            }
        }
    }
    else if (face == 4)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *new_x = floor(x + -1 * increment * increment_ratio_x);
            *new_y = floor(y - increment * increment_ratio_y);
            *new_z = floor(z + increment * increment_ratio_z);
        }
        else
        {
            *new_x = floor(x + 1 * increment * increment_ratio_x);
            *new_y = floor(y - increment * increment_ratio_y);
            *new_z = floor(z + increment * increment_ratio_z);
        }
    }
    else if (face == 5)
    {
        if (fabs(theta) < M_PI / 2)
        {
            *new_x = round(x + -1 * increment * increment_ratio_x);
            *new_y = round(y - increment * increment_ratio_y);
            *new_z = round(z + increment * increment_ratio_z);
        }
        else
        {
            *new_x = round(x + 1 * increment * increment_ratio_x);
            *new_y = round(y - increment * increment_ratio_y);
            *new_z = round(z - increment * increment_ratio_z);
        }
    }
    else if (face == 6)
    {
        if (theta > 0)
        {
            *new_x = floor(x + increment * increment_ratio_x);
            *new_y = floor(y - increment * increment_ratio_y);
            *new_z = floor(z + increment * increment_ratio_z);
        }
        else
        {
            *new_x = round(x + increment * increment_ratio_x);
            *new_y = round(y - increment * increment_ratio_y);
            *new_z = round(z + increment * increment_ratio_z);
        }
    }
}

void check_boundaries(
    int *new_x, int *new_y, int *new_z,
    int x_max, int y_max, int z_max)
{
    if (*new_x >= x_max)
        *new_x = x_max - 1;
    else if (*new_x < 0)
        *new_x = 0;

    if (*new_y >= y_max)
        *new_y = y_max - 1;
    else if (*new_y < 0)
        *new_y = 0;

    if (*new_z >= z_max)
        *new_z = z_max - 1;
    else if (*new_z < 0)
        *new_z = 0;
}

Path2_c cal_coord(
    float theta, float phi, int *coord, int face,
    int *shape, int8_t *label_list_1d, int full_iteration)
{
    Path2_c result;
    int z = coord[0], y = coord[1], x = coord[2];
    int z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int diagonal = x_max * sqrt(3);

    int *path_2 = (int *)malloc(diagonal * 3 * sizeof(int));
    int *classes_posi = (int *)malloc(diagonal * sizeof(int));
    int *classes = (int *)malloc(diagonal * sizeof(int));
    classes[0] = 3;
    classes_posi[0] = 0;

    float increment_ratio_x, increment_ratio_y, increment_ratio_z;
    get_increment_ratio(&increment_ratio_x, &increment_ratio_y, &increment_ratio_z, theta, phi, face);

    int pos = 0;
    int len_path_2 = 1;
    int len_classes = 1;
    int len_classes_posi = 1;
    int new_z, new_y, new_x;
    int max_increment = get_maximum_increment(x, y, z, x_max, y_max, z_max, theta, face);

    if (face >= 1 && face <= 6)
    {
        // for (int increment = 0; increment < max_increment; increment++){
        for (int increment = 0; increment < diagonal; increment++)
        {
            get_new_coordinates(
                &new_x, &new_y, &new_z,
                x, y, z,
                increment_ratio_x, increment_ratio_y, increment_ratio_z,
                increment, theta, face);

            // check_boundaries(&new_x, &new_y, &new_z, x_max, y_max, z_max);

            if (
                new_x >= x_max || new_x < 0 ||
                new_y >= y_max || new_y < 0 ||
                new_z >= z_max || new_z < 0)
                break;

            int potential_coord[3] = {new_z, new_y, new_x};
            // int label = label_list[new_z][new_y][new_x];
            pos = INDEX_3D(z_max, y_max, x_max, new_z, new_y, new_x);
            int label = label_list_1d[pos];
            if (!full_iteration)
            {
                if (label == 0)
                {
                    break;
                }
            }

            if (increment == 0)
            {
                path_2[increment * 3] = potential_coord[0];
                path_2[increment * 3 + 1] = potential_coord[1];
                path_2[increment * 3 + 2] = potential_coord[2];
                continue;
            }

            int previous_step[3] = {
                path_2[(increment - 1) * 3],
                path_2[(increment - 1) * 3 + 1],
                path_2[(increment - 1) * 3 + 2]};
            pos = INDEX_3D(z_max, y_max, x_max, previous_step[0], previous_step[1], previous_step[2]);
            int previous_label = label_list_1d[pos];
            appending(increment, path_2,
                      classes, classes_posi,
                      potential_coord,
                      label, previous_label,
                      &len_classes, &len_classes_posi, &len_path_2);
        }
    }
    else
    {
        printf("Error: face is not in the range of 1 to 6");
    }

    result.len_path_2 = len_path_2;
    result.len_classes = len_classes;
    result.len_classes_posi = len_classes_posi;
    result.posi = (int *)malloc(len_classes_posi * sizeof(int));
    result.classes = (int *)malloc(len_classes * sizeof(int));
    result.ray = (int *)malloc(len_path_2 * 3 * sizeof(int));
    result.ray_classes = (int *)malloc(len_path_2 * sizeof(int));
    for (int i = 0; i < len_path_2 * 3; i++)
    {
        result.ray[i] = path_2[i];
    }
    for (int i = 0; i < len_path_2; i++)
    {
        int pos = INDEX_3D(z_max, y_max, x_max, path_2[3 * i + 0], path_2[3 * i + 1], path_2[3 * i + 2]);
        int label = label_list_1d[pos];
        result.ray_classes[i] = label;
    }
    for (int i = 0; i < len_classes_posi; i++)
    {
        result.posi[i] = classes_posi[i];
    }
    for (int i = 0; i < len_classes; i++)
    {
        result.classes[i] = classes[i];
    }
    // printArray(result.ray, 30);
    // printArray(result.posi, result.len_classes_posi);
    // printArray(result.classes, result.len_classes);
    if (test_mod)
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        printf("Memory usage: %d KB\n", usage.ru_maxrss);
    }

    free(path_2);
    free(classes_posi);
    free(classes);
    // malloc_trim(0);
    return result;
}
//**********************************************

Path2_c cal_coord_ref(float theta, float phi, int *coord, int face,
                      int *shape, int8_t ***label_list, int full_iteration)
{
    Path2_c result;
    int z = coord[0], y = coord[1], x = coord[2];
    int z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int diagonal = x_max * sqrt(3);

    int *path_2 = (int *)malloc(diagonal * 3 * sizeof(int));
    int *classes_posi = (int *)malloc(diagonal * sizeof(int));
    int *classes = (int *)malloc(diagonal * sizeof(int));
    classes[0] = 3;
    classes_posi[0] = 0;
    // int path_2[x_max*y_max*z_max][3];
    // int classes_posi[x_max*y_max*z_max];
    // int classes[x_max*y_max*z_max];
    float increment_ratio_x, increment_ratio_y, increment_ratio_z;

    int len_path_2 = 1;
    int len_classes = 1;
    int len_classes_posi = 1;
    int new_z, new_y, new_x;
    if (face == 6)
    {
        // assert(fabs(theta) <= M_PI / 2);
        increment_ratio_x = -1;
        increment_ratio_y = tan(theta) / cos(phi);
        increment_ratio_z = tan(phi);
        for (int increment = 0; increment <= x - 0; increment++)
        {

            if (theta > 0)
            {
                new_x = floor(x + increment * increment_ratio_x);
                new_y = floor(y - increment * increment_ratio_y);
                new_z = floor(z + increment * increment_ratio_z);
            }
            else
            {
                new_x = round(x + increment * increment_ratio_x);
                new_y = round(y - increment * increment_ratio_y);
                new_z = round(z + increment * increment_ratio_z);
            }
            if (new_y >= y_max)
            {
                new_y = y_max - 1;
            }
            else if (new_y < 0)
            {
                new_y = 0;
            }

            if (new_x >= x_max)
            {
                new_x = x_max - 1;
            }
            else if (new_x < 0)
            {
                new_x = 0;
            }

            if (new_z >= z_max)
            {
                new_z = z_max - 1;
            }
            else if (new_z < 0)
            {
                new_z = 0;
            }
            int potential_coord[3] = {new_z, new_y, new_x};
            int label = label_list[new_z][new_y][new_x];
            if (!full_iteration)
            {
                if (label == 0)
                {
                    break;
                }
            }

            if (increment == 0)
            {
                path_2[increment * 3] = potential_coord[0];
                path_2[increment * 3 + 1] = potential_coord[1];
                path_2[increment * 3 + 2] = potential_coord[2];
                continue;
            }
            //  else if (label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]] != label_list[path_2[increment - 1]]) {
            //     int label = label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]];
            int previous_step[3] = {path_2[(increment - 1) * 3],
                                        path_2[(increment - 1) * 3 + 1],
                                        path_2[(increment - 1) * 3 + 2]};
            int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];
            appending(increment, path_2,
                      classes, classes_posi,
                      potential_coord,
                      label, previous_label,
                      &len_classes, &len_classes_posi, &len_path_2);
        }
    }

    else if (face == 2)
    {

        if (fabs(theta) < M_PI / 2)
        {
            float increment_ratio_x = 1 / tan(fabs(phi));
            float increment_ratio_y = tan(theta) / sin(fabs(phi));
            float increment_ratio_z = -1;

            for (int increment = 0; increment <= z; increment++)
            {

                if (theta > 0)
                {
                    new_x = floor(x + -1 * increment * increment_ratio_x);
                    new_y = floor(y - increment * increment_ratio_y);
                    new_z = floor(z + increment * increment_ratio_z);
                }
                else
                {
                    new_x = round(x + -1 * increment * increment_ratio_x);
                    new_y = round(y - increment * increment_ratio_y);
                    new_z = round(z + increment * increment_ratio_z);
                }
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                // printf("new_x: %d, new_y: %d, new_z: %d \n", new_x, new_y, new_z);
                // if (test_mod)
                // {
                //     printf("new_x: %d, new_y: %d, new_z: %d \n", new_x, new_y, new_z);
                // }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                // if (test_mod){
                //     printArray(potential_coord, 3);
                //     printf("label: %d \n", label);

                // }

                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
        else
        {
            float increment_ratio_x = 1 / tan(fabs(phi));
            float increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            float increment_ratio_z = -1;

            for (int increment = 0; increment <= (z - 0); increment++)
            {
                if (theta > 0)
                {
                    new_x = floor(x + 1 * increment * increment_ratio_x);
                    new_y = floor(y - increment * increment_ratio_y);
                    new_z = floor(z + increment * increment_ratio_z);
                }
                else
                {
                    new_x = round(x + 1 * increment * increment_ratio_x);
                    new_y = round(y - increment * increment_ratio_y);
                    new_z = round(z + increment * increment_ratio_z);
                }

                // if (test_mod)
                // {
                //     printf("new_x: %d, new_y: %d, new_z: %d \n", new_x, new_y, new_z);
                // }
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
    }

    else if (face == 3)
    {

        if (fabs(theta) < M_PI / 2)
        {
            float increment_ratio_x = 1 / tan(fabs(phi));
            float increment_ratio_y = tan(theta) / sin(fabs(phi));
            float increment_ratio_z = 1;
            for (int increment = 0; increment < (z_max - z); increment++)
            {
                if (theta > 0)
                {
                    new_x = floor(x + -1 * increment * increment_ratio_x);
                    new_y = floor(y - increment * increment_ratio_y);
                    new_z = floor(z + increment * increment_ratio_z);
                }
                else
                {
                    new_x = round(x + -1 * increment * increment_ratio_x);
                    new_y = round(y - increment * increment_ratio_y);
                    new_z = round(z + increment * increment_ratio_z);
                }
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }

                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }

        else
        {
            float increment_ratio_x = 1 / (tan(fabs(phi)));
            float increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            float increment_ratio_z = 1;
            for (int increment = 0; increment <= (z_max - z); increment++)
            {
                // increment on z-axis
                // new_x = x + 1 * increment / (tan(fabs(phi)))
                // new_y = y - increment * tan(M_PI - theta) / sin(fabs(phi))
                // new_z = z + increment * 1
                if (theta > 0)
                {
                    new_x = floor(x + 1 * increment * increment_ratio_x);
                    new_y = floor(y - increment * increment_ratio_y);
                    new_z = floor(z + increment * 1);
                }
                else
                {
                    new_x = round(x + 1 * increment * increment_ratio_x);
                    new_y = round(y - increment * increment_ratio_y);
                    new_z = round(z + increment * 1);
                }
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
    }

    else if (face == 4)
    {
        //    assert(theta > 0);
        if (fabs(theta) < M_PI / 2)
        {
            float increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
            float increment_ratio_y = 1;
            float increment_ratio_z = sin(phi) / tan(fabs(theta));

            for (int increment = 0; increment <= y - 0; increment++)
            {
                new_x = floor(x + -1 * increment * increment_ratio_x);
                new_y = floor(y - increment * increment_ratio_y);
                new_z = floor(z + increment * increment_ratio_z);
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
        else
        {
            float increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
            float increment_ratio_y = 1;
            float increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
            for (int increment = 0; increment < y - 0 + 1; increment++)
            {
                new_x = floor(x + 1 * increment * increment_ratio_x);
                new_y = floor(y - increment * increment_ratio_y);
                new_z = floor(z + increment * increment_ratio_z);
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
                // if (label != previous_label)
                // {

                //     if (label == 1)
                //     {
                //         classes[len_classes] = 1;
                //         classes_posi[len_classes_posi] = increment;
                //         len_classes_posi++;
                //         len_classes++;
                //     }
                //     else if (label == 2)
                //     {
                //         classes[len_classes] = 2;
                //         classes_posi[len_classes_posi] = increment;
                //         len_classes_posi++;
                //         len_classes++;
                //     }
                //     else if (label == 3)
                //     {
                //         classes[len_classes] = 3;
                //         classes_posi[len_classes_posi] = increment;
                //         len_classes_posi++;
                //         len_classes++;
                //     }
                //     else if (label == 4)
                //     {
                //         classes[len_classes] = 4;
                //         classes_posi[len_classes_posi] = increment;
                //         len_classes_posi++;
                //         len_classes++;
                //     }
                //     else if (label == 0)
                //     {
                //         classes[len_classes] = 0;
                //         classes_posi[len_classes_posi] = increment;
                //         len_classes_posi++;
                //     }
                // }

                // path_2[increment * 3] = potential_coord[0];
                // path_2[increment * 3 + 1] = potential_coord[1];
                // path_2[increment * 3 + 2] = potential_coord[2];
                // len_path_2++;
            }
        }
    }

    else if (face == 5)
    {

        if (fabs(theta) < M_PI / 2)
        {
            float increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
            float increment_ratio_y = -1;
            float increment_ratio_z = sin(phi) / (tan(fabs(theta)));
            // printf("increment_ratio_x: %f, increment_ratio_y: %f, increment_ratio_z: %f \n", increment_ratio_x, increment_ratio_y, increment_ratio_z);
            // printArray(coord, 3);
            for (int increment = 0; increment < y_max - y; increment++)
            {
                // decrement on y-axis
                // new_x = x + -1 * increment * np.cos(np.abs(phi))/(np.tan(np.abs(theta)))
                // new_y = y - increment*-1
                // new_z = z + increment*np.sin(phi)/ ( np.tan(np.abs(theta)) )
                new_x = round(x + -1 * increment * increment_ratio_x);
                new_y = round(y - increment * increment_ratio_y);
                new_z = round(z + increment * increment_ratio_z);
                // printf("increment %d", increment);
                // printf("new_x: %d, new_y: %d, new_z: %d \n", new_x, new_y, new_z);
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
                // if (label != previous_label)
            }
        }

        else
        {
            increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
            increment_ratio_y = -1;
            increment_ratio_z = sin(phi) / (tan(M_PI - fabs(theta)));
            for (int increment = 0; increment <= y_max - y; increment++)
            {
                // decrement on y-axis
                // new_x = x + 1 * increment * np.cos(np.abs(phi)) / ( np.tan(np.abs(np.pi-theta)) )
                // new_y = y - increment * -1
                // new_z = z - increment * np.sin(phi) / ( np.tan(np.abs(np.pi-theta)) ) #
                new_x = round(x + 1 * increment * increment_ratio_x);
                new_y = round(y - increment * increment_ratio_y);
                new_z = round(z - increment * increment_ratio_z);
                if (new_y >= y_max)
                {
                    new_y = y_max - 1;
                }
                else if (new_y < 0)
                {
                    new_y = 0;
                }

                if (new_x >= x_max)
                {
                    new_x = x_max - 1;
                }
                else if (new_x < 0)
                {
                    new_x = 0;
                }

                if (new_z >= z_max)
                {
                    new_z = z_max - 1;
                }
                else if (new_z < 0)
                {
                    new_z = 0;
                }
                int potential_coord[3] = {new_z, new_y, new_x};
                int label = label_list[new_z][new_y][new_x];
                if (!full_iteration)
                {
                    if (label == 0)
                    {
                        break;
                    }
                }

                if (increment == 0)
                {
                    path_2[increment * 3] = potential_coord[0];
                    path_2[increment * 3 + 1] = potential_coord[1];
                    path_2[increment * 3 + 2] = potential_coord[2];
                    continue;
                }
                int previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
    }

    else if (face == 1)

    {
        // assert(fabs(theta) <= M_PI / 2);
        increment_ratio_x = -1;
        increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
        increment_ratio_z = tan(phi);

        for (int increment = 0; increment < x_max - x; increment++)
        {
            // the absorption also count that coordinate in the path_2
            // decrement on x axis
            if (theta > 0)
            {
                new_x = floor(x - increment * increment_ratio_x); // this -1 represents that the opposition of direction
                // between the lab x-axis and the wavevector
                new_y = floor(y - increment * increment_ratio_y);
                new_z = floor(z - increment * increment_ratio_z);
            }
            else
            {
                new_x = round(x - increment * increment_ratio_x); // this -1 represents that the opposition of direction
                                                                  // between the lab x-axis and the wavevector
                new_y = round(y - increment * increment_ratio_y);
                new_z = round(z - increment * increment_ratio_z);
            }
            // printf("new_x: %d, new_y: %d, new_z: %d", new_x, new_y, new_z);
            if (new_y >= y_max)
            {
                new_y = y_max - 1;
            }
            else if (new_y < 0)
            {
                new_y = 0;
            }

            if (new_x >= x_max)
            {
                new_x = x_max - 1;
            }
            else if (new_x < 0)
            {
                new_x = 0;
            }

            if (new_z >= z_max)
            {
                new_z = z_max - 1;
            }
            else if (new_z < 0)
            {
                new_z = 0;
            }
            // printf("new_x: %d, new_y: %d, new_z: %d", new_x, new_y, new_z);
            int potential_coord[3] = {new_z, new_y, new_x};
            int label = label_list[new_z][new_y][new_x];
            if (!full_iteration)
            {
                if (label == 0)
                {
                    break;
                }
            }

            if (increment == 0)
            {
                path_2[increment * 3] = potential_coord[0];
                path_2[increment * 3 + 1] = potential_coord[1];
                path_2[increment * 3 + 2] = potential_coord[2];
                continue;
            }
            //  else if (label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]] != label_list[path_2[increment - 1]]) {
            //     int label = label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]];
            int previous_step[3] = {path_2[(increment - 1) * 3],
                                        path_2[(increment - 1) * 3 + 1],
                                        path_2[(increment - 1) * 3 + 2]};
            int previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];
            appending(increment, path_2,
                      classes, classes_posi,
                      potential_coord,
                      label, previous_label,
                      &len_classes, &len_classes_posi, &len_path_2);
        }
    }

    else
    {
        printf("Error: face is not in the range of 1 to 6");
    }
    // printArray(path_2, 6);
    // printArray(classes_posi, 6);
    // printArray(classes, 6);
    // printf("Length of 2d array in C: %d \n", len_path_2);
    // printf("Length of classes in C: %d \n", len_classes);
    // printf("Length of classes_posi in C: %d \n", len_classes_posi);
    // result.ray = path_2;
    // result.posi = classes_posi;
    // result.classes = classes;

    result.len_path_2 = len_path_2;
    result.len_classes = len_classes;
    result.len_classes_posi = len_classes_posi;
    result.posi = (int *)malloc(len_classes_posi * sizeof(int));
    result.classes = (int *)malloc(len_classes * sizeof(int));
    result.ray = (int *)malloc(len_path_2 * 3 * sizeof(int));
    result.ray_classes = (int *)malloc(len_path_2 * sizeof(int));
    for (int i = 0; i < len_path_2 * 3; i++)
    {
        result.ray[i] = path_2[i];
    }
    for (int i = 0; i < len_path_2; i++)
    {
        int label = label_list[path_2[3 * i + 0]][path_2[3 * i + 1]][path_2[3 * i + 2]];
        result.ray_classes[i] = label;
    }
    for (int i = 0; i < len_classes_posi; i++)
    {
        // printf("classes_posi is %d \n", classes_posi[i]);
        result.posi[i] = classes_posi[i];
    }
    for (int i = 0; i < len_classes; i++)
    {
        result.classes[i] = classes[i];
    }
    // printArray(result.ray, 30);
    // printArray(result.posi, result.len_classes_posi);
    // printArray(result.classes, result.len_classes);
    if (test_mod)
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        printf("Memory usage: %d KB\n", usage.ru_maxrss);
    }

    if (test_mod)
    {

        printf("diagonal is %d \n", diagonal);
        printf("len_path_2 is %d \n", len_path_2);
        printf("len_classes is %d \n", len_classes);
        printf("len_classes_posi is %d \n", len_classes_posi);
        // printArray(path_2, len_path_2*3);
        printArray(classes_posi, len_classes_posi);
        printArray(classes, len_classes);
    }

    free(path_2);
    if (test_mod)
    {
        printf("path_2 is free \n");
    }

    free(classes_posi);
    if (test_mod)
    {
        printf("classes_posi is free \n");
    }

    free(classes);
    if (test_mod)
    {
        printf(" class is free \n");
    }
    // malloc_trim(0);
    return result;
}

//**********************************************

float *cal_path2_plus(Path2_c path_2_cal_result, float *voxel_size)
{
    float *result = (float *)malloc(4 * sizeof(float));
    float voxel_length_z = voxel_size[0];
    float voxel_length_y = voxel_size[1];
    float voxel_length_x = voxel_size[2];
    int *path_ray = path_2_cal_result.ray;
    int *path_ray_classes = path_2_cal_result.ray_classes;
    int *posi = path_2_cal_result.posi;
    int *classes = path_2_cal_result.classes;
    int len_path_2 = path_2_cal_result.len_path_2;
    int len_classes = path_2_cal_result.len_classes;
    int len_classes_posi = path_2_cal_result.len_classes_posi;

    float dist_x = (path_ray[(len_path_2 - 1) * 3 + 2] - path_ray[2]);
    float dist_y = (path_ray[(len_path_2 - 1) * 3 + 1] - path_ray[1]);
    float dist_z = (path_ray[(len_path_2 - 1) * 3 + 0] - path_ray[0]);
    float total_length = sqrt(
        pow(dist_y * voxel_length_y, 2) +
        pow(dist_z * voxel_length_z, 2) +
        pow(dist_x * voxel_length_x, 2));
    // printf("CPU:==>total_length=%f; ", total_length);
    // printf ("len_path_2=%d\n", len_path_2);
    // printf ("dist_x=%f , dist_y=%f, dist_z=%f \n", dist_x, dist_y, dist_z);
    int cr_l_2_int = 0;
    int li_l_2_int = 0;
    int bu_l_2_int = 0;
    int lo_l_2_int = 0;

    for (int j = 0; j < len_path_2; j++)
    {
        if (path_ray_classes[j] == 3)
            cr_l_2_int++;
        else if (path_ray_classes[j] == 1)
            li_l_2_int++;
        else if (path_ray_classes[j] == 2)
            lo_l_2_int++;
        else if (path_ray_classes[j] == 4)
            bu_l_2_int++;
        else
        {
        }
    }
    // printf("li_l_2_int=%d; ", li_l_2_int);
    // printf("lo_l_2_int=%d; ", lo_l_2_int);
    // printf("cr_l_2_int=%d; ", cr_l_2_int);
    // printf("bu_l_2_int=%d\n", bu_l_2_int);
    // printf("path_ray_classes is \n")   ;
    // for(int k =0; k<len_path_2;k++){
    //     printf(" %d",path_ray_classes[k]);
    // }
    // printf("\n");
    int sum = cr_l_2_int + li_l_2_int + lo_l_2_int + bu_l_2_int;
    // printf("total_length=%f; len_path_2=%f; sum=%d; dst=[%f; %f; %f]\n", total_length, (float) len_path_2, sum, dist_x, dist_y, dist_z);
    float cr_l_2 = total_length * (((float)cr_l_2_int) / ((float)len_path_2));
    float li_l_2 = total_length * (((float)li_l_2_int) / ((float)len_path_2));
    float bu_l_2 = total_length * (((float)bu_l_2_int) / ((float)len_path_2));
    float lo_l_2 = total_length * (((float)lo_l_2_int) / ((float)len_path_2));
    result[2] = cr_l_2;
    result[1] = lo_l_2;
    result[0] = li_l_2;
    result[3] = bu_l_2;
    return result;
}

float *cal_path2_plus_ref(Path2_c path_2_cal_result, float *voxel_size)
{
    float *result = (float *)malloc(4 * sizeof(float));
    float voxel_length_z = voxel_size[0];
    float voxel_length_y = voxel_size[1];
    float voxel_length_x = voxel_size[2];
    int *path_ray = path_2_cal_result.ray;
    int *posi = path_2_cal_result.posi;
    int *classes = path_2_cal_result.classes;
    int len_path_2 = path_2_cal_result.len_path_2;
    int len_classes = path_2_cal_result.len_classes;
    int len_classes_posi = path_2_cal_result.len_classes_posi;

    float cr_l_2 = 0;
    float li_l_2 = 0;
    float bu_l_2 = 0;
    float lo_l_2 = 0;

    float total_length = sqrt(pow((path_ray[(len_path_2 - 1) * 3 + 1] - path_ray[1]) * voxel_length_y, 2) +
                               pow((path_ray[(len_path_2 - 1) * 3 + 0] - path_ray[0]) * voxel_length_z, 2) +
                               pow((path_ray[(len_path_2 - 1) * 3 + 2] - path_ray[2]) * voxel_length_x, 2));

    for (int j = 0; j < len_classes_posi; j++)
    {
        if (classes[j] == 3)
        {
            if (j < len_classes_posi - 1)
            {
                cr_l_2 += total_length * ((float)(posi[j + 1] - posi[j]) / (float)len_path_2);
            }
            else
            {
                cr_l_2 += total_length * ((float)(len_path_2 - posi[j]) / (float)len_path_2);
            }
        }
        else if (classes[j] == 1)
        {
            if (j < len_classes_posi - 1)
            {
                li_l_2 += total_length * ((float)(posi[j + 1] - posi[j]) / (float)len_path_2);
            }
            else
            {
                li_l_2 += total_length * ((float)(len_path_2 - posi[j]) / (float)len_path_2);
            }
        }
        else if (classes[j] == 2)
        {
            if (j < len_classes_posi - 1)
            {
                lo_l_2 += total_length * ((float)(posi[j + 1] - posi[j]) / (float)len_path_2);
            }
            else
            {
                lo_l_2 += total_length * ((float)(len_path_2 - posi[j]) / (float)len_path_2);
            }
        }
        else if (classes[j] == 4)
        {
            if (j < len_classes_posi - 1)
            {
                bu_l_2 += total_length * ((float)(posi[j + 1] - posi[j]) / (float)len_path_2);
            }
            else
            {
                bu_l_2 += total_length * ((float)(len_path_2 - posi[j]) / (float)len_path_2);
            }
        }
        else
        {
        }
    }

    result[2] = cr_l_2;
    result[1] = lo_l_2;
    result[0] = li_l_2;
    result[3] = bu_l_2;
    return result;
}

float cal_rate(float *numbers_1, float *numbers_2, float *coefficients,
                char Isexp)
{

    float mu_li = coefficients[0];
    float mu_lo = coefficients[1];
    float mu_cr = coefficients[2];
    float mu_bu = coefficients[3];

    float li_l_1 = numbers_1[0];
    float lo_l_1 = numbers_1[1];
    float cr_l_1 = numbers_1[2];
    float bu_l_1 = numbers_1[3];

    float li_l_2 = numbers_2[0];
    float lo_l_2 = numbers_2[1];
    float cr_l_2 = numbers_2[2];
    float bu_l_2 = numbers_2[3];

    float result = (mu_li * (li_l_1 + li_l_2) +
                     mu_lo * (lo_l_1 + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                     mu_bu * (bu_l_1 + bu_l_2));

    if (Isexp == 1)
    {
        result = exp(-result);
    }

    return result;
}

float cal_rate_single(float *numbers, float *coefficients,
                       char Isexp)
{

    float mu_li = coefficients[0];
    float mu_lo = coefficients[1];
    float mu_cr = coefficients[2];
    float mu_bu = coefficients[3];

    float li_l_1 = numbers[0];
    float lo_l_1 = numbers[1];
    float cr_l_1 = numbers[2];
    float bu_l_1 = numbers[3];

    float result = (mu_li * (li_l_1) +
                     mu_lo * (lo_l_1) +
                     mu_cr * (cr_l_1) +
                     mu_bu * (bu_l_1));

    if (Isexp == 1)
    {
        result = exp(-result);
    }

    return result;
}



int ray_tracing_gpu_overall_kernel(int32_t low, int32_t up,
                                   int *coord_list,
                                   int64_t len_coord_list,
                                   const float *scattering_vector_list, const float *omega_list,
                                   const float *raw_xray,
                                   const float *omega_axis, const float *kp_rotation_matrix,
                                   int64_t len_result,
                                   float *voxel_size, float *coefficients,
                                   int8_t *label_list_1d, int *shape, int32_t full_iteration,
                                   int32_t store_paths, float *h_result_list, int *h_face, float *h_angles, float *h_python_result_list);

#ifdef __cplusplus
extern "C"
{
#endif

    // float ray_tracing_gpu(
    //     int *coord_list,
    //     int len_coord_list,
    //     float *rotated_s1, float *xray,
    //     float *voxel_size, float *coefficients,
    //     int8_t *label_list_1d, int *shape, int full_iteration,
    //     int store_paths)
    // {
    //     printf("\n------------------ Ray tracing sampling --------------\n");
    //     printf("--------------> GPU version\n");
    //     int *h_face, *h_ray_classes;
    //     float *h_angles, *h_absorption;
    //     int z_max = shape[0], y_max = shape[1], x_max = shape[2];
    //     int diagonal = x_max * sqrt(3);
    //     int face_size = len_coord_list * 2 * sizeof(int);
    //     int absorption_size = len_coord_list * 2 * sizeof(float);
    //     int ray_classes_size = diagonal * len_coord_list * 2 * sizeof(int);
    //     int angle_size = 4 * sizeof(float);

    //     h_face = (int *)malloc(face_size);
    //     h_ray_classes = (int *)malloc(ray_classes_size);
    //     h_angles = (float *)malloc(angle_size);
    //     h_absorption = (float *)malloc(absorption_size);
    //     ray_tracing_path(h_face, h_angles, h_ray_classes, h_absorption, coord_list, len_coord_list, rotated_s1, xray, voxel_size, coefficients, label_list_1d, shape);

    //     printf("----> GPU version FINISHED;\n");

    //     float gpu_absorption = 0;
    //     for (int i = 0; i < len_coord_list; i++)
    //     {
    //         gpu_absorption += exp(-(h_absorption[2 * i + 0] + h_absorption[2 * i + 1]));
    //     }
    //     float gpu_absorption_mean = gpu_absorption / ((float)len_coord_list);

    //     free(h_face);
    //     free(h_ray_classes);
    //     free(h_angles);
    //     free(h_absorption);

    //     return gpu_absorption_mean;
    // }

    // // for (int i = 0; i < len_result; i++)
    // // {
    // //     float result;
    // //     float rotation_matrix_frame_omega[9];
    // //     float rotation_matrix[9];
    // //     float total_rotation_matrix[9];
    // //     float xray[3];
    // //     float rotated_s1[3];
    // //     kp_rotation(omega_axis, omega_list[i], (float *)rotation_matrix_frame_omega);
    // //     dot_product((float *)rotation_matrix_frame_omega, kp_rotation_matrix, (float *)rotation_matrix, 3, 3, 3);
    // //     transpose((float *)rotation_matrix, 3, 3, (float *)total_rotation_matrix);
    // //     dot_product((float *)total_rotation_matrix, raw_xray, (float *)xray, 3, 3, 1);
    // //     // printf("xray is \n");
    // //     // print_matrix(xray,1,3);
    // //     float scattering_vector[3] = {scattering_vector_list[i * 3],
    // //                                    scattering_vector_list[i * 3 + 1],
    // //                                    scattering_vector_list[i * 3 + 2]};
    // //     dot_product((float *)total_rotation_matrix, (float *)scattering_vector, (float *)rotated_s1, 3, 3, 1);

    // //     result = ray_tracing_gpu(
    // //         coord_list, len_coord_list,
    // //         (float *)rotated_s1, (float *)xray,
    // //         voxel_size, coefficients,
    // //         label_list_1d, shape, full_iteration,
    // //         store_paths);
    // //     // printf("result is %f \n",result);
    // //     result_list[i] = result;
    // //     printf("[%d/%d] rotation: %.4f, absorption: %.4f\n",
    // //            low + i, up, omega_list[i] * 180 / M_PI, result);
    // //     break;
    // //     // printf("index is %d, result is %f \n",i,result);
    // //     // printArrayD(result_list, 10);
    // // }

    // // return result_list;
    // // }
    // float ray_tracing_sampling_od(
    //     int *coord_list,
    //     int len_coord_list,
    //     const float *rotated_s1, const float *xray,
    //     float *voxel_size, float *coefficients,
    //     int8_t *label_list, int *shape, int full_iteration,
    //     int store_paths)
    // {
    //     // print_matrix(rotated_s1, 1, 3);
    //     // print_matrix(xray, 1, 3);
    //     // printArray(crystal_coordinate_shape, 3);
    //     // printArrayD(rotated_s1, 3);
    //     // printArrayD(xray, 3);
    //     // if (test_mod)
    //     // {
    //     //     struct rusage usage;
    //     //     getrusage(RUSAGE_SELF, &usage);
    //     //     printf("The starting Memory usage: %d KB\n", usage.ru_maxrss);
    //     // }

    //     // in the theta phi determination, xray will be reversed
    //     // so create a new array to store the original xray to process

    //     float x_ray_angle[3], x_ray_trans[3];
    //     float rotated_s1_angle[3], rotated_s1_trans[3];
    //     memcpy(x_ray_angle, xray, 3 * sizeof(xray));
    //     memcpy(x_ray_trans, xray, 3 * sizeof(xray));
    //     memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(rotated_s1));
    //     memcpy(rotated_s1_trans, rotated_s1, 3 * sizeof(rotated_s1));

    //     // for (int i = 0; i < 3; i++)
    //     // {
    //     //     x_ray_c[i] = xray[i];
    //     // }

    //     // printArrayD(voxel_size, 3);
    //     // printArrayD(coefficients, 3);
    //     // printArray(shape, 3);
    //     // printf("%d ", len_coordinate_list);
    //     ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
    //     ThetaPhi result_1 = dials_2_thetaphi_22(x_ray_angle, 1);
    //     // printf("rotated_s1_angle \n");
    //     // print_matrix(rotated_s1_angle, 1, 3);
    //     // printf("x_ray_angle \n");
    //     // print_matrix(x_ray_angle, 1, 3);
    //     // printf("\n");
    //     float theta = result_2.theta;
    //     float phi = result_2.phi;
    //     float theta_1 = result_1.theta;
    //     float phi_1 = result_1.phi;
    //     // printf("\n");
    //     // printf("theta: %f\n", theta);
    //     // printf("phi: %f\n", phi);
    //     // printf("theta_1: %f\n", theta_1);
    //     // printf("phi_1: %f\n", phi_1);
    //     Path2_c path_2, path_1;
    //     float *numbers_1, *numbers_2;
    //     float absorption;
    //     float absorption_sum = 0, absorption_mean = 0;

    //     float xray_direction[3], scattered_direction[3];
    //     dials_2_numpy(x_ray_trans, xray_direction);
    //     dials_2_numpy(rotated_s1_trans, scattered_direction);

    //     // if (test_mod)
    //     // {
    //     //     theta_1 = 0.660531;
    //     //     phi_1 = -0.001338;
    //     //     theta = -1.557793;
    //     //     phi = -1.560976;
    //     // }
    //     for (int i = 0; i < len_coord_list; i++)
    //     {

    //         int coord[3] = {coord_list[i * 3],
    //                             coord_list[i * 3 + 1],
    //                             coord_list[i * 3 + 2]};
    //         // printf("%d ",label_list[coord[0]][coord[1]][coord[2]]);
    //         // int face_1 = which_face(coord, shape, theta_1, phi_1);
    //         // int face_2 = which_face(coord, shape, theta, phi);

    //         int face_1 = cube_face(coord, xray_direction, shape, 1);
    //         int face_2 = cube_face(coord, scattered_direction, shape, 0);
    //         if (face_1 == 1 && fabs(theta_1) < M_PI / 2)
    //         {
    //             printArray(coord, 3);
    //             printf("face_1 is  %d \n", face_1);
    //             printf("theta_1 is %f ", theta_1);
    //             printf("phi_1 is %f \n", phi_1);
    //             print_matrix(xray, 1, 3);
    //         }
    //         if (face_2 == 1 && fabs(theta) < M_PI / 2)
    //         {
    //             printArray(coord, 3);
    //             printf("face_2 is  %d \n", face_2);
    //             printf("theta is %f ", theta);
    //             printf("phi is %f \n", phi);
    //             print_matrix(rotated_s1, 1, 3);
    //         }
    //         if (test_mod)
    //         {
    //             printf("\n");
    //             printf("theta_1 is %f ", theta_1);
    //             printf("phi_1 is %f ", phi_1);
    //             printf("\n");
    //             printf("face_1 at %d is %d \n", i, face_1);
    //         }

    //         path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list, full_iteration);
    //         if (test_mod)
    //         {
    //             printf("path_1111 at %d is good  \n", i);
    //             printf("face_2 is  ");
    //             printf("%d \n", face_2);
    //             printf("theta is %f ", theta);
    //             printf("phi is %f \n", phi);
    //         }
    //         // printf("face_2 at %d is %d \n",i, face_2);

    //         path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);
    //         if (test_mod)
    //         {
    //             printf("path_2222 at %d is good  \n", i);
    //         }

    //         // printf("Length of classes in ray tracing : %d \n", path_1.len_classes);
    //         // printf("Length of classes_posi in ray tracing: %d \n", path_1.len_classes_posi);
    //         // printArray(path_1.ray, 30);
    //         // printArray(path_1.posi, path_1.len_classes_posi);
    //         // printArray(path_1.classes, path_1.len_classes);

    //         numbers_1 = cal_path2_plus(path_1, voxel_size);
    //         numbers_2 = cal_path2_plus(path_2, voxel_size);
    //         // printArrayD(numbers_1, 4);
    //         // printArrayD(numbers_2, 4);

    //         absorption = cal_rate(numbers_1, numbers_2, coefficients, 1);

    //         if (test_mod)
    //         {
    //             printf("numbers_1 is  ");
    //             printArrayD(numbers_1, 4);
    //             printf("numbers_2 is");
    //             printArrayD(numbers_2, 4);
    //             printf("absorption is %f at %d \n", absorption, i);
    //             printf("\n");
    //             // if (i <10)
    //             // {

    //             // }
    //         }
    //         // if (i >3 ){
    //         //         free(path_1.ray);
    //         //         free(path_1.classes);
    //         //         free(path_1.posi);
    //         //         free(numbers_1);
    //         //         free(path_2.ray);
    //         //         free(path_2.classes);
    //         //         free(path_2.posi);
    //         //         free(numbers_2);
    //         //         break;
    //         // }
    //         absorption_sum += absorption;
    //         free(path_1.ray);
    //         free(path_1.classes);
    //         free(path_1.posi);

    //         free(numbers_1);
    //         free(path_2.ray);
    //         free(path_2.classes);
    //         free(path_2.posi);
    //         free(numbers_2);
    //         // printf("path_1 is \n");
    //         // printArrayD(numbers_1, 4);
    //         // printf("path_2 is \n");
    //         // printArrayD(numbers_2, 4);
    //         // printf("absorption is %.10lf \n",absorption);
    //         // printf("Length of classes in ray tracing : %d \n", path_1.len_classes);
    //         // printf("Length of classes_posi in ray tracing: %d \n", path_1.len_classes_posi);
    //         // on test, this is a top face,wich
    //         // path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);
    //     }
    //     absorption_mean = absorption_sum / len_coord_list;
    //     //         if (test_mod)
    //     //     {
    //     // printf("absorption_sum is %f \n", absorption_sum);
    //     // printf("len_coordinate_list is %f \n", len_coordinate_list);}
    //     // printf("finish \n");
    //     // free(path_1.ray);
    //     // free(path_1.classes);
    //     // free(path_1.posi);
    //     // free(numbers_1);
    //     // free(path_2.ray);
    //     // free(path_2.classes);
    //     // free(path_2.posi);
    //     // free(numbers_2);
    //     return absorption_mean;
    // }

    float ray_tracing_sampling(
        int *coord_list,
        int len_coord_list,
        float *rotated_s1, float *xray,
        float *voxel_size, float *coefficients,
        int8_t ***label_list, int8_t *label_list_1d, int *shape, int full_iteration,
        int store_paths)
    {
        printf("\n------------------ Ray tracing sampling --------------\n");
        printf("--------------> GPU version\n");
        int *h_face, *h_ray_classes;
        float *h_angles, *h_absorption;
        int z_max = shape[0], y_max = shape[1], x_max = shape[2];
        int diagonal = x_max * sqrt(3);
        int face_size = len_coord_list * 2 * sizeof(int);
        int absorption_size = len_coord_list * 2 * sizeof(float);
        int ray_classes_size = diagonal * len_coord_list * 2 * sizeof(int);
        int angle_size = 4 * sizeof(float);

        h_face = (int *)malloc(face_size);
        h_ray_classes = (int *)malloc(ray_classes_size);
        h_angles = (float *)malloc(angle_size);
        h_absorption = (float *)malloc(absorption_size);
        // ray_tracing_path(h_face, h_angles, h_ray_classes, h_absorption, coord_list, len_coord_list, rotated_s1, xray, voxel_size, coefficients, label_list_1d, shape);
        // printf("----> GPU : rayclass of the first voxel in for scattering is;\n");
        // for (int i=0;i<diagonal;i++){
        //     printf(" %d", h_ray_classes[i]);
        // }
        printf("----> GPU version FINISHED;\n");

        printf("-----> Testing label_list_1d: ");
        int errors_in_label_list = compare_voxels(label_list, label_list_1d, shape);
        if (errors_in_label_list == 0)
            printf("PASSED\n");
        else
            printf("FAILED\n");

        float x_ray_angle[3], rotated_s1_angle[3];
        float xray_direction[3], scattered_direction[3];
        float xray_switched[3], scattered_switched[3];
        dials_2_numpy(xray, xray_direction);
        // printf("Od order: [%f ; %f ; %f] New: [%f ; %f ; %f]\n", xray[0], xray[1], xray[2], xray_direction[0], xray_direction[1], xray_direction[2]);
        dials_2_numpy(rotated_s1, scattered_direction);
        memcpy(x_ray_angle, xray, 3 * sizeof(float));
        memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(float));

        ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
        ThetaPhi result_xray = dials_2_thetaphi_22(x_ray_angle, 1);
        float theta = result_2.theta;
        float phi = result_2.phi;
        float theta_xray = result_xray.theta;
        float phi_xray = result_xray.phi;

        // printf("rotated_s1 angles: theta: CPU=%f; GPU=%f diff=%f|| phi: CPU=%f; GPU=%f; diff=%f\n", theta, h_angles[0], abs(theta - h_angles[0]), phi, h_angles[1], abs(phi - h_angles[1]));
        // printf("xray angles: theta: CPU=%f; GPU=%f diff=%f|| phi: CPU=%f; GPU=%f; diff=%f\n", theta_xray, h_angles[2], abs(theta_xray - h_angles[2]), phi_xray, h_angles[3], abs(phi_xray - h_angles[3]));

        printf("-------> Increment test:\n");
        float ix, iy, iz;
        printf("Xray:\n");
        for (int f = 1; f <= 6; f++)
        {
            get_increment_ratio(&ix, &iy, &iz, theta_xray, phi_xray, f);
            printf("==> CPU: face=%d; i=[%f; %f; %f];\n", f, ix, iy, iz);
        }
        printf("rotated_s1:\n");
        for (int f = 1; f <= 6; f++)
        {
            get_increment_ratio(&ix, &iy, &iz, theta, phi, f);
            printf("==> CPU: face=%d; i=[%f; %f; %f];\n", f, ix, iy, iz);
        }
        printf("-------------------------------<\n");

        Path2_c path_2, path_1;
        Path2_c path_2_ref, path_1_ref;
        float *numbers_1_ref, *numbers_2_ref;
        float *numbers_1, *numbers_2;
        float absorption;
        float absorption_sum = 0, absorption_mean = 0;

        int nFaceErrors = 0;
        int nAbsorptionErrors = 0;
        int nClassesErrors = 0, path2error = 0, path1error = 0;
        float absorption_1, absorption_2;
        for (int i = 0; i < len_coord_list; i++)
        {

            int coord[3] = {coord_list[i * 3],
                                coord_list[i * 3 + 1],
                                coord_list[i * 3 + 2]};

            int face_1_ref = which_face(coord, shape, theta_xray, phi_xray);
            int face_2_ref = which_face(coord, shape, theta, phi);
            int face_1 = cube_face(coord, xray_direction, shape, 1);
            int face_2 = cube_face(coord, scattered_direction, shape, 0);
            if (((int)face_1) != (h_face[2 * i + 1]) || ((int)face_2) != h_face[2 * i + 0])
            {
                if (i < 32)
                    printf("face1: CPU=%d; GPU=%d || face2: CPU=%d; GPU=%d;\n", (int)face_1, h_face[2 * i + 1], (int)face_2, h_face[2 * i + 0]);
                nFaceErrors++;
            }

            int errors = 0;
            path_1_ref = cal_coord_ref(theta_xray, phi_xray, coord, face_1, shape, label_list, full_iteration);
            path_1 = cal_coord(theta_xray, phi_xray, coord, face_1, shape, label_list_1d, full_iteration);
            if (DEBUG)
            {
                errors = compare_Path2s(&path_1, &path_1_ref);
            }
            if (errors > 0)
                printf("Comparing path_1: FAILED\n");

            path_2_ref = cal_coord_ref(theta, phi, coord, face_2, shape, label_list, full_iteration);
            path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list_1d, full_iteration);
            if (DEBUG)
            {
                compare_Path2s(&path_2, &path_2_ref);
                errors = compare_Path2s(&path_1, &path_1_ref);
            }
            if (errors > 0)
                printf("Comparing path_2: FAILED\n");

            // for is_ray_incoming = 0;
            // this means even for GPU and path_2
            for (int f = 0; f < path_2.len_path_2; f++)
            {
                int CPU_class = (int)path_2.ray_classes[f];
                int GPU_pos = (2 * i + 0) * diagonal + f;
                int GPU_class = h_ray_classes[GPU_pos];
                if (CPU_class != GPU_class)
                {
                    path2error++;
                }
            }

            // if(i==0){
            //	for(int f=0; f<path_2.len_path_2+10; f++){
            //		int CPU_class = 0;
            //		if(f<path_2.len_path_2){
            //			CPU_class = (int) path_2.ray_classes[f];
            //		}
            //		int GPU_pos = (2*i + 0)*diagonal + f;
            //		int GPU_class = h_ray_classes[GPU_pos];
            //		printf("[%d ; %d] ", (int) CPU_class, (int) GPU_class);
            //	}
            //	printf("\n");
            // }

            if (path2error > 0)
            {
                // for(int f=0; f<path_2.len_path_2+10; f++){
                //	int CPU_class = 0;
                //	if(f<path_2.len_path_2){
                //		CPU_class = (int) path_2.ray_classes[f];
                //	}
                //	int GPU_pos = (2*i + 0)*diagonal + f;
                //	int GPU_class = h_ray_classes[GPU_pos];
                //	printf("[%d ; %d] ", (int) CPU_class, (int) GPU_class);
                // }

                nClassesErrors++;
                // printf("\n");
            }

            // for is_ray_incoming = 1;
            // this means even for GPU and path_1
            // printf("Incoming=1; ");
            for (int f = 0; f < path_1.len_path_2; f++)
            {
                int CPU_class = (int)path_1.ray_classes[f];
                int GPU_pos = (2 * i + 1) * diagonal + f;
                int GPU_class = h_ray_classes[GPU_pos];
                // printf("[%d ; %d] ", (int) CPU_class, (int) GPU_class);
                if (CPU_class != GPU_class)
                {
                    path1error++;
                }
            }
            // printf("\n");

            if (path1error > 0)
            {
                // for(int f=0; f<path_2.len_path_2+10; f++){
                //	int CPU_class = 0;
                //	if(f<path_2.len_path_2){
                //		CPU_class = (int) path_2.ray_classes[f];
                //	}
                //	int GPU_pos = (2*i + 0)*diagonal + f;
                //	int GPU_class = h_ray_classes[GPU_pos];
                //	printf("[%d ; %d] ", (int) CPU_class, (int) GPU_class);
                // }

                nClassesErrors++;
                // printf("\n");
            }

            numbers_1_ref = cal_path2_plus_ref(path_1, voxel_size);
            numbers_1 = cal_path2_plus(path_1, voxel_size);
            compare_classes_lengths(numbers_1, numbers_1_ref);

            numbers_2_ref = cal_path2_plus_ref(path_2, voxel_size);
            numbers_2 = cal_path2_plus(path_2, voxel_size);
            compare_classes_lengths(numbers_2, numbers_2_ref);

            // if (i<1){
            //     // test_ray_classes(path_1, coord, h_ray_classes, diagonal);
            //     printf("numbers_1:%f, %f, %f, %f \n",numbers_1[0]*coefficients[0],numbers_1[1]*coefficients[1],numbers_1[2]*coefficients[2],numbers_1[3]*coefficients[3]);
            //     printf("numbers_2:%f, %f, %f, %f \n",numbers_2[0]*coefficients[0],numbers_2[1]*coefficients[1],numbers_2[2]*coefficients[2],numbers_2[3]*coefficients[3]);
            //     printf("numbers_1:%f, %f, %f, %f \n",numbers_1[0],numbers_1[1],numbers_1[2],numbers_1[3]);
            //     printf("numbers_2:%f, %f, %f, %f \n",numbers_2[0],numbers_2[1],numbers_2[2],numbers_2[3]);
            //     printf(" path_2.coordinte s is \n");
            //     for(int k=0; k<path_2.len_path_2; k++){
            //         printf(" [%d,%d,%d] ", path_2.ray[k*3+0], path_2.ray[k*3+1], path_2.ray[k*3+2]);
            //     }
            //     printf ("\n ");
            //     printf(" path_2.ray_classes is \n");
            //     for(int k=0; k<path_2.len_path_2; k++){
            //         printf("%d ", path_2.ray_classes[k]);
            //     }
            //     printf ("\n ");
            // }
            // if (i>1){
            //     break;
            // }

            // absorption = cal_rate(numbers_1, numbers_2, coefficients, 1);
            absorption_1 = cal_rate_single(numbers_1, coefficients, 0);
            absorption_2 = cal_rate_single(numbers_2, coefficients, 0);

            absorption = exp(-(absorption_1 + absorption_2));

            // printf("i = %d; CPU = %f; GPU = %f; diff = %f;\n", (int) i, absorption, exp(-(h_absorption[2*i+0] + h_absorption[2*i+1])), abs(absorption - (exp(-(h_absorption[2*i+0] + h_absorption[2*i+1])))) );
            // printf("i = %d;path_1; face_c=%d, face_g = %d;CPU = %f; GPU = %f; diff = %f;\n", (int) i, face_1,h_face[2*i+1],absorption_1, h_absorption[2*i+1] , abs(absorption_1 -h_absorption[2*i+1] ) );
            // printf("i = %d;path_2; face_c=%d, face_g = %d;CPU = %f; GPU = %f; diff = %f;\n", (int) i,face_2, h_face[2*i+0],absorption_2, h_absorption[2*i+0] , abs(absorption_2 -h_absorption[2*i+0] ) );
            absorption_sum += absorption;

            free(path_1.ray);
            free(path_1.ray_classes);
            free(path_1.classes);
            free(path_1.posi);
            // free(path_1);
            free(numbers_1_ref);
            free(numbers_1);
            free(path_2.ray);
            free(path_2.ray_classes);
            free(path_2.classes);
            free(path_2.posi);
            // free(path_2);
            free(numbers_2_ref);
            free(numbers_2);

            free(path_1_ref.ray);
            free(path_1_ref.ray_classes);
            free(path_1_ref.classes);
            free(path_1_ref.posi);

            free(path_2_ref.ray);
            free(path_2_ref.ray_classes);
            free(path_2_ref.classes);
            free(path_2_ref.posi);

            path2error = 0;
            path1error = 0;
        }
        absorption_mean = absorption_sum / len_coord_list;
        int nAngleErrors = 0;
        if (abs(theta - h_angles[0]) > 1.0e-4)
            nAngleErrors++;
        if (abs(phi - h_angles[1]) > 1.0e-4)
            nAngleErrors++;
        if (abs(theta_xray - h_angles[2]) > 1.0e-4)
            nAngleErrors++;
        if (abs(phi_xray - h_angles[3]) > 1.0e-4)
            nAngleErrors++;
        // printf("rotated_s1 angles: theta: CPU=%f; GPU=%f diff=%f|| phi: CPU=%f; GPU=%f; diff=%f\n", theta, h_angles[0], abs(theta - h_angles[0]), phi, h_angles[1], abs(phi - h_angles[1]));
        // printf("xray angles: theta: CPU=%f; GPU=%f diff=%f|| phi: CPU=%f; GPU=%f; diff=%f\n", theta_xray, h_angles[2], abs(theta_xray - h_angles[2]), phi_xray, h_angles[3], abs(phi_xray - h_angles[3]));

        float gpu_absorption = 0;
        for (int i = 0; i < len_coord_list; i++)
        {
            gpu_absorption += exp(-(h_absorption[2 * i + 0] + h_absorption[2 * i + 1]));
        }
        float gpu_absorption_mean = gpu_absorption / ((float)len_coord_list);
        printf("CPU mean absorption: %f; GPU mean absorption: %f;\n", absorption_mean, gpu_absorption_mean);

        printf("--> Number of angle errors: %d;\n", nAngleErrors);
        printf("--> Number of face errors: %d;\n", nFaceErrors);
        printf("--> Number of class errors: %d;\n", nClassesErrors);

        free(h_face);
        free(h_ray_classes);
        free(h_angles);
        free(h_absorption);

        return absorption_mean;
    }

    float *ray_tracing_gpu_overall(int32_t low, int32_t up,
                                    int *coord_list,
                                    int32_t len_coord_list,
                                    const float *scattering_vector_list, const float *omega_list,
                                    const float *raw_xray,
                                    const float *omega_axis, const float *kp_rotation_matrix,
                                    int32_t len_result,
                                    float *voxel_size, float *coefficients, int8_t ***label_list,
                                    int8_t *label_list_1d, int *shape, int32_t full_iteration,
                                    int32_t store_paths)
    {

        // float *result_list = malloc( len_result* sizeof(float));
        // size_t len_result_float = (int32_t) len_result* sizeof(float);
        // int32_t len_result_float = (int32_t) len_result;
        printf("low is %d \n", low);
        printf("up is %d \n", up);
        float factor = 1;
        // len_result = (int)(len_result*factor);
        printf("len_result is %d \n", len_result);
        float *h_result_list = (float *)malloc(len_result * len_coord_list * 2 * sizeof(float));
        float *h_python_result_list = (float *)malloc(len_result * sizeof(float));
        int *h_face = (int *)malloc(len_coord_list * 2 * sizeof(int));
        float *h_angles = (float *)malloc(4 * sizeof(float));

        ray_tracing_gpu_overall_kernel(low, up, coord_list, len_coord_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, len_result, voxel_size, coefficients, label_list_1d, shape, full_iteration, store_paths, h_result_list, h_face, h_angles, h_python_result_list);
        // printf("h_angles is [%f, %f, %f, %f]\n", h_angles[0], h_angles[1], h_angles[2], h_angles[3]);
        // for (int i = 0; i < len_result; i++)
        // {
        //     if (i==100){
        //         break;
        //     }
        //     // float gpu_absorption = 0;
        //     // for (int j = 0; j < len_coord_list; j++)
        //     // {
        //     //     gpu_absorption += exp(-(h_result_list[2*i * len_coord_list + 2 * j + 0] + h_result_list[2*i * len_coord_list + 2 * j + 1]));

        //     // }

        //     // float gpu_absorption_mean = gpu_absorption / ((float)len_coord_list);
        //     float gpu_absorption_mean = h_python_result_list[i];
        //     printf("\n");
        //     printf("GPU mean absorption in cpu code: %f;\n", gpu_absorption_mean);

        //     // continue;
        //     float result;
        //     float rotation_matrix_frame_omega[9];
        //     float rotation_matrix[9];
        //     float total_rotation_matrix[9];
        //     float xray[3];
        //     float rotated_s1[3];
        //     // printf("kap roation  \n");
        //     kp_rotation(omega_axis, omega_list[i], (float *)rotation_matrix_frame_omega);
        //     // printf("rotation_matrix_frame_omega is \n");
        //     // print_matrix((float*)rotation_matrix_frame_omega,3,3);
        //     dot_product((float *)rotation_matrix_frame_omega, kp_rotation_matrix, (float *)rotation_matrix, 3, 3, 3);

        //     transpose((float *)rotation_matrix, 3, 3, (float *)total_rotation_matrix);
        //     // printf("total_rotation_matrix is \n");
        //     // print_matrix((float*)total_rotation_matrix,3,3);

        //     // printf("xray is \n");
        //     // print_matrix(raw_xray,1,3);
        //     dot_product((float *)total_rotation_matrix, raw_xray, (float *)xray, 3, 3, 1);
        //     // printf("xray is \n");
        //     // print_matrix(xray,1,3);
        //     float scattering_vector[3] = {scattering_vector_list[i * 3],
        //                                    scattering_vector_list[i * 3 + 1],
        //                                    scattering_vector_list[i * 3 + 2]};
        //     dot_product((float *)total_rotation_matrix, (float *)scattering_vector, (float *)rotated_s1, 3, 3, 1);

        //     result = ray_tracing_sampling_od(
        //         coord_list, len_coord_list,
        //         (float *)rotated_s1, (float *)xray,
        //         voxel_size, coefficients,
        //         label_list_1d,  shape, full_iteration,
        //         store_paths);
        //     // result = ray_tracing_sampling(
        //     //     coord_list, len_coord_list,
        //     //     (float *)rotated_s1, (float *)xray,
        //     //     voxel_size, coefficients,label_list,
        //     //     label_list_1d,  shape, full_iteration,
        //     //     store_paths);
        //     printf("result is %f \n",result);

        //     printf("[%d/%d] rotation: %.4f, absorption: %.4f\n",
        //            low + i, up, omega_list[i] * 180 / M_PI, result);
        //     printf("gpu_absorption_mean is %f\n", gpu_absorption_mean);
        //     printf("difference is %f\n", (gpu_absorption_mean - result) / result * 100);
        // }

        // printf("len_result_float is %d \n", len_result_float);
        // printf("result_list is %p \n", result_list);
        free(h_result_list);
        return h_python_result_list;
    }

#ifdef __cplusplus
}
#endif