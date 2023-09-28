#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#define M_PI 3.14159265358979323846
#include <stdint.h>
#include <sys/resource.h>
#include "ray_tracing.h"


ThetaPhi dials_2_thetaphi_22(double rotated_s1[3], int64_t L1)
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

void dials_2_numpy(double vector[3], double result[3])
{
    double numpy_2_dials_1[3][3] = {
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

int64_t cube_face(int64_t ray_origin[3], double ray_direction[3], int64_t cube_size[3], int L1)
{
        // deciding which plane to go out, to see which direction (xyz) has increment of 1
    /*  'FRONTZY' = 1;
*   'LEYX' = 2 ;
*   'RIYX' = 3;
    'TOPZX' = 4;
    'BOTZX' = 5;
    "BACKZY" = 6 ;

*/  
    int64_t min_x = 0;
    int64_t max_x = cube_size[2];
    int64_t min_y = 0;
    int64_t max_y = cube_size[1];
    int64_t min_z = 0;
    int64_t max_z = cube_size[0];
    

    double tx_min = (min_x - ray_origin[2]) / ray_direction[2];
    double tx_max = (max_x - ray_origin[2]) / ray_direction[2];
    double ty_min = (min_y - ray_origin[1]) / ray_direction[1];
    double ty_max = (max_y - ray_origin[1]) / ray_direction[1];
    double tz_min = (min_z - ray_origin[0]) / ray_direction[0];
    double tz_max = (max_z - ray_origin[0]) / ray_direction[0];

    if (L1){
        tx_min = -tx_min;
        tx_max = -tx_max;
        ty_min = -ty_min;
        ty_max = -ty_max;
        tz_min = -tz_min;
        tz_max = -tz_max;
    }

    double t_numbers[6] = {tx_min, ty_min, tz_min, tx_max, ty_max, tz_max};
    int t_numbers_len = sizeof(t_numbers) / sizeof(t_numbers[0]);

    double non_negative_numbers[t_numbers_len];
    int non_negative_len = 0;
    for (int i = 0; i < t_numbers_len; i++)
    {
        if (t_numbers[i] >= 0)
        {
            non_negative_numbers[non_negative_len++] = t_numbers[i];
        }
    }

    double t_min = non_negative_numbers[0];
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
        return  6;
    }
    else if (t_min == tx_max)
    {
        return  1;
    }
    else if (t_min == ty_min)
    {
        return  4;
    }
    else if (t_min == ty_max)
    {
        return  5;
    }
    else if (t_min == tz_min)
    {
        return  2;
    }
    else if (t_min == tz_max)
    {
        return  3;
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

int64_t which_face(int64_t coord[3], int64_t shape[3], double theta, double phi)
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
     * coord: the point64_t which was calculated the ray length
     * shape: shape of the tomography matrix
     * theta: calculated theta angle to the point64_t on the detector, positive means rotate clockwisely, vice versa
     * phi: calculated phi angle to the point64_t on the detector, positive means rotate clockwisely
     * return: which face of the ray to exit, that represents the which (x,y,z) increment is 1
     *
     * top front left is the origin, not bottom front left
     */
    // the detector and the x-ray anti-clockwise rotation is positive
    double z_max = shape[0] - 1;
    double y_max = shape[1] - 1;
    double x_max = shape[2] - 1;
    double x = coord[2];
    double y = coord[1];
    double z = coord[0];
    // if (test_mod)
    // {
    //     struct rusage usage;
    //     getrusage(RUSAGE_SELF, &usage);
    //     printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    // }
    if (fabs(theta) < M_PI / 2)
    {
        double theta_up = atan((y - 0) / (x - 0 + 0.001));
        double theta_down = -atan((y_max - y) / (x - 0 + 0.001)); // negative
        double phi_right = atan((z_max - z) / (x - 0 + 0.001));
        double phi_left = -atan((z - 0) / (x - 0 + 0.001)); // negative
        double omega = atan(tan(theta) * cos(phi));

        if (omega > theta_up)
        {
            // at this case, theta is positive,
            // normally the most cases for theta > theta_up, the ray passes the top ZX plane
            // if the phis are smaller than both edge limits
            // the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            double side = (y - 0) * sin(fabs(phi)) / tan(theta); // the length of rotation is the projected length on x
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
            double side = (y_max - y) * sin(fabs(phi)) / tan(-theta);
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
        double theta_up = atan((y - 0) / (x_max - x + 0.001));
        double theta_down = atan((y_max - y) / (x_max - x + 0.001)); // negative
        double phi_left = atan((z_max - z) / (x_max - x + 0.001));   // it is the reverse of the top phi_left
        double phi_right = -atan((z - 0) / (x_max - x + 0.001));     // negative
        //
        //
        if ((M_PI - theta) > theta_up && theta > 0)
        {
            // at this case, theta is positive,
            // normally the most cases for theta > theta_up, the ray passes the top ZX plane
            // if the phis are smaller than both edge limits
            // the ray only goes through right/left plane when the  reflection coordinate is too close to the  right/left plane
            double side = (y - 0) * sin(fabs(phi)) / fabs(tan(theta));
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
            double side = (y_max - y) * sin(fabs(phi)) / fabs(tan(-theta));
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

void appending(int64_t increment, int64_t *path_2,
               int64_t *classes, int64_t *classes_posi,
               int64_t *potential_coord,
               int64_t label, int64_t previous_label,
               int64_t *len_classes, int64_t *len_classes_posi,
               int64_t *len_path_2)
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

Path2_c cal_coord(double theta, double phi, int64_t *coord, int64_t face,
                  int64_t *shape, int8_t ***label_list, int64_t full_iteration)
{
    Path2_c result;
    int64_t z = coord[0], y = coord[1], x = coord[2];
    int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int64_t diagonal = x_max * sqrt(3);

    int64_t *path_2 = (int64_t *)malloc(diagonal * 3 * sizeof(int64_t));
    int64_t *classes_posi = (int64_t *)malloc(diagonal * sizeof(int64_t));
    int64_t *classes = (int64_t *)malloc(diagonal * sizeof(int64_t));
    classes[0] = 3;
    classes_posi[0] = 0;
    // int64_t path_2[x_max*y_max*z_max][3];
    // int64_t classes_posi[x_max*y_max*z_max];
    // int64_t classes[x_max*y_max*z_max];
    double increment_ratio_x, increment_ratio_y, increment_ratio_z;

    int64_t len_path_2 = 1;
    int64_t len_classes = 1;
    int64_t len_classes_posi = 1;
    int64_t new_z, new_y, new_x;
    if (face == 6)
    {
        assert(fabs(theta) <= M_PI / 2);
        increment_ratio_x = -1;
        increment_ratio_y = tan(theta) / cos(phi);
        increment_ratio_z = tan(phi);

        for (int64_t increment = 0; increment <= x - 0; increment++)
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
            int64_t potential_coord[3] = {new_z, new_y, new_x};

            int64_t label = label_list[new_z][new_y][new_x];

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
            //     int64_t label = label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]];
            int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                        path_2[(increment - 1) * 3 + 1],
                                        path_2[(increment - 1) * 3 + 2]};
            int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];
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
            assert (phi < 0);
            double increment_ratio_x = 1 / tan(fabs(phi));
            double increment_ratio_y = tan(theta) / sin(fabs(phi));
            double increment_ratio_z = -1;

            for (int64_t increment = 0; increment <= z; increment++)
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
                // printf("new_x: %ld, new_y: %ld, new_z: %ld \n", new_x, new_y, new_z);
                // if (test_mod)
                // {
                //     printf("new_x: %ld, new_y: %ld, new_z: %ld \n", new_x, new_y, new_z);
                // }
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
                // if (test_mod){
                //     printArray(potential_coord, 3);
                //     printf("label: %ld \n", label);

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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
        else
        {
            assert (phi > 0);
            double increment_ratio_x = 1 / tan(fabs(phi));
            double increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            double increment_ratio_z = -1;

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
                //     printf("new_x: %ld, new_y: %ld, new_z: %ld \n", new_x, new_y, new_z);
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

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
            assert (phi > 0);
            double increment_ratio_x = 1 / tan(fabs(phi));
            double increment_ratio_y = tan(theta) / sin(fabs(phi));
            double increment_ratio_z = 1;
            for (int64_t increment = 0; increment < (z_max - z); increment++)
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

                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }

        else
        {
            assert (phi<0);
            double increment_ratio_x = 1 / (tan(fabs(phi)));
            double increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
            double increment_ratio_z = 1;
            for (int64_t increment = 0; increment <= (z_max - z); increment++)
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

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
           assert(theta > 0);
        if (fabs(theta) < M_PI / 2)
        {
            double increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
            double increment_ratio_y = 1;
            double increment_ratio_z = sin(phi) / tan(fabs(theta));

            for (int64_t increment = 0; increment <= y - 0; increment++)
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

                appending(increment, path_2,
                          classes, classes_posi,
                          potential_coord,
                          label, previous_label,
                          &len_classes, &len_classes_posi, &len_path_2);
            }
        }
        else
        {
            double increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
            double increment_ratio_y = 1;
            double increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
            for (int64_t increment = 0; increment < y - 0 + 1; increment++)
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

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
        assert (theta<0);
        if (fabs(theta) < M_PI / 2)
        {
            
            double increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
            double increment_ratio_y = -1;
            double increment_ratio_z = sin(phi) / (tan(fabs(theta)));
            // printf("increment_ratio_x: %f, increment_ratio_y: %f, increment_ratio_z: %f \n", increment_ratio_x, increment_ratio_y, increment_ratio_z);
            // printArray(coord, 3);
            for (int64_t increment = 0; increment < y_max - y; increment++)
            {
                // decrement on y-axis
                // new_x = x + -1 * increment * np.cos(np.abs(phi))/(np.tan(np.abs(theta)))
                // new_y = y - increment*-1
                // new_z = z + increment*np.sin(phi)/ ( np.tan(np.abs(theta)) )
                new_x = round(x + -1 * increment * increment_ratio_x);
                new_y = round(y - increment * increment_ratio_y);
                new_z = round(z + increment * increment_ratio_z);
                // printf("increment %d", increment);
                // printf("new_x: %ld, new_y: %ld, new_z: %ld \n", new_x, new_y, new_z);
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

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
            for (int64_t increment = 0; increment <= y_max - y; increment++)
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
                int64_t potential_coord[3] = {new_z, new_y, new_x};
                int64_t label = label_list[new_z][new_y][new_x];
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
                int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                            path_2[(increment - 1) * 3 + 1],
                                            path_2[(increment - 1) * 3 + 2]};
                int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];

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

        
        assert(fabs(theta) >= M_PI / 2 );
        increment_ratio_x = -1;
        increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
        increment_ratio_z = tan(phi);

        for (int64_t increment = 0; increment < x_max - x; increment++)
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
            int64_t potential_coord[3] = {new_z, new_y, new_x};
            int64_t label = label_list[new_z][new_y][new_x];
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
            //     int64_t label = label_list[potential_coord[0] * y_max * x_max + potential_coord[1] * x_max + potential_coord[2]];
            int64_t previous_step[3] = {path_2[(increment - 1) * 3],
                                        path_2[(increment - 1) * 3 + 1],
                                        path_2[(increment - 1) * 3 + 2]};
            int64_t previous_label = label_list[previous_step[0]][previous_step[1]][previous_step[2]];
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

    result.posi =(int64_t*) realloc(classes_posi, len_classes_posi * sizeof(int64_t));
    result.classes = (int64_t*)realloc(classes, len_classes * sizeof(int64_t));
    result.ray = (int64_t*)realloc(path_2, len_path_2 * 3 * sizeof(int64_t));
    // result.posi = malloc(len_classes_posi * sizeof(int64_t));
    // result.classes = malloc(len_classes * sizeof(int64_t));
    // result.ray = malloc(len_path_2 * 3 * sizeof(int64_t));
    // for (int64_t i = 0; i < len_path_2 * 3; i++)
    // {
    //     result.ray[i] = path_2[i];
    // }
    // for (int64_t i = 0; i < len_classes_posi; i++)
    // {
    //     // printf("classes_posi is %d \n", classes_posi[i]);
    //     result.posi[i] = classes_posi[i];
    // }
    // for (int64_t i = 0; i < len_classes; i++)
    // {
    //     result.classes[i] = classes[i];
    // }
    // // printArray(result.ray, 30);
    // // printArray(result.posi, result.len_classes_posi);
    // // printArray(result.classes, result.len_classes);
    // if (test_mod)
    // {
    //         struct rusage usage;
    // getrusage(RUSAGE_SELF, &usage);
    // printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    // }

    // if (test_mod)
    // {

    // printf( "diagonal is %d \n", diagonal);
    // printf("len_path_2 is %d \n", len_path_2);
    // printf("len_classes is %d \n", len_classes);
    // printf("len_classes_posi is %d \n", len_classes_posi);
    // // printArray(path_2, len_path_2*3);
    // printArray(classes_posi, len_classes_posi);
    // printArray(classes, len_classes);
    // }

    // free(path_2);
    // if (test_mod)
    // {
    // printf("path_2 is free \n");

    // }

    // free(classes_posi);
    // if (test_mod)
    // {
    // printf("classes_posi is free \n");

    // }

    // free(classes);
    // if (test_mod)
    // {
    // printf(" class is free \n");

    // }
    // malloc_trim(0);
    return result;
}

double *cal_path2_plus(Path2_c path_2_cal_result, double *voxel_size)
{
    double *result = (double *)malloc(4 * sizeof(double));
    double voxel_length_z = voxel_size[0];
    double voxel_length_y = voxel_size[1];
    double voxel_length_x = voxel_size[2];
    int64_t *path_ray = path_2_cal_result.ray;
    int64_t *posi = path_2_cal_result.posi;
    int64_t *classes = path_2_cal_result.classes;
    int64_t len_path_2 = path_2_cal_result.len_path_2;
    int64_t len_classes = path_2_cal_result.len_classes;
    int64_t len_classes_posi = path_2_cal_result.len_classes_posi;

    double cr_l_2 = 0;
    double li_l_2 = 0;
    double bu_l_2 = 0;
    double lo_l_2 = 0;

    double total_length = sqrt(pow((path_ray[(len_path_2 - 1) * 3 + 1] - path_ray[1]) * voxel_length_y, 2) +
                               pow((path_ray[(len_path_2 - 1) * 3 + 0] - path_ray[0]) * voxel_length_z, 2) +
                               pow((path_ray[(len_path_2 - 1) * 3 + 2] - path_ray[2]) * voxel_length_x, 2));

    for (int j = 0; j < len_classes_posi; j++)
    {
        if (classes[j] == 3)
        {
            if (j < len_classes_posi - 1)
            {
                cr_l_2 += total_length * ((double)(posi[j + 1] - posi[j]) / (double)len_path_2);
            }
            else
            {
                cr_l_2 += total_length * ((double)(len_path_2 - posi[j]) / (double)len_path_2);
            }
        }
        else if (classes[j] == 1)
        {
            if (j < len_classes_posi - 1)
            {
                li_l_2 += total_length * ((double)(posi[j + 1] - posi[j]) / (double)len_path_2);
            }
            else
            {
                li_l_2 += total_length * ((double)(len_path_2 - posi[j]) / (double)len_path_2);
            }
        }
        else if (classes[j] == 2)
        {
            if (j < len_classes_posi - 1)
            {
                lo_l_2 += total_length * ((double)(posi[j + 1] - posi[j]) / (double)len_path_2);
            }
            else
            {
                lo_l_2 += total_length * ((double)(len_path_2 - posi[j]) / (double)len_path_2);
            }
        }
        else if (classes[j] == 4)
        {
            if (j < len_classes_posi - 1)
            {
                bu_l_2 += total_length * ((double)(posi[j + 1] - posi[j]) / (double)len_path_2);
            }
            else
            {
                bu_l_2 += total_length * ((double)(len_path_2 - posi[j]) / (double)len_path_2);
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

double cal_rate(double *numbers_1, double *numbers_2, double *coefficients,
                char Isexp)
{

    double mu_li = coefficients[0];
    double mu_lo = coefficients[1];
    double mu_cr = coefficients[2];
    double mu_bu = coefficients[3];

    double li_l_1 = numbers_1[0];
    double lo_l_1 = numbers_1[1];
    double cr_l_1 = numbers_1[2];
    double bu_l_1 = numbers_1[3];

    double li_l_2 = numbers_2[0];
    double lo_l_2 = numbers_2[1];
    double cr_l_2 = numbers_2[2];
    double bu_l_2 = numbers_2[3];

    double result = (mu_li * (li_l_1 + li_l_2) +
                     mu_lo * (lo_l_1 + lo_l_2) +
                     mu_cr * (cr_l_1 + cr_l_2) +
                     mu_bu * (bu_l_1 + bu_l_2));

    if (Isexp == 1)
    {
        result = exp(-result);
    }

    return result;
}
