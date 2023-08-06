// #define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "unit_test.h"
#include "ray_tracing.h"
#include <sys/resource.h>
#define M_PI 3.14159265
#define test_mod 0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)

// TODO: 
// Use 1d data in ray calculation
// 


int64_t compare_Path2s(Path2_c *path, Path2_c *path_ref){
	int64_t total_errors = 0;
	if(path->len_path_2 != path_ref->len_path_2){
		printf("--> Comparing Path2_c: Wrong len_path_2 C:%ld; R:%ld;\n", path->len_path_2, path_ref->len_path_2);
		total_errors++;
	}
	if(path->len_classes_posi != path_ref->len_classes_posi){
		printf("--> Comparing Path2_c: Wrong len_classes_posi C:%ld; R:%ld;\n", path->len_classes_posi, path_ref->len_classes_posi);
		total_errors++;
	}
	if(path->len_classes != path_ref->len_classes){
		printf("--> Comparing Path2_c: Wrong len_classes C:%ld; R:%ld;\n", path->len_classes, path_ref->len_classes);
		total_errors++;
	}
	
	// Comparing ray coordinates
	int64_t ray_errors = 0;
	for(int64_t f = 0; f < (path->len_path_2)*3; f++){
		if(path->ray[f]!=path_ref->ray[f]) ray_errors++;
	}
	if(ray_errors>0) {
		printf("--> Comparing Path2_c: path->ray do not agree!\n");
		total_errors++;
	}
	
	// Comparing ray classes
	int64_t ray_classes_errors = 0;
	for(int64_t f = 0; f < (path->len_path_2); f++){
		if(path->ray_classes[f]!=path_ref->ray_classes[f]) ray_classes_errors++;
		//printf("%ld, ", path->ray_classes[f]);
		//if(f==(path->len_path_2)-1) printf("\n=\n");
	}
	if(ray_classes_errors>0) {
		printf("--> Comparing Path2_c: path->ray_classes do not agree!\n");
		total_errors++;
	}
	
	// Comparing position of the borders
	int64_t posi_errors = 0;
	for(int64_t f = 0; f < (path->len_classes_posi); f++){
		if(path->posi[f]!=path_ref->posi[f]) posi_errors++;
	}
	if(posi_errors>0) {
		printf("--> Comparing Path2_c: path->posi do not agree!\n");
		total_errors++;
	}
	
	// Comparing border labels
	int64_t classes_errors = 0;
	for(int64_t f = 0; f < (path->len_classes); f++){
		if(path->classes[f]!=path_ref->classes[f]) classes_errors++;
	}
	if(classes_errors>0) {
		printf("--> Comparing Path2_c: path->classes do not agree!\n");
		total_errors++;
	}
	
	return(total_errors);
}

int64_t compare_classes_lengths(double *lengths, double *lengths_ref){
	double max_difference = 1.0e-6;
	double total_difference = 0;
	total_difference += abs(lengths[0] - lengths_ref[0]);
	total_difference += abs(lengths[1] - lengths_ref[1]);
	total_difference += abs(lengths[2] - lengths_ref[2]);
	total_difference += abs(lengths[3] - lengths_ref[3]);
	if(total_difference>max_difference) {
		printf("--> Comparing classes lenghts: Error!\n");
		return 1;
	}
	else return 0;
}

int64_t compare_voxels(int8_t ***label_list, int8_t *label_list_1d, int64_t *shape){
	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
	int64_t nErrors = 0;
	for(int64_t z = 0; z < z_max; z++){
		for(int64_t y = 0; y < y_max; y++){
			for(int64_t x = 0; x < x_max; x++){
				int64_t pos = INDEX_3D(z_max, y_max, x_max, z, y, x);
				if(label_list_1d[pos]!=label_list[z][y][x]){
					printf("--> Comparing label lists: Error! %d!=%d;\n", (int) label_list_1d[pos], (int) label_list[z][y][x]);
					nErrors++;
				}
			}
		}
	}
	return nErrors;
}

typedef struct
{
    double theta;
    double phi;
} ThetaPhi;

typedef struct
{
    double li;
    double lo;
    double cr;
    double bu;
} classes_lengths;

// typedef struct {
//     Point64_t *path_ray;
//     int64_t *posi;
//     int64_t *classes;
// } Path2;

int64_t count_len(int64_t *arr)
{
    int64_t count = 0;
    while (*arr != '\0')
    {
        count++;
        arr++;
    }
    printf("Length of array: %ld\n", count);

    return count;
}

void printArray(int64_t arr[], int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%ld ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayshort(int64_t arr[], char n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%ld ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayD(double arr[], int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%.15lf ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

ThetaPhi dials_2_thetaphi_22(double rotated_s1[3], int64_t L1){
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
    if (test_mod)
    {
            struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    }
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

void dials_2_numpy(double input[3], double output[3]){
	output[0] = input[0];
	output[1] = input[2];
	output[2] = input[1];
}

void dials_2_numpy_matrix(double vector[3], double result[3])
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

void get_increment_ratio(
	double *increment_ratio_x, 
	double *increment_ratio_y, 
	double *increment_ratio_z,
	double theta,
	double phi,
	double face
){
	if (face == 1) {
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(M_PI - theta) / cos(fabs(phi));
		*increment_ratio_z = tan(phi);
	}
	else if (face == 2) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
		else {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = -1;
		}
	}
	else if (face == 3) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = 1 / tan(fabs(phi));
			*increment_ratio_y = tan(theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
		else {
			*increment_ratio_x = 1 / (tan(fabs(phi)));
			*increment_ratio_y = tan(M_PI - theta) / sin(fabs(phi));
			*increment_ratio_z = 1;
		}
	}
	else if (face == 4) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = cos(fabs(phi)) / tan(fabs(theta));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(phi) / tan(fabs(theta));
		}
		else {
			*increment_ratio_x = cos(fabs(phi)) / (tan((M_PI - fabs(theta))));
			*increment_ratio_y = 1;
			*increment_ratio_z = sin(-phi) / (tan((M_PI - fabs(theta))));
		}
	}
	else if (face == 5) {
		if (fabs(theta) < M_PI / 2) {
			*increment_ratio_x = cos(fabs(phi)) / (tan(fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(fabs(theta)));
		}
		else {
			*increment_ratio_x = cos(fabs(phi)) / (tan(M_PI - fabs(theta)));
			*increment_ratio_y = -1;
			*increment_ratio_z = sin(phi) / (tan(M_PI - fabs(theta)));
		}
	}
	else if (face == 6) {
		*increment_ratio_x = -1;
		*increment_ratio_y = tan(theta) / cos(phi);
		*increment_ratio_z = tan(phi);
	}
	else {
		printf("ERROR! Unrecognised value of face.\n");
		*increment_ratio_x = 0;
		*increment_ratio_y = 0;
		*increment_ratio_z = 0;
	}
}

int64_t get_maximum_increment(
	int64_t x, int64_t y, int64_t z,
	int64_t x_max, int64_t y_max, int64_t z_max,
	double theta, int64_t face
){
	if (face == 1) { // FRONTZY
		return(x_max - x);
	}
	else if (face == 2) { // LEYX
		return(z + 1);
	}
	else if (face == 3) { // RIYX
		if (fabs(theta) < M_PI / 2) {
			return(z_max - z);
		}
		else {
			return(z_max - z + 1);
		}
	}
	else if (face == 4) { // TOPZX
		return(y + 1);
	}
	else if (face == 5) { // BOTZX
		if (fabs(theta) < M_PI / 2) {
			return(y_max - y);
		}
		else {
			return(y_max - y + 1);
		}
	}
	else if (face == 6) { // BACKZY
		return(x + 1);
	}
	else {
		printf("ERROR! Unrecognised value of face.\n");
		return(0);
	}
}

void get_new_coordinates(
	int64_t *new_x, int64_t *new_y, int64_t *new_z,
	int64_t x, int64_t y, int64_t z,
	double increment_ratio_x, double increment_ratio_y, double increment_ratio_z,
	int64_t increment, double theta, int64_t face
){
	if (face == 1) {
		if (theta > 0) {
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = floor(x - increment * increment_ratio_x); 
			*new_y = floor(y - increment * increment_ratio_y);
			*new_z = floor(z - increment * increment_ratio_z);
		}
		else {
			// this -1 represents that the opposition of direction
			// between the lab x-axis and the wavevector
			*new_x = round(x - increment * increment_ratio_x);
			*new_y = round(y - increment * increment_ratio_y);
			*new_z = round(z - increment * increment_ratio_z);
		}
	}
	else if (face == 2) {
		if (fabs(theta) < M_PI / 2) {
			if (theta > 0) {
				*new_x = floor(x + -1 * increment * increment_ratio_x);
				*new_y = floor(y - increment * increment_ratio_y);
				*new_z = floor(z + increment * increment_ratio_z);
			}
			else {
				*new_x = round(x + -1 * increment * increment_ratio_x);
				*new_y = round(y - increment * increment_ratio_y);
				*new_z = round(z + increment * increment_ratio_z);
			}
		}
		else {
			if (theta > 0) {
				*new_x = floor(x + 1 * increment * increment_ratio_x);
				*new_y = floor(y - increment * increment_ratio_y);
				*new_z = floor(z + increment * increment_ratio_z);
			}
			else {
				*new_x = round(x + 1 * increment * increment_ratio_x);
				*new_y = round(y - increment * increment_ratio_y);
				*new_z = round(z + increment * increment_ratio_z);
			}
		}
	}
	else if (face == 3) {
		if (fabs(theta) < M_PI / 2) {
			if (theta > 0) {
				*new_x = floor(x + -1 * increment * increment_ratio_x);
				*new_y = floor(y - increment * increment_ratio_y);
				*new_z = floor(z + increment * increment_ratio_z);
			}
			else {
				*new_x = round(x + -1 * increment * increment_ratio_x);
				*new_y = round(y - increment * increment_ratio_y);
				*new_z = round(z + increment * increment_ratio_z);
			}
		}
		else {
			if (theta > 0) {
				*new_x = floor(x + 1 * increment * increment_ratio_x);
				*new_y = floor(y - increment * increment_ratio_y);
				*new_z = floor(z + increment * 1);
			}
			else {
				*new_x = round(x + 1 * increment * increment_ratio_x);
				*new_y = round(y - increment * increment_ratio_y);
				*new_z = round(z + increment * 1);
			}
		}
	}
	else if (face == 4) {
		if (fabs(theta) < M_PI / 2) {
			*new_x = floor(x + -1 * increment * increment_ratio_x);
			*new_y = floor(y - increment * increment_ratio_y);
			*new_z = floor(z + increment * increment_ratio_z);
		}
		else {
			*new_x = floor(x + 1 * increment * increment_ratio_x);
			*new_y = floor(y - increment * increment_ratio_y);
			*new_z = floor(z + increment * increment_ratio_z);
		}
	}
	else if (face == 5) {
		if (fabs(theta) < M_PI / 2) {
			*new_x = round(x + -1 * increment * increment_ratio_x);
			*new_y = round(y - increment * increment_ratio_y);
			*new_z = round(z + increment * increment_ratio_z);
		}
		else {
			*new_x = round(x + 1 * increment * increment_ratio_x);
			*new_y = round(y - increment * increment_ratio_y);
			*new_z = round(z - increment * increment_ratio_z);
		}
	}
	else if (face == 6) {
		if (theta > 0) {
			*new_x = floor(x + increment * increment_ratio_x);
			*new_y = floor(y - increment * increment_ratio_y);
			*new_z = floor(z + increment * increment_ratio_z);
		}
		else {
			*new_x = round(x + increment * increment_ratio_x);
			*new_y = round(y - increment * increment_ratio_y);
			*new_z = round(z + increment * increment_ratio_z);
		}
	}
}

void check_boundaries(
	int64_t *new_x, int64_t *new_y, int64_t *new_z,
	int64_t x_max, int64_t y_max, int64_t z_max
) {
	if (*new_x >= x_max) *new_x = x_max - 1;
	else if (*new_x < 0) *new_x = 0;
	
	if (*new_y >= y_max) *new_y = y_max - 1;
	else if (*new_y < 0) *new_y = 0;
	
	if (*new_z >= z_max) *new_z = z_max - 1;
	else if (*new_z < 0) *new_z = 0;
}


Path2_c cal_coord(
	double theta, double phi, int64_t *coord, int64_t face,
	int64_t *shape, int8_t *label_list_1d, int64_t full_iteration
){
    Path2_c result;
    int64_t z = coord[0], y = coord[1], x = coord[2];
    int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int64_t diagonal = x_max * sqrt(3);

    int64_t *path_2 = (int64_t *) malloc(diagonal *3* sizeof(int64_t));
    int64_t *classes_posi = (int64_t *) malloc(diagonal * sizeof(int64_t));
    int64_t *classes = (int64_t *) malloc(diagonal * sizeof(int64_t));
    classes[0] = 3;
    classes_posi[0] = 0;

    double increment_ratio_x, increment_ratio_y, increment_ratio_z;
    get_increment_ratio(&increment_ratio_x, &increment_ratio_y, &increment_ratio_z, theta, phi, face);

    int64_t pos = 0;
    int64_t len_path_2 = 1;
    int64_t len_classes = 1;
    int64_t len_classes_posi = 1;
    int64_t new_z, new_y, new_x;
    int64_t max_increment = get_maximum_increment(x, y, z, x_max, y_max, z_max, theta, face);

    if (face >= 1 && face <= 6) {
        //for (int64_t increment = 0; increment < max_increment; increment++){
        for (int64_t increment = 0; increment < diagonal; increment++){
            get_new_coordinates(
                &new_x, &new_y, &new_z,
                x, y, z,
                increment_ratio_x, increment_ratio_y, increment_ratio_z,
                increment, theta, face
            );
            
            //check_boundaries(&new_x, &new_y, &new_z, x_max, y_max, z_max);
			
            if(
				new_x>=x_max || new_x < 0 ||
				new_y>=y_max || new_y < 0 ||
				new_z>=z_max || new_z < 0
			) break;
			
            
            int64_t potential_coord[3] = {new_z, new_y, new_x};
            //int64_t label = label_list[new_z][new_y][new_x];
            pos = INDEX_3D(z_max, y_max, x_max, new_z, new_y, new_x);
            int64_t label = label_list_1d[pos];
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

            int64_t previous_step[3] = {
                path_2[(increment - 1) * 3],
                path_2[(increment - 1) * 3 + 1],
                path_2[(increment - 1) * 3 + 2]
            };
            pos = INDEX_3D(z_max, y_max, x_max, previous_step[0], previous_step[1], previous_step[2]);
            int64_t previous_label = label_list_1d[pos];
            appending(increment, path_2,
                classes, classes_posi,
                potential_coord,
                label, previous_label,
                &len_classes, &len_classes_posi, &len_path_2
            );
        }
    }
    else {
        printf("Error: face is not in the range of 1 to 6");
    }


    result.len_path_2 = len_path_2;
    result.len_classes = len_classes;
    result.len_classes_posi = len_classes_posi;
    result.posi = (int64_t *) malloc(len_classes_posi * sizeof(int64_t));
    result.classes = (int64_t *) malloc(len_classes * sizeof(int64_t));
    result.ray = (int64_t *) malloc(len_path_2 * 3 * sizeof(int64_t));
    result.ray_classes = (int64_t *) malloc(len_path_2 * sizeof(int64_t));
    for (int64_t i = 0; i < len_path_2 * 3; i++)
    {
        result.ray[i] = path_2[i];
    }
    for (int64_t i = 0; i < len_path_2; i++)
    {
        int64_t pos = INDEX_3D(z_max, y_max, x_max, path_2[3*i+0], path_2[3*i+1], path_2[3*i+2]);
        int64_t label = label_list_1d[pos];
        result.ray_classes[i] = label;
    }
    for (int64_t i = 0; i < len_classes_posi; i++)
    {
        result.posi[i] = classes_posi[i];
    }
    for (int64_t i = 0; i < len_classes; i++)
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
    printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    }


    

    free(path_2);
    free(classes_posi);
    free(classes);
    // malloc_trim(0);
    return result;
}
//**********************************************

Path2_c cal_coord_ref(double theta, double phi, int64_t *coord, int64_t face,
                  int64_t *shape, int8_t ***label_list, int64_t full_iteration
)
{
    Path2_c result;
    int64_t z = coord[0], y = coord[1], x = coord[2];
    int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
    int64_t diagonal = x_max * sqrt(3);

    int64_t *path_2 = (int64_t *) malloc(diagonal *3* sizeof(int64_t));
    int64_t *classes_posi = (int64_t *) malloc(diagonal * sizeof(int64_t));
    int64_t *classes = (int64_t *) malloc(diagonal * sizeof(int64_t));
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
        // assert(fabs(theta) <= M_PI / 2);
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
        //    assert(theta > 0);
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
                // printf("increment %ld", increment);
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
        // assert(fabs(theta) <= M_PI / 2);
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
            // printf("new_x: %ld, new_y: %ld, new_z: %ld", new_x, new_y, new_z);
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
            // printf("new_x: %ld, new_y: %ld, new_z: %ld", new_x, new_y, new_z);
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
    // printf("Length of 2d array in C: %ld \n", len_path_2);
    // printf("Length of classes in C: %ld \n", len_classes);
    // printf("Length of classes_posi in C: %ld \n", len_classes_posi);
    // result.ray = path_2;
    // result.posi = classes_posi;
    // result.classes = classes;


    result.len_path_2 = len_path_2;
    result.len_classes = len_classes;
    result.len_classes_posi = len_classes_posi;
    result.posi = (int64_t *) malloc(len_classes_posi * sizeof(int64_t));
    result.classes = (int64_t *) malloc(len_classes * sizeof(int64_t));
    result.ray = (int64_t *) malloc(len_path_2 * 3 * sizeof(int64_t));
    result.ray_classes = (int64_t *) malloc(len_path_2 * sizeof(int64_t));
    for (int64_t i = 0; i < len_path_2 * 3; i++)
    {
        result.ray[i] = path_2[i];
    }
    for (int64_t i = 0; i < len_path_2; i++)
    {
        int64_t label = label_list[path_2[3*i+0]][path_2[3*i+1]][path_2[3*i+2]];
        result.ray_classes[i] = label;
    }
    for (int64_t i = 0; i < len_classes_posi; i++)
    {
        // printf("classes_posi is %ld \n", classes_posi[i]);
        result.posi[i] = classes_posi[i];
    }
    for (int64_t i = 0; i < len_classes; i++)
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
    printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    }

    if (test_mod)
    {

    printf( "diagonal is %ld \n", diagonal);
    printf("len_path_2 is %ld \n", len_path_2);
    printf("len_classes is %ld \n", len_classes);
    printf("len_classes_posi is %ld \n", len_classes_posi);
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


double *cal_path2_plus(Path2_c path_2_cal_result, double *voxel_size)
{
    double *result = (double *) malloc(4 * sizeof(double));
    double voxel_length_z = voxel_size[0];
    double voxel_length_y = voxel_size[1];
    double voxel_length_x = voxel_size[2];
    int64_t *path_ray = path_2_cal_result.ray;
    int64_t *path_ray_classes = path_2_cal_result.ray_classes;
    int64_t *posi = path_2_cal_result.posi;
    int64_t *classes = path_2_cal_result.classes;
    int64_t len_path_2 = path_2_cal_result.len_path_2;
    int64_t len_classes = path_2_cal_result.len_classes;
    int64_t len_classes_posi = path_2_cal_result.len_classes_posi;
    
    double dist_x = (path_ray[(len_path_2 - 1) * 3 + 2] - path_ray[2]);
    double dist_y = (path_ray[(len_path_2 - 1) * 3 + 1] - path_ray[1]);
    double dist_z = (path_ray[(len_path_2 - 1) * 3 + 0] - path_ray[0]);
    double total_length = sqrt(
		pow(dist_y * voxel_length_y, 2) +
		pow(dist_z * voxel_length_z, 2) +
        pow(dist_x * voxel_length_x, 2)
	);
    // printf("CPU:==>total_length=%f; ", total_length);
    // printf ("len_path_2=%ld\n", len_path_2);
    // printf ("dist_x=%f , dist_y=%f, dist_z=%f \n", dist_x, dist_y, dist_z);
    int64_t cr_l_2_int = 0;
    int64_t li_l_2_int = 0;
    int64_t bu_l_2_int = 0;
    int64_t lo_l_2_int = 0;

    for (int j = 0; j < len_path_2; j++) {
        if (path_ray_classes[j] == 3) cr_l_2_int++;
        else if (path_ray_classes[j] == 1) li_l_2_int++;
        else if (path_ray_classes[j] == 2) lo_l_2_int++;
        else if (path_ray_classes[j] == 4) bu_l_2_int++;
        else {
        }
    }
    // printf("li_l_2_int=%ld; ", li_l_2_int);
    // printf("lo_l_2_int=%ld; ", lo_l_2_int);
    // printf("cr_l_2_int=%ld; ", cr_l_2_int);
    // printf("bu_l_2_int=%ld\n", bu_l_2_int);
    // printf("path_ray_classes is \n")   ;
    // for(int k =0; k<len_path_2;k++){
    //     printf(" %d",path_ray_classes[k]);
    // }
    // printf("\n");
	int64_t sum = cr_l_2_int + li_l_2_int + lo_l_2_int + bu_l_2_int;
	//printf("total_length=%f; len_path_2=%f; sum=%ld; dst=[%f; %f; %f]\n", total_length, (double) len_path_2, sum, dist_x, dist_y, dist_z);
    double cr_l_2 = total_length*(((double) cr_l_2_int)/((double) len_path_2));
    double li_l_2 = total_length*(((double) li_l_2_int)/((double) len_path_2));
    double bu_l_2 = total_length*(((double) bu_l_2_int)/((double) len_path_2));
    double lo_l_2 = total_length*(((double) lo_l_2_int)/((double) len_path_2));
    result[2] = cr_l_2;
    result[1] = lo_l_2;
    result[0] = li_l_2;
    result[3] = bu_l_2;
    return result;
}

double *cal_path2_plus_ref(Path2_c path_2_cal_result, double *voxel_size)
{
    double *result = (double *) malloc(4 * sizeof(double));
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


double cal_rate_single(double *numbers, double *coefficients,
                char Isexp)
{

    double mu_li = coefficients[0];
    double mu_lo = coefficients[1];
    double mu_cr = coefficients[2];
    double mu_bu = coefficients[3];

    double li_l_1 = numbers[0];
    double lo_l_1 = numbers[1];
    double cr_l_1 = numbers[2];
    double bu_l_1 = numbers[3];



    double result = (mu_li * (li_l_1 ) +
                     mu_lo * (lo_l_1 ) +
                     mu_cr * (cr_l_1 ) +
                     mu_bu * (bu_l_1 ));

    if (Isexp == 1)
    {
        result = exp(-result);
    }

    return result;
}




int verificaton_ray_tracing_path(double *h_angle_list,int32_t h_len_result, int64_t * h_coord_list,int32_t h_len_coord_list,int8_t *label_list_1d ,double * voxel_size,double *coefficients,int64_t *shape, double * h_result_list);

#ifdef __cplusplus
extern "C" {
#endif

double* ray_tracing_gpu_verification(
double *angle_list,int32_t len_result, int64_t * coord_list,int32_t len_coord_list,int8_t *label_list_1d ,double *voxel_size,double *coefficients,int64_t *shape)
{
    printf("\n------------------ Ray tracing sampling --------------\n");
	printf("--------------> GPU version\n");
    double *result_list = (double*) malloc(len_result * sizeof(double));
	int *h_face, *h_ray_classes;
	double *h_angles, *h_absorption,*h_result_list;
	int64_t z_max = shape[0], y_max = shape[1], x_max = shape[2];
	int64_t diagonal = x_max*sqrt(3);
	// int64_t face_size = len_result*len_coord_list*2*sizeof(int);
	int64_t absorption_size = len_result *sizeof(double);
	int64_t ray_classes_size = diagonal*len_coord_list*2*sizeof(int);
	int64_t angle_size = 4*sizeof(double);
    h_result_list = (double *) malloc(len_result*sizeof(double));
	verificaton_ray_tracing_path(angle_list,len_result,  coord_list,len_coord_list,label_list_1d , voxel_size,coefficients,shape,h_result_list);

	// h_face = (int *) malloc(face_size);
	// h_ray_classes = (int *) malloc(ray_classes_size);
	// h_angles = (double *) malloc(angle_size); 
	// h_absorption = (double *) malloc(absorbtion_size);

	// ray_tracing_path(h_face, h_angles, h_ray_classes, h_absorption, coord_list, len_coord_list, rotated_s1, xray, voxel_size, coefficients, label_list_1d, shape);

	// printf("----> GPU : rayclass of the first voxel in for scattering is;\n");
    // for (int i=0;i<diagonal;i++){
    //     printf(" %d", h_ray_classes[i]);
    // }
	printf("----> GPU version FINISHED;\n");
    
    // for (int i=0;i<len_result;i++){
    //     result_list[i] = angle_list[i]/180*M_PI;
    // }
    // printf("crystall coordinates: \n");
    // for (int i=0;i<10;i++){
    //     printf(" [%ld,%ld,%ld] ", coord_list[i*3+0], coord_list[i*3+1], coord_list[i*3+2]);

    // }
	// printf("\n");
    // for (int i=0;i<10;i++){
    //     int k=i+10;
    //     printf(" [%d] ",label_list_1d[k]);

    // }
    printf("\n");
	free(h_face);
	free(h_ray_classes);
	free(h_angles);
	free(h_absorption);
    
    return result_list;
}


#ifdef __cplusplus
}
#endif
