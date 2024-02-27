// #define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/resource.h>
#include "bisection.h"
#include "testkit.h"
#include "matrices.h"
#include "ray_tracing.h"
#include <unistd.h>
#include <omp.h>
#include <sys/types.h>
// #include "ray_tracing.h"
#define M_PI 3.14159265358979323846
#define test_mod 0

// #ifdef __cplusplus
// extern "C"
// {
// #endif
double ib_test(
    int64_t *coord_list,
    int64_t len_coord_list,
    double *rotated_s1, double *xray,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration,
    int64_t store_paths,int num_cls)
{

    // printArray(crystal_coordinate_shape, 3);
    // printArrayD(rotated_s1, 3);
    // printArrayD(xray, 3);
    // if (test_mod)
    // {
    //     struct rusage usage;
    //     getrusage(RUSAGE_SELF, &usage);
    //     printf("The starting Memory usage: %ld KB\n", usage.ru_maxrss);
    // }

    // in the theta phi determination, xray will be reversed
    // so create a new array to store the original xray to process

    double x_ray_angle[3], x_ray_trans[3];
    double rotated_s1_angle[3], rotated_s1_trans[3];
    memcpy(x_ray_angle, xray, sizeof(xray) * 3);
    memcpy(x_ray_trans, xray, sizeof(xray) * 3);
    memcpy(rotated_s1_angle, rotated_s1, sizeof(rotated_s1) * 3);
    memcpy(rotated_s1_trans, rotated_s1, sizeof(rotated_s1) * 3);

    // for (int64_t i = 0; i < 3; i++)
    // {
    //     x_ray_c[i] = xray[i];
    // }

    // printArrayD(voxel_size, 3);
    // printArrayD(coefficients, 3);
    // printArray(shape, 3);
    // printf("%d ", len_coordinate_list);

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
    ThetaPhi result_1 = dials_2_thetaphi_22(x_ray_angle, 1);

    double theta = result_2.theta;
    double phi = result_2.phi;
    double theta_1 = result_1.theta;
    double phi_1 = result_1.phi;

    Path2_c path_2, path_1;
    double *numbers_1, *numbers_2;
    double absorption;
    double absorption_sum = 0, absorption_mean = 0;

    double resolution = 1.0;
    double xray_direction[3], scattered_direction[3];
    dials_2_myframe(x_ray_trans, xray_direction);
    dials_2_myframe(rotated_s1_trans, scattered_direction);

    for (int64_t i = 0; i < len_coord_list; i++)
    {

        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};

        int64_t face_1 = cube_face(coord, xray_direction, shape, 1);
        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);
        // printf("ibpath_1\n");
        // printf("face_1 is %d \n", face_1);
        // printf("theta_1 is %f \n", theta_1);
        // printf("phi_1 is %f \n", phi_1);
        Path_iterative_bisection ibpath_1 = iterative_bisection(theta_1, phi_1,
                                                                coord, face_1, label_list, shape, resolution, num_cls);
        // printf("ibpath_2\n");
        Path_iterative_bisection ibpath_2 = iterative_bisection(theta, phi,
                                                                coord, face_2, label_list, shape, resolution, num_cls);
        if (test_mod)
        {
            printf("i is %d \n", i);
            printf("ibpath_1\n");
            printArray(ibpath_1.path, (ibpath_1.length + 1) * 3);
            printArrayshort(ibpath_1.classes, ibpath_1.length + 1);
            printArray(ibpath_1.boundary_list, ibpath_1.length + 1);
            printf("ibpath_2\n");
            printArray(ibpath_2.path, (ibpath_2.length + 1) * 3);
            printArrayshort(ibpath_2.classes, ibpath_2.length + 1);
            printArray(ibpath_2.boundary_list, ibpath_2.length + 1);
        }
        numbers_1 = cal_path_bisection(ibpath_1, voxel_size);
        numbers_2 = cal_path_bisection(ibpath_2, voxel_size);
        if (test_mod)
        {
            printf("numbers_1\n");
            print_matrix(numbers_1, 1, 4);
            printf("numbers_2\n");
            print_matrix(numbers_2, 1, 4);
            printf("\n");
        }
        absorption = cal_rate(numbers_1, numbers_2, coefficients, 1);
        absorption_sum += absorption;
        // if (test_mod)
        // {
        //     printf("ibpath_1\n");
        //     printArray(ibpath_1.path, (ibpath_1.length) * 3);
        //     printf("ibpath_1.classes\n");
        //     printArrayshort(ibpath_1.classes, ibpath_1.length );
        //     printf("ibpath_1.boundary_list\n");
        //     printArrayshort(ibpath_1.boundary_list, ibpath_1.length);
        //     printf("%d\n", ibpath_1.length);
        //     printf("numbers_1\n");
        //     printArrayD(numbers_1, 4);

        //     printf("ibpath_2\n");
        //     printArray(ibpath_2.path, (ibpath_2.length ) * 3);
        //     printf("ibpath_2.classes\n");
        //     printArrayshort(ibpath_2.classes, ibpath_2.length );
        //     printf("ibpath_2.boundary_list\n");
        //     printArrayshort(ibpath_2.boundary_list, ibpath_2.length );
        //     printf("%d\n", ibpath_2.length);
        //     printf("numbers_2\n");
        //     printArrayD(numbers_2, 4);
        // }
        free(ibpath_1.path);
        free(ibpath_2.path);
        free(ibpath_1.classes);
        free(ibpath_2.classes);
        free(ibpath_1.boundary_list);
        free(ibpath_2.boundary_list);
        free(numbers_1);
        free(numbers_2);
    }
    // free(numbers_2);

    absorption_mean = absorption_sum / len_coord_list;
    // printf("absorption_mean: %f\n", absorption_mean);
    return absorption_mean;
}





double ray_tracing_single(
    int64_t *coord_list,
    int64_t len_coord_list,
    const double *rotated_s1, const double *xray,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration,
    int64_t store_paths,int IsExp)
{

    double x_ray_angle[3], x_ray_trans[3];
    double rotated_s1_angle[3], rotated_s1_trans[3];
    memcpy(x_ray_angle, xray, 3 * sizeof(xray));
    memcpy(x_ray_trans, xray, 3 * sizeof(xray));
    memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(rotated_s1));
    memcpy(rotated_s1_trans, rotated_s1, 3 * sizeof(rotated_s1));

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
    ThetaPhi result_1 = dials_2_thetaphi_22(x_ray_angle, 1);

    double theta = result_2.theta;
    double phi = result_2.phi;
    double theta_1 = result_1.theta;
    double phi_1 = result_1.phi;

    Path2_c path_2, path_1;
    double *numbers_1, *numbers_2;
    double absorption;
    double absorption_sum = 0, absorption_mean = 0;

    double xray_direction[3], scattered_direction[3];
    dials_2_myframe(x_ray_trans, xray_direction);
    dials_2_myframe(rotated_s1_trans, scattered_direction);

    for (int64_t i = 0; i < len_coord_list; i++)
    {

        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};

        int64_t face_1 = cube_face(coord, xray_direction, shape, 1);
        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);
        if (face_1 == 1 && fabs(theta_1) < M_PI / 2)
        {
            printArray(coord, 3);
            printf("face_1 is  %d \n", face_1);
            printf("theta_1 is %f ", theta_1);
            printf("phi_1 is %f \n", phi_1);
            print_matrix(xray, 1, 3);
        }
        if (face_2 == 1 && fabs(theta) < M_PI / 2)
        {
            printArray(coord, 3);
            printf("face_2 is  %d \n", face_2);
            printf("theta is %f ", theta);
            printf("phi is %f \n", phi);
            print_matrix(rotated_s1, 1, 3);
        }
        if (test_mod)
        {
            printf("\n");
            printf("theta_1 is %f ", theta_1);
            printf("phi_1 is %f ", phi_1);
            printf("\n");
            printf("face_1 at %d is %d \n", i, face_1);
        }

        path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list, full_iteration);
        if (test_mod)
        {
            printf("path_1111 at %d is good  \n", i);
            printf("face_2 is  ");
            printf("%d \n", face_2);
            printf("theta is %f ", theta);
            printf("phi is %f \n", phi);
            printf("\n");
            printf("classes are ");
            for (int64_t j = 0; j < path_1.len_classes_posi; j++)
            {
                printf("%d ", path_1.classes[j]);
            }
            printf("\n");
            printf("posi are ");
            for (int64_t j = 0; j < path_1.len_classes_posi; j++)
            {
                printf("%d ", path_1.posi[j]);
            }
           
        }
        // printf("face_2 at %d is %d \n",i, face_2);

        path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);
        if (test_mod)
        {
            printf("path_2222 at %d is good  \n", i);
        }

        numbers_1 = cal_path2_plus(path_1, voxel_size);
        numbers_2 = cal_path2_plus(path_2, voxel_size);

        absorption = cal_rate(numbers_1, numbers_2, coefficients, IsExp);

        if (test_mod)
        {
            printf("numbers_1 is  ");
            printArrayD(numbers_1, 4);
            printf("numbers_2 is");
            printArrayD(numbers_2, 4);
            printf("absorption is %f at %d \n", absorption, i);
            printf("\n");
        }

        absorption_sum += absorption;
        free(path_1.ray);
        free(path_1.classes);
        free(path_1.posi);

        free(numbers_1);
        free(path_2.ray);
        free(path_2.classes);
        free(path_2.posi);
        free(numbers_2);
    }
    absorption_mean = absorption_sum / len_coord_list;

    return absorption_mean;
}

double ray_tracing_single_mp(
    int64_t *coord_list,
    int64_t len_coord_list,
    const double *rotated_s1, const double *xray,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration,
    int64_t store_paths,int IsExp,int num_workers){
    omp_set_num_threads(num_workers);
        double x_ray_angle[3], x_ray_trans[3];
    double rotated_s1_angle[3], rotated_s1_trans[3];
    memcpy(x_ray_angle, xray, 3 * sizeof(xray));
    memcpy(x_ray_trans, xray, 3 * sizeof(xray));
    memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(rotated_s1));
    memcpy(rotated_s1_trans, rotated_s1, 3 * sizeof(rotated_s1));

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
    ThetaPhi result_1 = dials_2_thetaphi_22(x_ray_angle, 1);

    double theta = result_2.theta;
    double phi = result_2.phi;
    double theta_1 = result_1.theta;
    double phi_1 = result_1.phi;
    // printf("theta_1 is %f \n", theta_1);
    // printf("phi_1 is %f \n", phi_1);
    // printf("theta is %f \n", theta);
    // printf("phi is %f \n", phi);

    double absorption_sum = 0, absorption_mean = 0;

    double xray_direction[3], scattered_direction[3];
    dials_2_myframe(x_ray_trans, xray_direction);
    dials_2_myframe(rotated_s1_trans, scattered_direction);

    #pragma omp parallel for default(none) shared(coord_list, xray_direction, scattered_direction, shape, label_list, full_iteration, voxel_size, coefficients, len_coord_list, theta_1, phi_1, theta, phi, store_paths, IsExp) reduction(+ : absorption_sum)

    for (int64_t i = 0; i < len_coord_list; i++)
    {
            Path2_c path_2, path_1;
    double *numbers_1, *numbers_2;
    double absorption;
        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};

        int64_t face_1 = cube_face(coord, xray_direction, shape, 1);
        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);

        path_1 = cal_coord(theta_1, phi_1, coord, face_1, shape, label_list, full_iteration);

        path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);

        numbers_1 = cal_path2_plus(path_1, voxel_size);
        numbers_2 = cal_path2_plus(path_2, voxel_size);

        absorption = cal_rate(numbers_1, numbers_2, coefficients, IsExp);
 
        absorption_sum += absorption;
        free(path_1.ray);
        free(path_1.classes);
        free(path_1.posi);

        free(numbers_1);
        free(path_2.ray);
        free(path_2.classes);
        free(path_2.posi);
        free(numbers_2);
    }
    // printf("absorption_sum is %f \n", absorption_sum);
    absorption_mean = absorption_sum / len_coord_list;
    return absorption_mean;
    }
double *ray_tracing_overall(int64_t low, int64_t up,
                            int64_t *coord_list,
                            int64_t len_coord_list,
                            const double *scattering_vector_list, const double *omega_list,
                            const double *raw_xray,
                            const double *omega_axis, const double *kp_rotation_matrix,
                            int64_t len_result,
                            double *voxel_size, double *coefficients,
                            int8_t ***label_list, int64_t *shape, int full_iteration,
                            int store_paths, int num_workers,int IsExp)
{
    omp_set_num_threads(num_workers);

    printf("low is %d \n", low);
    printf("up is %d \n", up);
    double *result_list = (double *)malloc(len_result * sizeof(double));
    printf("result_list is %p \n", result_list);

#pragma omp parallel for default(none) shared(label_list, coord_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, len_result, voxel_size, coefficients, shape, full_iteration, store_paths, low, up, len_coord_list, result_list,IsExp)
    for (int64_t i = 0; i < len_result; i++)
    {
        double result;
        double rotation_matrix_frame_omega[9];
        double rotation_matrix[9];
        double total_rotation_matrix[9];
        double xray[3];
        double rotated_s1[3];

        kp_rotation(omega_axis, omega_list[i], (double *)rotation_matrix_frame_omega);

        dot_product((double *)rotation_matrix_frame_omega, kp_rotation_matrix, (double *)rotation_matrix, 3, 3, 3);

        transpose((double *)rotation_matrix, 3, 3, (double *)total_rotation_matrix);
        dot_product((double *)total_rotation_matrix, raw_xray, (double *)xray, 3, 3, 1);

        double scattering_vector[3] = {scattering_vector_list[i * 3],
                                       scattering_vector_list[i * 3 + 1],
                                       scattering_vector_list[i * 3 + 2]};
        dot_product((double *)total_rotation_matrix, (double *)scattering_vector, (double *)rotated_s1, 3, 3, 1);

        result = ray_tracing_single(
            coord_list, len_coord_list,
            (double *)rotated_s1, (double *)xray,
            voxel_size, coefficients,
            label_list, shape, full_iteration,
            store_paths,IsExp);

        result_list[i] = result;
        printf("[%d/%d] rotation: %.4f, absorption: %.4f\n",
               low + i, up, omega_list[i] * 180 / M_PI, result);
    }

    return result_list;
}

double ib_single_gridding(
    int64_t *coord,
    const double *rotated_s1, double theta, double phi,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration, int index,int num_cls,int IsExp)
{

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1, 0);
    if (fabs(result_2.theta - theta) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where theta is %f \n", index, theta);
        printf("result_2.theta is %f \n", result_2.theta);
        assert(fabs(result_2.theta - theta) < 1e-6);
    }
    if (fabs(result_2.phi - phi) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where phi is %f \n", index, phi);
        assert(fabs(result_2.phi - phi) < 1e-6);
    }

    double resolution = 1.0;
    Path2_c path_2;
    double *numbers_2;
    double numbers_1[4] = {0, 0, 0, 0};
    double absorption;
    double scattered_direction[3];
    dials_2_myframe(rotated_s1, scattered_direction);

    int64_t face_2;
    face_2 = cube_face(coord, scattered_direction, shape, 0);

    Path_iterative_bisection ibpath_2 = iterative_bisection(theta, phi,
                                                                coord, face_2, label_list, shape, resolution, num_cls);

    numbers_2 = cal_path_bisection(ibpath_2, voxel_size);

    absorption = cal_rate(numbers_1, numbers_2, coefficients, IsExp);

        // free(ibpath_1.path);
        free(ibpath_2.path);
        // free(ibpath_1.classes);
        free(ibpath_2.classes);
        // free(ibpath_1.boundary_list);
        free(ibpath_2.boundary_list);
        // free(numbers_1);
        free(numbers_2);
    return absorption;

}


double ray_tracing_single_gridding(
    int64_t *coord,
    const double *rotated_s1, double theta, double phi,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration, int index,int IsExp)
{

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1, 0);
    if (fabs(result_2.theta - theta) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where theta is %f \n", index, theta);
        printf("result_2.theta is %f \n", result_2.theta);
        assert(fabs(result_2.theta - theta) < 1e-6);
    }
    if (fabs(result_2.phi - phi) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where phi is %f \n", index, phi);
        assert(fabs(result_2.phi - phi) < 1e-6);
    }

    
    Path2_c path_2;
    double *numbers_2;
    double numbers_1[4] = {0, 0, 0, 0};
    double absorption;
    double absorption_sum = 0, absorption_mean = 0;
    double scattered_direction[3];
    dials_2_myframe(rotated_s1, scattered_direction);

    int64_t face_2;
    face_2 = cube_face(coord, scattered_direction, shape, 0);

    path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);

    numbers_2 = cal_path2_plus(path_2, voxel_size);

    absorption = cal_rate(numbers_1, numbers_2, coefficients, IsExp);

    free(path_2.ray);
    free(path_2.classes);
    free(path_2.posi);
    free(numbers_2);
    return absorption;

}

double ib_am(
    int64_t *coord_list,
    int64_t len_coord_list,
    const double *rotated_s1, double theta, double phi,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration, int index,int num_cls,int IsExp)
{

    // in the theta phi determination, xray will be reversed
    // so create a new array to store the original xray to process
    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1, 0);
    if (fabs(result_2.theta - theta) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where theta is %f \n", index, theta);
        printf("result_2.theta is %f \n", result_2.theta);
        assert(fabs(result_2.theta - theta) < 1e-6);
    }
    if (fabs(result_2.phi - phi) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where phi is %f \n", index, phi);
        assert(fabs(result_2.phi - phi) < 1e-6);
    }

    Path2_c path_2;
    double *numbers_2;

    double absorption;
    double absorption_sum = 0, absorption_mean = 0;

    double scattered_direction[3];

    dials_2_myframe(rotated_s1, scattered_direction);
    double numbers_1[4] = {0, 0, 0, 0};
    double resolution = 1.0;

    for (int64_t i = 0; i < len_coord_list; i++)
    {

        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};

        // int64_t face_1 = cube_face(coord, xray_direction, shape, 1);
        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);

        // Path_iterative_bisection ibpath_1 = iterative_bisection(theta_1, phi_1,
        //                                                         coord, face_1, label_list, shape, resolution, num_cls);
        // printf("ibpath_2\n");
        Path_iterative_bisection ibpath_2 = iterative_bisection(theta, phi,
                                                                coord, face_2, label_list, shape, resolution, num_cls);

        // numbers_1 = cal_path_bisection(ibpath_1, voxel_size);
        numbers_2 = cal_path_bisection(ibpath_2, voxel_size);

        absorption = cal_rate(numbers_1, numbers_2, coefficients,  IsExp);
        absorption_sum += absorption;

        // free(ibpath_1.path);
        free(ibpath_2.path);
        // free(ibpath_1.classes);
        free(ibpath_2.classes);
        // free(ibpath_1.boundary_list);
        free(ibpath_2.boundary_list);
        // free(numbers_1);
        free(numbers_2);
    }
    // free(numbers_2);

    absorption_mean = absorption_sum / len_coord_list;
    return absorption_mean;
}



double ray_tracing_single_am(
    int64_t *coord_list,
    int64_t len_coord_list,
    const double *rotated_s1, double theta, double phi,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration, int index,int IsExp)
{

    ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1, 0);
    if (fabs(result_2.theta - theta) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where theta is %f \n", index, theta);
        printf("result_2.theta is %f \n", result_2.theta);
        assert(fabs(result_2.theta - theta) < 1e-6);
    }
    if (fabs(result_2.phi - phi) < 1e-6)
    {
    }
    else
    {
        printf("ERROR! there is a problem in %d where phi is %f \n", index, phi);
        assert(fabs(result_2.phi - phi) < 1e-6);
    }

    Path2_c path_2;
    double *numbers_2;
    double absorption;
    double absorption_sum = 0, absorption_mean = 0;

    double scattered_direction[3];

    dials_2_myframe(rotated_s1, scattered_direction);
    double numbers_1[4] = {0, 0, 0, 0};

    for (int64_t i = 0; i < len_coord_list; i++)
    {

        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};

        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);

        path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);

        numbers_2 = cal_path2_plus(path_2, voxel_size);

        absorption = cal_rate(numbers_1, numbers_2, coefficients, IsExp);

        absorption_sum += absorption;

        free(path_2.ray);
        free(path_2.classes);
        free(path_2.posi);
        free(numbers_2);
    }
    absorption_mean = absorption_sum / len_coord_list;

    return absorption_mean;
}

double *ray_tracing_overall_am(int64_t low, int64_t up,
                            int64_t *coord_list,
                            int64_t len_coord_list,
                            const double *scattering_vector_list, const double *omega_list,
                            const double *raw_xray,
                            const double *omega_axis, const double *kp_rotation_matrix,
                            int64_t len_result,
                            double *voxel_size, double *coefficients,
                            int8_t ***label_list, int64_t *shape, int full_iteration,
                            int store_paths, int num_workers, const double *thetaphi_list, const double *map_vector_list,int IsExp)
{
    omp_set_num_threads(num_workers);

    printf("low is %d \n", low);
    printf("up is %d \n", up);
    double *result_list = (double *)malloc(len_result * sizeof(double));
    printf("result_list is %p \n", result_list);

#pragma omp parallel for default(none) shared(label_list, coord_list, scattering_vector_list, len_result, voxel_size, coefficients, shape, full_iteration, low, up, len_coord_list, result_list, thetaphi_list,map_vector_list,IsExp)
    for (int64_t i = 0; i < len_result; i++)
    {
        double result;

        double map_vector[3] = {map_vector_list[i * 3],
                                map_vector_list[i * 3 + 1],
                                map_vector_list[i * 3 + 2]};

        double theta = thetaphi_list[i * 2];
        double phi = thetaphi_list[i * 2 + 1];
        result = ray_tracing_single_am(
            coord_list, len_coord_list,
            (double *)map_vector, theta, phi,
            voxel_size, coefficients,
            label_list, shape, full_iteration, i,IsExp);

        result_list[i] = result;
        printf("[%d/%d] map_absorption: %.4f\n",
               low + i, up, result);
    }

    return result_list;
}

// float *ray_tracing_gpu_overall(size_t low, size_t up,
//                                     int *coord_list,
//                                     size_t len_coord_list,
//                                     const float *scattering_vector_list, const float *omega_list,
//                                     const float *raw_xray,
//                                     const float *omega_axis, const float *kp_rotation_matrix,
//                                     size_t len_result,
//                                     float *voxel_size, float *coefficients, int8_t ***label_list,
//                                     int8_t *label_list_1d, int *shape, int full_iteration,
//                                     int store_paths,int gpumethod)
//     {

//         // float *result_list = malloc( len_result* sizeof(float));
//         // size_t len_result_float = (int32_t) len_result* sizeof(float);
//         // int32_t len_result_float = (int32_t) len_result;
//         printf("low is %d \n", low);
//         printf("up is %d \n", up);
//         float factor = 1;
//         // len_result = (int)(len_result*factor);
//         printf("len_result is %d \n", len_result);
//         float *h_result_list = (float *)malloc(len_result * len_coord_list * 2 * sizeof(float));
//         float *h_python_result_list = (float *)malloc(len_result * sizeof(float));
//         int *h_face = (int *)malloc(len_coord_list * 2 * sizeof(int));
//         float *h_angles = (float *)malloc(4 * sizeof(float));

//         ray_tracing_gpu_overall_kernel(low, up, coord_list, len_coord_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, len_result, voxel_size, coefficients, label_list_1d, shape, full_iteration, store_paths, h_result_list, h_face, h_angles, h_python_result_list,gpumethod);

//         free(h_result_list);
//         return h_python_result_list;
//     }
// #ifdef __cplusplus
// }
// #endif