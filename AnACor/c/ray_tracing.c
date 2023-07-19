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
#include <sys/types.h>
// #include "ray_tracing.h"
#define M_PI 3.14159265358979323846
#define test_mod 0


double ib_test(
    int64_t *coord_list,
    int64_t len_coord_list,
    double *rotated_s1, double *xray,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration,
    int64_t store_paths)
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

    int num_cls = 4;
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
    dials_2_numpy(x_ray_trans, xray_direction);
    dials_2_numpy(rotated_s1_trans, scattered_direction);

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
            printArrayshort(ibpath_1.boundary_list, ibpath_1.length + 1);
            printf("ibpath_2\n");
            printArray(ibpath_2.path, (ibpath_2.length + 1) * 3);
            printArrayshort(ibpath_2.classes, ibpath_2.length + 1);
            printArrayshort(ibpath_2.boundary_list, ibpath_2.length + 1);
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



double ray_tracing_sampling(
    int64_t *coord_list,
    int64_t len_coord_list,
    const double *rotated_s1, const double *xray,
    double *voxel_size, double *coefficients,
    int8_t ***label_list, int64_t *shape, int full_iteration,
    int64_t store_paths)
{
    // print_matrix(rotated_s1, 1, 3);
    // print_matrix(xray, 1, 3);
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
    memcpy(x_ray_angle, xray, 3 * sizeof(xray));
    memcpy(x_ray_trans, xray, 3 * sizeof(xray));
    memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(rotated_s1));
    memcpy(rotated_s1_trans, rotated_s1, 3 * sizeof(rotated_s1));

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
    // printf("rotated_s1_angle \n");
    // print_matrix(rotated_s1_angle, 1, 3);
    // printf("x_ray_angle \n");
    // print_matrix(x_ray_angle, 1, 3);
    // printf("\n");
    double theta = result_2.theta;
    double phi = result_2.phi;
    double theta_1 = result_1.theta;
    double phi_1 = result_1.phi;
    // printf("\n");
    // printf("theta: %f\n", theta);
    // printf("phi: %f\n", phi);
    // printf("theta_1: %f\n", theta_1);
    // printf("phi_1: %f\n", phi_1);
    Path2_c path_2, path_1;
    double *numbers_1, *numbers_2;
    double absorption;
    double absorption_sum = 0, absorption_mean = 0;

    double xray_direction[3], scattered_direction[3];
    dials_2_numpy(x_ray_trans, xray_direction);
    dials_2_numpy(rotated_s1_trans, scattered_direction);

    // if (test_mod)
    // {
    //     theta_1 = 0.660531;
    //     phi_1 = -0.001338;
    //     theta = -1.557793;
    //     phi = -1.560976;
    // }
    for (int64_t i = 0; i < len_coord_list; i++)
    {

        int64_t coord[3] = {coord_list[i * 3],
                            coord_list[i * 3 + 1],
                            coord_list[i * 3 + 2]};
        // printf("%d ",label_list[coord[0]][coord[1]][coord[2]]);
        // int64_t face_1 = which_face(coord, shape, theta_1, phi_1);
        // int64_t face_2 = which_face(coord, shape, theta, phi);

        int64_t face_1 = cube_face(coord, xray_direction, shape, 1);
        int64_t face_2 = cube_face(coord, scattered_direction, shape, 0);
        if (face_1 == 1 && fabs(theta_1)<M_PI/2)
        {
            printArray(coord, 3);
            printf("face_1 is  %d \n", face_1);
            printf("theta_1 is %f ", theta_1);
            printf("phi_1 is %f \n", phi_1);
            print_matrix(xray, 1, 3);
        }
        if (face_2 == 1 && fabs(theta)<M_PI/2)
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
        }
        // printf("face_2 at %d is %d \n",i, face_2);

        path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);
        if (test_mod)
        {
            printf("path_2222 at %d is good  \n", i);
        }

        // printf("Length of classes in ray tracing : %d \n", path_1.len_classes);
        // printf("Length of classes_posi in ray tracing: %d \n", path_1.len_classes_posi);
        // printArray(path_1.ray, 30);
        // printArray(path_1.posi, path_1.len_classes_posi);
        // printArray(path_1.classes, path_1.len_classes);

        numbers_1 = cal_path2_plus(path_1, voxel_size);
        numbers_2 = cal_path2_plus(path_2, voxel_size);
        // printArrayD(numbers_1, 4);
        // printArrayD(numbers_2, 4);

        absorption = cal_rate(numbers_1, numbers_2, coefficients, 1);

        if (test_mod)
        {
            printf("numbers_1 is  ");
            printArrayD(numbers_1, 4);
            printf("numbers_2 is");
            printArrayD(numbers_2, 4);
            printf("absorption is %f at %d \n", absorption, i);
            printf("\n");
            // if (i <10)
            // {

            // }
        }
        // if (i >3 ){
        //         free(path_1.ray);
        //         free(path_1.classes);
        //         free(path_1.posi);
        //         free(numbers_1);
        //         free(path_2.ray);
        //         free(path_2.classes);
        //         free(path_2.posi);
        //         free(numbers_2);
        //         break;
        // }
        absorption_sum += absorption;
        free(path_1.ray);
        free(path_1.classes);
        free(path_1.posi);

        free(numbers_1);
        free(path_2.ray);
        free(path_2.classes);
        free(path_2.posi);
        free(numbers_2);
        // printf("path_1 is \n");
        // printArrayD(numbers_1, 4);
        // printf("path_2 is \n");
        // printArrayD(numbers_2, 4);
        // printf("absorption is %.10lf \n",absorption);
        // printf("Length of classes in ray tracing : %d \n", path_1.len_classes);
        // printf("Length of classes_posi in ray tracing: %d \n", path_1.len_classes_posi);
        // on test, this is a top face,wich
        // path_2 = cal_coord(theta, phi, coord, face_2, shape, label_list, full_iteration);
    }
    absorption_mean = absorption_sum / len_coord_list;
    //         if (test_mod)
    //     {
    // printf("absorption_sum is %f \n", absorption_sum);
    // printf("len_coordinate_list is %f \n", len_coordinate_list);}
    // printf("finish \n");
    // free(path_1.ray);
    // free(path_1.classes);
    // free(path_1.posi);
    // free(numbers_1);
    // free(path_2.ray);
    // free(path_2.classes);
    // free(path_2.posi);
    // free(numbers_2);
    return absorption_mean;
}

double *ray_tracing_overall(int32_t low, int32_t up,
                            int64_t *coord_list,
                            int32_t len_coord_list,
                            const double *scattering_vector_list, const double *omega_list,
                            const double *raw_xray,
                            const double *omega_axis, const double *kp_rotation_matrix,
                            int32_t len_result,
                            double *voxel_size, double *coefficients,
                            int8_t ***label_list, int64_t *shape, int32_t full_iteration,
                            int32_t store_paths)
{

    // double *result_list = malloc( len_result* sizeof(double));
    // size_t len_result_double = (int32_t) len_result* sizeof(double);
    // int32_t len_result_double = (int32_t) len_result;
    printf("low is %d \n", low);
    printf("up is %d \n", up);
    double *result_list = malloc(len_result * sizeof(double));
    // printf("len_result_double is %d \n", len_result_double);
    printf("result_list is %p \n", result_list);
    for (int64_t i = 0; i < len_result; i++)
    {
        double result;
        double rotation_matrix_frame_omega[9];
        double rotation_matrix[9];
        double total_rotation_matrix[9];
        double xray[3];
        double rotated_s1[3];
        // printf("kap roation  \n");
        kp_rotation(omega_axis, omega_list[i], (double *)rotation_matrix_frame_omega);
        // printf("rotation_matrix_frame_omega is \n");
        // print_matrix((double*)rotation_matrix_frame_omega,3,3);
        dot_product((double *)rotation_matrix_frame_omega, kp_rotation_matrix, (double *)rotation_matrix, 3, 3, 3);

        transpose((double *)rotation_matrix, 3, 3, (double *)total_rotation_matrix);
        // printf("total_rotation_matrix is \n");
        // print_matrix((double*)total_rotation_matrix,3,3);

        // printf("xray is \n");
        // print_matrix(raw_xray,1,3);
        dot_product((double *)total_rotation_matrix, raw_xray, (double *)xray, 3, 3, 1);
        // printf("xray is \n");
        // print_matrix(xray,1,3);
        double scattering_vector[3] = {scattering_vector_list[i * 3],
                                       scattering_vector_list[i * 3 + 1],
                                       scattering_vector_list[i * 3 + 2]};
        dot_product((double *)total_rotation_matrix, (double *)scattering_vector, (double *)rotated_s1, 3, 3, 1);
        // printf("rotated_s1 is \n");
        // print_matrix(rotated_s1,1,3);
        // ThetaPhi scattering_angles = dials_2_thetaphi_22((double *)rotated_s1, 0);
        // // printf("scattering_angles is \n");
        // // printf("theta is %f \n",scattering_angles.theta);
        // // printf("phi is %f \n",scattering_angles.phi);
        // ThetaPhi incident_angles = dials_2_thetaphi_22((double *)xray, 1);
        // printf("incident_angles is %f\n",incident_angles.theta);
        // printf("incident_angles is %f\n",incident_angles.phi);

        result = ray_tracing_sampling(
            coord_list, len_coord_list,
            (double *)rotated_s1, (double *)xray,
            voxel_size, coefficients,
            label_list, shape, full_iteration,
            store_paths);
        // printf("result is %f \n",result);
        result_list[i] = result;
        printf("[%d/%d] rotation: %.4f, absorption: %.4f\n",
               low + i, up, omega_list[i] * 180 / M_PI, result);

        // printf("index is %d, result is %f \n",i,result);
        // printArrayD(result_list, 10);
    }

    return result_list;
}
