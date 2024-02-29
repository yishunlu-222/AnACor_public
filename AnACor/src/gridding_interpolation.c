#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "ray_tracing.h"
#include "matrices.h"
double absorption(double inter_1, double inter_2)
{
  return exp(-(inter_1 + inter_2));
}


double nearest_neighbor_custom(double *grid, int nx, int ny, double theta, double phi, double theta_min, double theta_max, double phi_min, double phi_max) {
    // Calculate the indices for theta and phi
    double theta_step = (theta_max - theta_min) / (nx - 1);
    double phi_step = (phi_max - phi_min) / (ny - 1);

    int theta_idx = (int)((theta - theta_min) / theta_step);
    int phi_idx = (int)((phi - phi_min) / phi_step);

    // Clamp indices to be within the grid bounds
    if (theta_idx < 0) theta_idx = 0;
    if (theta_idx >= nx) theta_idx = nx - 1;
    if (phi_idx < 0) phi_idx = 0;
    if (phi_idx >= ny) phi_idx = ny - 1;
    double output = grid[phi_idx * nx + theta_idx];
    // Access the value from the grid
    return output;
}

double nearest_neighbor_interpolate(size_t len_coord, double *theta_list, double *phi_list, double *gridding_data, size_t nx, size_t ny, double theta_1, double phi_1, double theta, double phi, double theta_min, double theta_max, double phi_min, double phi_max) {
    size_t len_gridding_data = nx * ny;
    double result = 0;

    for (size_t i = 0; i < len_coord; i++) {
        double *grid = gridding_data + i * len_gridding_data;

        // Find the nearest neighbors for theta and phi
        double inter_1 = nearest_neighbor_custom(grid, nx, ny, theta_1, phi_1, theta_min, theta_max, phi_min, phi_max);
        double inter_2 = nearest_neighbor_custom(grid, nx, ny, theta, phi, theta_min, theta_max, phi_min, phi_max);

        result += absorption(inter_1, inter_2);
    }

    return result / len_coord;
}



double interpolate(size_t len_coord, double *theta_list, double *phi_list, double *gridding_data, size_t nx, size_t ny, double theta_1, double phi_1, double theta, double phi)
{
  size_t len_gridding_data = nx * ny;
  double result = 0;
  gsl_interp2d *interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx, ny);
  gsl_interp_accel *xacc = gsl_interp_accel_alloc();
  gsl_interp_accel *yacc = gsl_interp_accel_alloc();
  for (int i = 0; i < len_coord; i++)
  {
    double *grid = gridding_data + (size_t)i * len_gridding_data;
    gsl_interp2d *interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx, ny);
    gsl_interp2d_init(interp, theta_list, phi_list, grid, nx, ny);

    double inter_1 = gsl_interp2d_eval(interp, theta_list, phi_list, grid, theta_1, phi_1, xacc, yacc);
    double inter_2 = gsl_interp2d_eval(interp, theta_list, phi_list, grid, theta, phi, xacc, yacc);

    result += absorption(inter_1, inter_2);
    gsl_interp_accel_reset(xacc);
    gsl_interp_accel_reset(yacc);
  }
  gsl_interp2d_free(interp);
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);
  return result / len_coord;
}

double interpolate_single(double *theta_list, double *phi_list, double *gridding_data, size_t nx, size_t ny, double theta_1, double phi_1)
{
  gsl_interp2d *interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx, ny);

  gsl_interp2d_init(interp, theta_list, phi_list, gridding_data, nx, ny);

  gsl_interp_accel *xacc = gsl_interp_accel_alloc();
  gsl_interp_accel *yacc = gsl_interp_accel_alloc();

  double zi = gsl_interp2d_eval(interp, theta_list, phi_list, gridding_data, theta_1, phi_1, xacc, yacc);

  gsl_interp2d_free(interp);
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);

  return zi;
}


double *nearest_neighbor_interpolate_overall(int64_t low, int64_t up,
                            int64_t *coord_list,
                            int64_t len_coord_list,
                            const double *scattering_vector_list, const double *omega_list,
                            const double *raw_xray,
                            const double *omega_axis, const double *kp_rotation_matrix,
                            int64_t len_result,
                            double *voxel_size, double *coefficients,
                             int num_workers,int IsExp, double *theta_list, double *phi_list, double *gridding_data, size_t nx, size_t ny,double theta_min, double theta_max, double phi_min, double phi_max,int interpolation_method)
{
    omp_set_num_threads(num_workers);

    printf("low is %d \n", low);
    printf("up is %d \n", up);
    double *result_list = (double *)malloc(len_result * sizeof(double));
    printf("result_list is %p \n", result_list);

#pragma omp parallel for default(none) shared(coord_list, scattering_vector_list, omega_list, raw_xray, omega_axis, kp_rotation_matrix, len_result, voxel_size, coefficients, low, up, len_coord_list, result_list,IsExp,gridding_data,theta_list,phi_list,nx,ny,theta_min,theta_max,phi_min,phi_max,interpolation_method)
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

        double x_ray_angle[3];
        double rotated_s1_angle[3];
        memcpy(x_ray_angle, xray, 3 * sizeof(xray));
        memcpy(rotated_s1_angle, rotated_s1, 3 * sizeof(rotated_s1));

        ThetaPhi result_2 = dials_2_thetaphi_22(rotated_s1_angle, 0);
        ThetaPhi result_1 = dials_2_thetaphi_22(x_ray_angle, 1);

        double theta = result_2.theta;
        double phi = result_2.phi;
        double theta_1 = result_1.theta;
        double phi_1 = result_1.phi;
        if (interpolation_method == 1){
          result = nearest_neighbor_interpolate(len_coord_list, theta_list, phi_list, gridding_data, nx, ny,   theta_1, phi_1, theta, phi, theta_min, theta_max, phi_min, phi_max);}

        else if (interpolation_method  ==2){
          result = interpolate(len_coord_list, theta_list, phi_list, gridding_data, nx, ny,   theta_1, phi_1, theta, phi);}
        
        result_list[i] = result;
        printf("[%d/%d] rotation: %.4f, absorption: %.4f\n",
               low + i, up, omega_list[i] * 180 / M_PI, result);
    }

    return result_list;
}


