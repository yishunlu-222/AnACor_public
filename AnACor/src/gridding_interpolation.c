#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double absorption(double inter_1, double inter_2)
{
  return exp(-(inter_1 + inter_2));
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
