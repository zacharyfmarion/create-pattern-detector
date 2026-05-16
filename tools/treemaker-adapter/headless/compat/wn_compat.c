#include <float.h>

double wn_machine_tolerance(void) {
  return DBL_EPSILON;
}

void wn_scale_vect(double* vect, double scalar, int len) {
  for (int i = 0; i < len; ++i) vect[i] *= scalar;
}

void wn_add_scaled_vect(double* to_vect, double* from_vect, double scalar, int len) {
  for (int i = 0; i < len; ++i) to_vect[i] += scalar * from_vect[i];
}

void wn_multiply_vect_by_vect(double* v1, double* v2, int len) {
  for (int i = 0; i < len; ++i) v1[i] *= v2[i];
}

void wn_divide_vect_by_vect(double* v1, double* v2, int len) {
  for (int i = 0; i < len; ++i) v1[i] /= v2[i];
}
