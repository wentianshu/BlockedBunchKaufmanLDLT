#include "kernels.h"

#include <cmath>

#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
#include <mkl_cblas.h>
#endif

namespace naive_block_ldlt::kernels {

void ScaleOneByOnePivot(int count, double pivot, const double* source,
                        int source_stride, double* destination,
                        int destination_stride) {
  if (count <= 0) {
    return;
  }
  if (std::abs(pivot) == 0.0) {
    for (int i = 0; i < count; ++i) {
      destination[i * destination_stride] = 0.0;
    }
    return;
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  cblas_dcopy(count, source, source_stride, destination, destination_stride);
  cblas_dscal(count, 1.0 / pivot, destination, destination_stride);
#else
  for (int i = 0; i < count; ++i) {
    destination[i * destination_stride] = source[i * source_stride] / pivot;
  }
#endif
}

void ScaleTwoByTwoPivot(int count, double pivot_top_left,
                        double pivot_off_diagonal, double pivot_bottom_right,
                        const double* first_source, int first_source_stride,
                        const double* second_source, int second_source_stride,
                        double* first_destination,
                        int first_destination_stride,
                        double* second_destination,
                        int second_destination_stride) {
  if (count <= 0) {
    return;
  }
  const double determinant =
      pivot_top_left * pivot_bottom_right -
      pivot_off_diagonal * pivot_off_diagonal;
  if (std::abs(determinant) == 0.0) {
    for (int i = 0; i < count; ++i) {
      first_destination[i * first_destination_stride] = 0.0;
      second_destination[i * second_destination_stride] = 0.0;
    }
    return;
  }
  for (int i = 0; i < count; ++i) {
    const double first_value = first_source[i * first_source_stride];
    const double second_value = second_source[i * second_source_stride];
    first_destination[i * first_destination_stride] =
        (first_value * pivot_bottom_right -
         second_value * pivot_off_diagonal) /
        determinant;
    second_destination[i * second_destination_stride] =
        (-first_value * pivot_off_diagonal + second_value * pivot_top_left) /
        determinant;
  }
}

void UpdateTrailingBlock(int rows, int cols, int inner_dim,
                         const double* left_panel, int left_panel_stride,
                         const double* workspace_panel,
                         int workspace_panel_stride, double* target,
                         int target_stride) {
  if (rows <= 0 || cols <= 0 || inner_dim <= 0) {
    return;
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rows, cols, inner_dim,
              -1.0, left_panel, left_panel_stride, workspace_panel,
              workspace_panel_stride, 1.0, target, target_stride);
#else
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double update = 0.0;
      for (int k = 0; k < inner_dim; ++k) {
        update += left_panel[k * left_panel_stride + i] *
                  workspace_panel[k * workspace_panel_stride + j];
      }
      target[i * target_stride + j] -= update;
    }
  }
#endif
}

}  // namespace naive_block_ldlt::kernels
