#include "naive_block_ldlt/kernels.h"

#include <cmath>

#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
#include <mkl_cblas.h>
#endif

namespace naive_block_ldlt::kernels {

void ScalePivot1x1(int count, double pivot, const double* source,
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

void ScalePivot2x2(int count, double d00, double d01, double d11,
                   const double* source0, int source0_stride,
                   const double* source1, int source1_stride,
                   double* destination0, int destination0_stride,
                   double* destination1, int destination1_stride) {
  if (count <= 0) {
    return;
  }
  const double determinant = d00 * d11 - d01 * d01;
  if (std::abs(determinant) == 0.0) {
    for (int i = 0; i < count; ++i) {
      destination0[i * destination0_stride] = 0.0;
      destination1[i * destination1_stride] = 0.0;
    }
    return;
  }
  for (int i = 0; i < count; ++i) {
    const double value0 = source0[i * source0_stride];
    const double value1 = source1[i * source1_stride];
    destination0[i * destination0_stride] =
        (value0 * d11 - value1 * d01) / determinant;
    destination1[i * destination1_stride] =
        (-value0 * d01 + value1 * d00) / determinant;
  }
}

void UpdateTrailingBlock(int rows, int cols, int inner_dim, const double* u,
                         int ldu, const double* w, int ldw, double* a,
                         int lda) {
  if (rows <= 0 || cols <= 0 || inner_dim <= 0) {
    return;
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rows, cols, inner_dim,
              -1.0, u, ldu, w, ldw, 1.0, a, lda);
#else
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double update = 0.0;
      for (int k = 0; k < inner_dim; ++k) {
        update += u[k * ldu + i] * w[k * ldw + j];
      }
      a[i * lda + j] -= update;
    }
  }
#endif
}

}  // namespace naive_block_ldlt::kernels
