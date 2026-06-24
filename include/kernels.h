#ifndef NAIVE_BLOCK_LDLT_KERNELS_H_
#define NAIVE_BLOCK_LDLT_KERNELS_H_

namespace naive_block_ldlt::kernels {

void ScalePivot1x1(int count, double pivot, const double* source,
                   int source_stride, double* destination,
                   int destination_stride);

void ScalePivot2x2(int count, double d00, double d01, double d11,
                   const double* source0, int source0_stride,
                   const double* source1, int source1_stride,
                   double* destination0, int destination0_stride,
                   double* destination1, int destination1_stride);

void UpdateTrailingBlock(int rows, int cols, int inner_dim, const double* u,
                         int ldu, const double* w, int ldw, double* a,
                         int lda);

}  // namespace naive_block_ldlt::kernels

#endif  // NAIVE_BLOCK_LDLT_KERNELS_H_
