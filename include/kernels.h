#ifndef NAIVE_BLOCK_LDLT_KERNELS_H_
#define NAIVE_BLOCK_LDLT_KERNELS_H_

namespace naive_block_ldlt::kernels {

void ScaleOneByOnePivot(int count, double pivot, const double* source,
                        int source_stride, double* destination,
                        int destination_stride);

void ScaleTwoByTwoPivot(int count, double pivot_top_left,
                        double pivot_off_diagonal, double pivot_bottom_right,
                        const double* first_source, int first_source_stride,
                        const double* second_source, int second_source_stride,
                        double* first_destination,
                        int first_destination_stride,
                        double* second_destination,
                        int second_destination_stride);

void UpdateTrailingBlock(int rows, int cols, int inner_dim,
                         const double* left_panel, int left_panel_stride,
                         const double* workspace_panel,
                         int workspace_panel_stride, double* target,
                         int target_stride);

}  // namespace naive_block_ldlt::kernels

#endif  // NAIVE_BLOCK_LDLT_KERNELS_H_
