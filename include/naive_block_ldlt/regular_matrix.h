#ifndef NAIVE_BLOCK_LDLT_REGULAR_MATRIX_H_
#define NAIVE_BLOCK_LDLT_REGULAR_MATRIX_H_

#include <vector>

#include "naive_block_ldlt/aligned_buffer.h"
#include "naive_block_ldlt/matrix.h"

namespace naive_block_ldlt {

class RegularMatrix : public Matrix {
 public:
  RegularMatrix() = default;
  RegularMatrix(int rows, int cols);
  RegularMatrix(int rows, int cols, const std::vector<double>& values);

  RegularMatrix(const RegularMatrix& other);
  RegularMatrix& operator=(const RegularMatrix& other);
  RegularMatrix(RegularMatrix&& other) noexcept = default;
  RegularMatrix& operator=(RegularMatrix&& other) noexcept = default;

  int Rows() const override { return rows_; }
  int Cols() const override { return cols_; }
  int LeadingDimension() const { return leading_dimension_; }

  double* Data() { return data_.data(); }
  const double* Data() const { return data_.data(); }

  double& operator()(int row, int col) override;
  const double& operator()(int row, int col) const override;

  void Fill(double value);
  static RegularMatrix Identity(int size);

 private:
  int rows_ = 0;
  int cols_ = 0;
  int leading_dimension_ = 0;
  AlignedBuffer data_;
};

double FrobeniusNorm(const RegularMatrix& matrix);
RegularMatrix Multiply(const RegularMatrix& left, const RegularMatrix& right);
RegularMatrix Transpose(const RegularMatrix& matrix);
RegularMatrix Subtract(const RegularMatrix& left, const RegularMatrix& right);

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_REGULAR_MATRIX_H_
