#include "regular_matrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
#include <mkl_cblas.h>
#endif

namespace naive_block_ldlt {
namespace {

std::size_t CheckedElementCount(int rows, int cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("matrix shape must be nonnegative");
  }
  return static_cast<std::size_t>(rows) * cols;
}

}  // namespace

RegularMatrix::RegularMatrix(int rows, int cols)
    : rows_(rows),
      cols_(cols),
      leading_dimension_(cols),
      data_(CheckedElementCount(rows, cols)) {}

RegularMatrix::RegularMatrix(int rows, int cols,
                             const std::vector<double>& values)
    : RegularMatrix(rows, cols) {
  if (values.size() != static_cast<std::size_t>(rows) * cols) {
    throw std::invalid_argument("value count does not match matrix shape");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    Data()[i] = values[i];
  }
}

RegularMatrix::RegularMatrix(const RegularMatrix& other)
    : RegularMatrix(other.rows_, other.cols_) {
  std::copy(other.Data(), other.Data() + other.data_.size(), Data());
}

RegularMatrix& RegularMatrix::operator=(const RegularMatrix& other) {
  if (this == &other) {
    return *this;
  }
  rows_ = other.rows_;
  cols_ = other.cols_;
  leading_dimension_ = other.leading_dimension_;
  data_.Resize(other.data_.size());
  std::copy(other.Data(), other.Data() + other.data_.size(), Data());
  return *this;
}

double& RegularMatrix::operator()(int row, int col) {
  return data_.data()[row * leading_dimension_ + col];
}

const double& RegularMatrix::operator()(int row, int col) const {
  return data_.data()[row * leading_dimension_ + col];
}

void RegularMatrix::Fill(double value) { data_.Fill(value); }

RegularMatrix RegularMatrix::Identity(int size) {
  RegularMatrix identity(size, size);
  for (int i = 0; i < size; ++i) {
    identity(i, i) = 1.0;
  }
  return identity;
}

double FrobeniusNorm(const RegularMatrix& matrix) {
  if (matrix.Rows() == 0 || matrix.Cols() == 0) {
    return 0.0;
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  return cblas_dnrm2(matrix.Rows() * matrix.Cols(), matrix.Data(), 1);
#else
  double sum = 0.0;
  for (int i = 0; i < matrix.Rows(); ++i) {
    for (int j = 0; j < matrix.Cols(); ++j) {
      sum += matrix(i, j) * matrix(i, j);
    }
  }
  return std::sqrt(sum);
#endif
}

RegularMatrix Multiply(const RegularMatrix& left, const RegularMatrix& right) {
  if (left.Cols() != right.Rows()) {
    throw std::invalid_argument("incompatible matrix multiplication shapes");
  }
  RegularMatrix product(left.Rows(), right.Cols());
  if (left.Rows() == 0 || left.Cols() == 0 || right.Cols() == 0) {
    return product;
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, left.Rows(),
              right.Cols(), left.Cols(), 1.0, left.Data(),
              left.LeadingDimension(), right.Data(), right.LeadingDimension(),
              0.0, product.Data(), product.LeadingDimension());
#else
  for (int i = 0; i < left.Rows(); ++i) {
    for (int k = 0; k < left.Cols(); ++k) {
      const double left_value = left(i, k);
      for (int j = 0; j < right.Cols(); ++j) {
        product(i, j) += left_value * right(k, j);
      }
    }
  }
#endif
  return product;
}

RegularMatrix Transpose(const RegularMatrix& matrix) {
  RegularMatrix transpose(matrix.Cols(), matrix.Rows());
  for (int i = 0; i < matrix.Rows(); ++i) {
    for (int j = 0; j < matrix.Cols(); ++j) {
      transpose(j, i) = matrix(i, j);
    }
  }
  return transpose;
}

RegularMatrix Subtract(const RegularMatrix& left, const RegularMatrix& right) {
  if (left.Rows() != right.Rows() || left.Cols() != right.Cols()) {
    throw std::invalid_argument("incompatible matrix subtraction shapes");
  }
#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
  RegularMatrix difference(left);
  if (left.Rows() == 0 || left.Cols() == 0) {
    return difference;
  }
  cblas_daxpy(left.Rows() * left.Cols(), -1.0, right.Data(), 1,
              difference.Data(), 1);
#else
  RegularMatrix difference(left.Rows(), left.Cols());
  for (int i = 0; i < left.Rows(); ++i) {
    for (int j = 0; j < left.Cols(); ++j) {
      difference(i, j) = left(i, j) - right(i, j);
    }
  }
#endif
  return difference;
}

}  // namespace naive_block_ldlt
