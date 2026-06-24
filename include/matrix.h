#ifndef NAIVE_BLOCK_LDLT_MATRIX_H_
#define NAIVE_BLOCK_LDLT_MATRIX_H_

namespace naive_block_ldlt {

class Matrix {
 public:
  virtual ~Matrix() = default;

  virtual int Rows() const = 0;
  virtual int Cols() const = 0;

  virtual double& operator()(int row, int col) = 0;
  virtual const double& operator()(int row, int col) const = 0;
};

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_MATRIX_H_
