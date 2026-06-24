#ifndef NAIVE_BLOCK_LDLT_BLOCK_H_
#define NAIVE_BLOCK_LDLT_BLOCK_H_

namespace naive_block_ldlt {

class Block {
 public:
  Block() = default;

  Block(double* data, int rows, int cols, int leading_dimension, int row0,
        int col0, int block_row, int block_col)
      : data_(data),
        rows_(rows),
        cols_(cols),
        leading_dimension_(leading_dimension),
        row0_(row0),
        col0_(col0),
        block_row_(block_row),
        block_col_(block_col) {}

  double& operator()(int row, int col) {
    return data_[row * leading_dimension_ + col];
  }

  const double& operator()(int row, int col) const {
    return data_[row * leading_dimension_ + col];
  }

  double* Data() { return data_; }
  const double* Data() const { return data_; }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int leading_dimension() const { return leading_dimension_; }
  int row0() const { return row0_; }
  int col0() const { return col0_; }
  int block_row() const { return block_row_; }
  int block_col() const { return block_col_; }

  void Reset(double* data, int rows, int cols, int leading_dimension, int row0,
             int col0, int block_row, int block_col) {
    data_ = data;
    rows_ = rows;
    cols_ = cols;
    leading_dimension_ = leading_dimension;
    row0_ = row0;
    col0_ = col0;
    block_row_ = block_row;
    block_col_ = block_col;
  }

 private:
  double* data_ = nullptr;
  int rows_ = 0;
  int cols_ = 0;
  int leading_dimension_ = 0;
  int row0_ = 0;
  int col0_ = 0;
  int block_row_ = 0;
  int block_col_ = 0;
};

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_BLOCK_H_
