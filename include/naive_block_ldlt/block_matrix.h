#ifndef NAIVE_BLOCK_LDLT_BLOCK_MATRIX_H_
#define NAIVE_BLOCK_LDLT_BLOCK_MATRIX_H_

#include <vector>

#include "naive_block_ldlt/aligned_buffer.h"
#include "naive_block_ldlt/block.h"
#include "naive_block_ldlt/matrix.h"
#include "naive_block_ldlt/regular_matrix.h"

namespace naive_block_ldlt {

struct BlockColumnLocation {
  int block_row = 0;
  int block_col = 0;
  int local_col = 0;
};

enum class MatrixTriangle {
  kUpper,
  kLower,
};

class BlockMatrix : public Matrix {
 public:
  BlockMatrix() = default;
  BlockMatrix(int size, int block_size);
  BlockMatrix(const RegularMatrix& matrix, int block_size,
              MatrixTriangle input_triangle = MatrixTriangle::kUpper);
  BlockMatrix(const BlockMatrix& other);
  BlockMatrix& operator=(const BlockMatrix& other);
  BlockMatrix(BlockMatrix&& other) noexcept = default;
  BlockMatrix& operator=(BlockMatrix&& other) noexcept = default;

  int Rows() const override { return size_; }
  int Cols() const override { return size_; }
  int block_size() const { return block_size_; }
  int num_blocks() const { return num_blocks_; }

  double& operator()(int row, int col) override;
  const double& operator()(int row, int col) const override;

  int BlockId(int block_row, int block_col) const;
  Block& GetBlock(int block_row, int block_col);
  const Block& GetBlock(int block_row, int block_col) const;
  BlockColumnLocation GlobalColumnToBlockColumn(int block_row,
                                                int preferred_block_col,
                                                int column) const;

  RegularMatrix ToRegular() const;
  RegularMatrix ToUpperRegular() const;

  void ExpandBlock(int block_row, int block_col, int col_size = 1);
  void ShrinkBlock(int block_row, int block_col);
  void AddColumnLeft(int block_row, int block_col, int col_size = 1);

 private:
  void AllocateBlocks();
  int BlockIndexForGlobalIndex(int index) const;
  Block* FindBlockContaining(int row, int col);
  const Block* FindBlockContaining(int row, int col) const;
  void ResizeAndCopyBlock(int block_row, int block_col, int rows, int cols,
                          int row_offset, int col_offset, int row0, int col0);

  int size_ = 0;
  int block_size_ = 0;
  int num_blocks_ = 0;
  std::vector<AlignedBuffer> block_storage_;
  std::vector<Block> blocks_;
  std::vector<int> row_capacities_;
  std::vector<int> col_capacities_;
};

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_BLOCK_MATRIX_H_
