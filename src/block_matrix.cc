#include "naive_block_ldlt/block_matrix.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace naive_block_ldlt {

BlockMatrix::BlockMatrix(int size, int block_size)
    : size_(size),
      block_size_(block_size),
      num_blocks_((size + block_size - 1) / block_size) {
  if (size <= 0) {
    throw std::invalid_argument("matrix size must be positive");
  }
  if (block_size <= 0) {
    throw std::invalid_argument("block size must be positive");
  }
  AllocateBlocks();
}

BlockMatrix::BlockMatrix(const RegularMatrix& matrix, int block_size,
                         MatrixTriangle input_triangle)
    : BlockMatrix(matrix.Rows(), block_size) {
  if (matrix.Rows() != matrix.Cols()) {
    throw std::invalid_argument("block matrix input must be square");
  }
  for (int i = 0; i < size_; ++i) {
    for (int j = i; j < size_; ++j) {
      (*this)(i, j) = input_triangle == MatrixTriangle::kUpper ? matrix(i, j)
                                                               : matrix(j, i);
    }
  }
}

BlockMatrix::BlockMatrix(const BlockMatrix& other)
    : size_(other.size_),
      block_size_(other.block_size_),
      num_blocks_(other.num_blocks_),
      row_capacities_(other.row_capacities_),
      col_capacities_(other.col_capacities_) {
  block_storage_.reserve(other.block_storage_.size());
  blocks_.reserve(other.blocks_.size());
  for (std::size_t block_id = 0; block_id < other.blocks_.size(); ++block_id) {
    const Block& other_block = other.blocks_[block_id];
    block_storage_.emplace_back(other.block_storage_[block_id].size());
    blocks_.emplace_back(
        block_storage_.back().data(), other_block.rows(), other_block.cols(),
        other_block.leading_dimension(), other_block.row0(),
        other_block.col0(), other_block.block_row(), other_block.block_col());
    std::copy(other.block_storage_[block_id].data(),
              other.block_storage_[block_id].data() +
                  other.block_storage_[block_id].size(),
              block_storage_.back().data());
  }
}

BlockMatrix& BlockMatrix::operator=(const BlockMatrix& other) {
  if (this == &other) {
    return *this;
  }
  BlockMatrix copy(other);
  *this = std::move(copy);
  return *this;
}

double& BlockMatrix::operator()(int row, int col) {
  if (row > col) {
    std::swap(row, col);
  }
  Block* block = FindBlockContaining(row, col);
  if (block == nullptr) {
    throw std::out_of_range("matrix index is outside active block views");
  }
  return (*block)(row - block->row0(), col - block->col0());
}

const double& BlockMatrix::operator()(int row, int col) const {
  if (row > col) {
    std::swap(row, col);
  }
  const Block* block = FindBlockContaining(row, col);
  if (block == nullptr) {
    throw std::out_of_range("matrix index is outside active block views");
  }
  return (*block)(row - block->row0(), col - block->col0());
}

int BlockMatrix::BlockId(int block_row, int block_col) const {
  if (block_row > block_col) {
    std::swap(block_row, block_col);
  }
  return block_row + block_col * (block_col + 1) / 2;
}

Block& BlockMatrix::GetBlock(int block_row, int block_col) {
  return blocks_.at(BlockId(block_row, block_col));
}

const Block& BlockMatrix::GetBlock(int block_row, int block_col) const {
  return blocks_.at(BlockId(block_row, block_col));
}

BlockColumnLocation BlockMatrix::GlobalColumnToBlockColumn(
    int block_row, int preferred_block_col, int column) const {
  const Block& preferred = GetBlock(block_row, preferred_block_col);
  if (preferred.col0() <= column &&
      column < preferred.col0() + preferred.cols()) {
    return {block_row, preferred_block_col, column - preferred.col0()};
  }

  for (int block_col = preferred_block_col - 1; block_col >= 0; --block_col) {
    const Block& block = GetBlock(block_row, block_col);
    if (block.col0() <= column && column < block.col0() + block.cols()) {
      return {block_row, block_col, column - block.col0()};
    }
  }

  for (int block_col = preferred_block_col + 1; block_col < num_blocks_;
       ++block_col) {
    const Block& block = GetBlock(block_row, block_col);
    if (block.col0() <= column && column < block.col0() + block.cols()) {
      return {block_row, block_col, column - block.col0()};
    }
  }

  throw std::out_of_range("global column is outside active block row");
}

RegularMatrix BlockMatrix::ToRegular() const {
  RegularMatrix regular(size_, size_);
  for (const Block& block : blocks_) {
    for (int i = 0; i < block.rows(); ++i) {
      for (int j = 0; j < block.cols(); ++j) {
        const int row = block.row0() + i;
        const int col = block.col0() + j;
        if (row < 0 || row >= size_ || col < 0 || col >= size_) {
          continue;
        }
        if (row > col) {
          continue;
        }
        regular(row, col) = block(i, j);
        regular(col, row) = block(i, j);
      }
    }
  }
  return regular;
}

RegularMatrix BlockMatrix::ToUpperRegular() const {
  RegularMatrix regular(size_, size_);
  for (const Block& block : blocks_) {
    for (int i = 0; i < block.rows(); ++i) {
      for (int j = 0; j < block.cols(); ++j) {
        const int row = block.row0() + i;
        const int col = block.col0() + j;
        if (row < 0 || row >= size_ || col < 0 || col >= size_) {
          continue;
        }
        if (row > col) {
          continue;
        }
        regular(row, col) = block(i, j);
      }
    }
  }
  return regular;
}

void BlockMatrix::ExpandBlock(int block_row, int block_col, int col_size) {
  const Block& block = GetBlock(block_row, block_col);
  if (block_row == block_col) {
    ResizeAndCopyBlock(block_row, block_col, block.rows() + 1,
                       block.cols() + col_size, col_size, 1,
                       block.row0() - 1, block.col0() - col_size);
  } else {
    ResizeAndCopyBlock(block_row, block_col, block.rows() + 1, block.cols(), 1,
                       0, block.row0() - 1, block.col0());
  }
}

void BlockMatrix::ShrinkBlock(int block_row, int block_col) {
  const Block& block = GetBlock(block_row, block_col);
  if (block_row == block_col) {
    if (block.rows() == 0 || block.cols() == 0) {
      return;
    }
    ResizeAndCopyBlock(block_row, block_col, block.rows() - 1,
                       block.cols() - 1, 0, 0, block.row0(), block.col0());
  } else {
    if (block.rows() == 0) {
      return;
    }
    ResizeAndCopyBlock(block_row, block_col, block.rows() - 1, block.cols(), 0,
                       0, block.row0(), block.col0());
  }
}

void BlockMatrix::AddColumnLeft(int block_row, int block_col, int col_size) {
  const Block& block = GetBlock(block_row, block_col);
  ResizeAndCopyBlock(block_row, block_col, block.rows(),
                     block.cols() + col_size, 0, col_size, block.row0(),
                     block.col0() - col_size);
}

void BlockMatrix::AllocateBlocks() {
  const int block_count = num_blocks_ * (num_blocks_ + 1) / 2;
  block_storage_.reserve(block_count);
  blocks_.reserve(block_count);
  row_capacities_.reserve(block_count);
  col_capacities_.reserve(block_count);
  for (int block_col = 0; block_col < num_blocks_; ++block_col) {
    const int col0 = block_col * block_size_;
    const int cols = std::min(block_size_, size_ - col0);
    const int col_capacity = cols + (block_col > 0 ? 1 : 0);
    for (int block_row = 0; block_row <= block_col; ++block_row) {
      const int row0 = block_row * block_size_;
      const int rows = std::min(block_size_, size_ - row0);
      const int row_capacity = rows + (block_row > 0 ? 1 : 0);
      row_capacities_.push_back(row_capacity);
      col_capacities_.push_back(col_capacity);
      block_storage_.emplace_back(static_cast<std::size_t>(row_capacity) *
                                  col_capacity);
      blocks_.emplace_back(block_storage_.back().data(), rows, cols,
                           col_capacity, row0, col0, block_row, block_col);
    }
  }
}

int BlockMatrix::BlockIndexForGlobalIndex(int index) const {
  if (index < 0 || index >= size_) {
    throw std::out_of_range("matrix index out of range");
  }
  return index / block_size_;
}

Block* BlockMatrix::FindBlockContaining(int row, int col) {
  for (Block& block : blocks_) {
    if (block.row0() <= row && row < block.row0() + block.rows() &&
        block.col0() <= col && col < block.col0() + block.cols()) {
      return &block;
    }
  }
  return nullptr;
}

const Block* BlockMatrix::FindBlockContaining(int row, int col) const {
  for (const Block& block : blocks_) {
    if (block.row0() <= row && row < block.row0() + block.rows() &&
        block.col0() <= col && col < block.col0() + block.cols()) {
      return &block;
    }
  }
  return nullptr;
}

void BlockMatrix::ResizeAndCopyBlock(int block_row, int block_col, int rows,
                                     int cols, int row_offset, int col_offset,
                                     int row0, int col0) {
  const int block_id = BlockId(block_row, block_col);
  const Block old_block = blocks_.at(block_id);

  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("block dimensions must be non-negative");
  }
  if (rows > row_capacities_.at(block_id) ||
      cols > col_capacities_.at(block_id)) {
    throw std::logic_error("block resize exceeds preallocated capacity");
  }

  double* data = block_storage_.at(block_id).data();
  const int leading_dimension = col_capacities_.at(block_id);
  const int row_begin = row_offset > 0 ? old_block.rows() - 1 : 0;
  const int row_end = row_offset > 0 ? -1 : old_block.rows();
  const int row_step = row_offset > 0 ? -1 : 1;
  const int col_begin = col_offset > 0 ? old_block.cols() - 1 : 0;
  const int col_end = col_offset > 0 ? -1 : old_block.cols();
  const int col_step = col_offset > 0 ? -1 : 1;

  for (int i = row_begin; i != row_end; i += row_step) {
    const int target_row = i + row_offset;
    if (target_row < 0 || target_row >= rows) {
      continue;
    }
    for (int j = col_begin; j != col_end; j += col_step) {
      const int target_col = j + col_offset;
      if (target_col < 0 || target_col >= cols) {
        continue;
      }
      data[target_row * leading_dimension + target_col] =
          data[i * old_block.leading_dimension() + j];
    }
  }

  for (int i = 0; i < rows; ++i) {
    const int old_row = i - row_offset;
    for (int j = 0; j < cols; ++j) {
      const int old_col = j - col_offset;
      if (old_row < 0 || old_row >= old_block.rows() || old_col < 0 ||
          old_col >= old_block.cols()) {
        data[i * leading_dimension + j] = 0.0;
      }
    }
  }

  Block& block = blocks_.at(block_id);
  block.Reset(data, rows, cols, leading_dimension, row0, col0, block_row,
              block_col);
}

}  // namespace naive_block_ldlt
