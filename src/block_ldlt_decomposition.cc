#include "block_ldlt_decomposition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "kernels.h"

namespace naive_block_ldlt {
namespace {

constexpr double kAlpha = 0.6403882032022076;
constexpr int kNoPivot = 0;
constexpr int kCurrentPivot1x1 = 1;
constexpr int kSwappedPivot1x1 = -1;
constexpr int kPivot2x2 = 2;
constexpr int kMinimumParallelBlocks = 3;

double SafeDenominator(double value) {
  if (std::abs(value) <= std::numeric_limits<double>::epsilon()) {
    return 0.0;
  }
  return value;
}

}  // namespace

BlockLdltDecomposition::BlockLdltDecomposition(
    const BlockMatrix& matrix, FactorizationMode mode)
    : original_(mode == FactorizationMode::kFull ? matrix.ToRegular()
                                                 : RegularMatrix()),
      factors_(matrix),
      workspace_(matrix.Rows(), matrix.block_size()),
      l_factor_(mode == FactorizationMode::kFull
                    ? RegularMatrix::Identity(matrix.Rows())
                    : RegularMatrix()),
      d_factor_(mode == FactorizationMode::kFull
                    ? RegularMatrix(matrix.Rows(), matrix.Rows())
                    : RegularMatrix()),
      permutation_(matrix.Rows()),
      pivot_types_(matrix.Rows(), 0),
      mode_(mode) {
  for (int i = 0; i < matrix.Rows(); ++i) {
    permutation_[i] = i;
  }
}

void BlockLdltDecomposition::Factorize() {
  for (int panel = 0; panel < factors_.num_blocks(); ++panel) {
    FactorPanel(panel);
    UpdateTrailingBlocks(panel);
  }
  if (mode_ == FactorizationMode::kFull) {
    ExtractRegularFactorsFromCompact();
  }
  factorized_ = true;
}

RegularMatrix BlockLdltDecomposition::ReconstructOriginal() const {
  if (mode_ != FactorizationMode::kFull) {
    throw std::logic_error("ReconstructOriginal() requires full factor storage");
  }
  if (!factorized_) {
    throw std::logic_error(
        "Factorize() must be called before ReconstructOriginal()");
  }
  const RegularMatrix permuted =
      Multiply(Multiply(l_factor_, d_factor_), Transpose(l_factor_));
  RegularMatrix original(original_.Rows(), original_.Cols());
  for (int i = 0; i < original.Rows(); ++i) {
    for (int j = 0; j < original.Cols(); ++j) {
      original(permutation_[i], permutation_[j]) = permuted(i, j);
    }
  }
  return original;
}

void BlockLdltDecomposition::FactorPanel(int panel) {
  int local_pivot = 0;
  Block& diagonal_block = factors_.GetBlock(panel, panel);
  const int panel_cols = diagonal_block.cols();
  while (local_pivot < panel_cols) {
    CopyCurrentRowToWorkspace(panel, local_pivot);
    UpdateCurrentRowInWorkspace(panel, local_pivot);

    const PivotCandidate candidate = FindCurrentRowMax(panel, local_pivot);
    int pivot_size = 1;
    const double diagonal =
        std::abs(workspace_.GetBlock(panel, panel)(local_pivot, local_pivot));
    if (!(diagonal >= kAlpha * candidate.omega || candidate.omega == 0.0)) {
      CopyCandidateToWorkspace(panel, local_pivot, candidate);
      UpdateCandidateInWorkspace(panel, local_pivot, candidate);
      const int pivot_choice = SelectPivot(panel, local_pivot, candidate);

      if (pivot_choice == kNoPivot) {
        break;
      }
      pivot_size = std::abs(pivot_choice);
      if (pivot_choice == kSwappedPivot1x1) {
        SwapPivot1x1(panel, local_pivot, candidate);
      } else if (pivot_size == kPivot2x2) {
        if (local_pivot == diagonal_block.cols() - 1 &&
            panel + 1 < factors_.num_blocks()) {
          ShiftBoundaryColumnToNextBlock(panel);
          break;
        }
        SwapPivot2x2(panel, local_pivot, candidate);
      }
    }

    if (pivot_size == 1) {
      ScalePivot1x1(panel, local_pivot);
    } else {
      ScalePivot2x2(panel, local_pivot);
    }
    local_pivot += pivot_size;
  }
}

void BlockLdltDecomposition::CopyCurrentRowToWorkspace(int panel,
                                                       int local_pivot) {
  const int num_blocks = factors_.num_blocks();
#if defined(NAIVE_BLOCK_LDLT_USE_OPENMP)
  const bool parallel_blocks = num_blocks - panel > kMinimumParallelBlocks;
#pragma omp parallel for schedule(static) if (parallel_blocks)
#endif
  for (int block_col = panel; block_col < num_blocks; ++block_col) {
    const Block& source = factors_.GetBlock(panel, block_col);
    Block& destination = workspace_.GetBlock(panel, block_col);
    const int start_col = block_col == panel ? local_pivot : 0;
    for (int col = start_col; col < source.cols(); ++col) {
      destination(local_pivot, col) = source(local_pivot, col);
    }
  }
}

void BlockLdltDecomposition::UpdateCurrentRowInWorkspace(int panel,
                                                         int local_pivot) {
  if (local_pivot == 0) {
    return;
  }
  const int num_blocks = factors_.num_blocks();
  const Block& diagonal = factors_.GetBlock(panel, panel);
#if defined(NAIVE_BLOCK_LDLT_USE_OPENMP)
  const bool parallel_blocks = num_blocks - panel > kMinimumParallelBlocks;
#pragma omp parallel for schedule(static) if (parallel_blocks)
#endif
  for (int block_col = panel; block_col < num_blocks; ++block_col) {
    Block& workspace_block = workspace_.GetBlock(panel, block_col);
    const int start_col = block_col == panel ? local_pivot : 0;
    for (int col = start_col; col < workspace_block.cols(); ++col) {
      double update = 0.0;
      for (int previous = 0; previous < local_pivot; ++previous) {
        update += diagonal(previous, local_pivot) *
                  workspace_block(previous, col);
      }
      workspace_block(local_pivot, col) -= update;
    }
  }
}

BlockLdltDecomposition::PivotCandidate
BlockLdltDecomposition::FindCurrentRowMax(int panel, int local_pivot) const {
  PivotCandidate candidate;
  const int num_blocks = workspace_.num_blocks();
  for (int block_col = panel; block_col < num_blocks; ++block_col) {
    const Block& block = workspace_.GetBlock(panel, block_col);
    const int start_col = block_col == panel ? local_pivot + 1 : 0;
    for (int col = start_col; col < block.cols(); ++col) {
      const double value = std::abs(block(local_pivot, col));
      if (value > candidate.omega) {
        candidate.omega = value;
        candidate.block_col = block_col;
        candidate.local_col = col;
        candidate.global_col = block.col0() + col;
      }
    }
  }
  return candidate;
}

void BlockLdltDecomposition::CopyCandidateToWorkspace(
    int panel, int local_pivot, const PivotCandidate& candidate) {
  const int num_blocks = workspace_.num_blocks();
  const int candidate_block_col = candidate.block_col;
  const int candidate_col = candidate.local_col;
  if (panel == candidate_block_col) {
    const Block& source = factors_.GetBlock(panel, panel);
    Block& destination = workspace_.GetBlock(panel, panel);
    for (int row = local_pivot; row < candidate_col; ++row) {
      destination(row, candidate_col) = source(row, candidate_col);
    }
    for (int col = candidate_col; col < source.cols(); ++col) {
      destination(candidate_col, col) = source(candidate_col, col);
    }
    for (int block_col = panel + 1; block_col < num_blocks; ++block_col) {
      const Block& source_block = factors_.GetBlock(panel, block_col);
      Block& destination_block = workspace_.GetBlock(panel, block_col);
      for (int col = 0; col < source_block.cols(); ++col) {
        destination_block(candidate_col, col) = source_block(candidate_col, col);
      }
    }
    return;
  }

  const Block& panel_candidate_source =
      factors_.GetBlock(panel, candidate_block_col);
  Block& panel_candidate_destination =
      workspace_.GetBlock(panel, candidate_block_col);
  for (int row = local_pivot; row < panel_candidate_source.rows(); ++row) {
    panel_candidate_destination(row, candidate_col) =
        panel_candidate_source(row, candidate_col);
  }

  for (int block_row = panel + 1; block_row < candidate_block_col;
       ++block_row) {
    const Block& source = factors_.GetBlock(block_row, candidate_block_col);
    Block& destination = workspace_.GetBlock(block_row, candidate_block_col);
    for (int row = 0; row < source.rows(); ++row) {
      destination(row, candidate_col) = source(row, candidate_col);
    }
  }

  const Block& diagonal_source =
      factors_.GetBlock(candidate_block_col, candidate_block_col);
  Block& diagonal_destination =
      workspace_.GetBlock(candidate_block_col, candidate_block_col);
  for (int row = 0; row < candidate_col; ++row) {
    diagonal_destination(row, candidate_col) =
        diagonal_source(row, candidate_col);
  }
  for (int col = candidate_col; col < diagonal_source.cols(); ++col) {
    diagonal_destination(candidate_col, col) =
        diagonal_source(candidate_col, col);
  }

  for (int block_col = candidate_block_col + 1; block_col < num_blocks;
       ++block_col) {
    const Block& source = factors_.GetBlock(candidate_block_col, block_col);
    Block& destination = workspace_.GetBlock(candidate_block_col, block_col);
    for (int col = 0; col < source.cols(); ++col) {
      destination(candidate_col, col) = source(candidate_col, col);
    }
  }
}

void BlockLdltDecomposition::UpdateCandidateInWorkspace(
    int panel, int local_pivot, const PivotCandidate& candidate) {
  if (local_pivot == 0) {
    return;
  }

  const int num_blocks = workspace_.num_blocks();
  const int candidate_block_col = candidate.block_col;
  const int candidate_col = candidate.local_col;
  const Block& factor_panel_candidate =
      factors_.GetBlock(panel, candidate_block_col);

  if (panel == candidate_block_col) {
    Block& diagonal_workspace = workspace_.GetBlock(panel, panel);
    for (int row = local_pivot; row < candidate_col; ++row) {
      double update = 0.0;
      for (int previous = 0; previous < local_pivot; ++previous) {
        update += factor_panel_candidate(previous, candidate_col) *
                  diagonal_workspace(previous, row);
      }
      diagonal_workspace(row, candidate_col) -= update;
    }
    for (int col = candidate_col; col < diagonal_workspace.cols(); ++col) {
      double update = 0.0;
      for (int previous = 0; previous < local_pivot; ++previous) {
        update += factor_panel_candidate(previous, candidate_col) *
                  diagonal_workspace(previous, col);
      }
      diagonal_workspace(candidate_col, col) -= update;
    }
    for (int block_col = panel + 1; block_col < num_blocks; ++block_col) {
      Block& workspace_block = workspace_.GetBlock(panel, block_col);
      for (int col = 0; col < workspace_block.cols(); ++col) {
        double update = 0.0;
        for (int previous = 0; previous < local_pivot; ++previous) {
          update += factor_panel_candidate(previous, candidate_col) *
                    workspace_block(previous, col);
        }
        workspace_block(candidate_col, col) -= update;
      }
    }
    return;
  }

  Block& panel_candidate_workspace =
      workspace_.GetBlock(panel, candidate_block_col);
  const Block& diagonal_workspace = workspace_.GetBlock(panel, panel);
  for (int row = local_pivot; row < panel_candidate_workspace.rows(); ++row) {
    double update = 0.0;
    for (int previous = 0; previous < local_pivot; ++previous) {
      update += factor_panel_candidate(previous, candidate_col) *
                diagonal_workspace(previous, row);
    }
    panel_candidate_workspace(row, candidate_col) -= update;
  }

  for (int block_row = panel + 1; block_row < candidate_block_col;
       ++block_row) {
    Block& target = workspace_.GetBlock(block_row, candidate_block_col);
    const Block& source = workspace_.GetBlock(panel, block_row);
    for (int row = 0; row < target.rows(); ++row) {
      double update = 0.0;
      for (int previous = 0; previous < local_pivot; ++previous) {
        update += factor_panel_candidate(previous, candidate_col) *
                  source(previous, row);
      }
      target(row, candidate_col) -= update;
    }
  }

  Block& candidate_diagonal_workspace =
      workspace_.GetBlock(candidate_block_col, candidate_block_col);
  for (int row = 0; row < candidate_col; ++row) {
    double update = 0.0;
    for (int previous = 0; previous < local_pivot; ++previous) {
      update += factor_panel_candidate(previous, candidate_col) *
                panel_candidate_workspace(previous, row);
    }
    candidate_diagonal_workspace(row, candidate_col) -= update;
  }
  for (int col = candidate_col; col < candidate_diagonal_workspace.cols();
       ++col) {
    double update = 0.0;
    for (int previous = 0; previous < local_pivot; ++previous) {
      update += factor_panel_candidate(previous, candidate_col) *
                panel_candidate_workspace(previous, col);
    }
    candidate_diagonal_workspace(candidate_col, col) -= update;
  }

  for (int block_col = candidate_block_col + 1; block_col < num_blocks;
       ++block_col) {
    Block& target = workspace_.GetBlock(candidate_block_col, block_col);
    const Block& source = workspace_.GetBlock(panel, block_col);
    for (int col = 0; col < target.cols(); ++col) {
      double update = 0.0;
      for (int previous = 0; previous < local_pivot; ++previous) {
        update += factor_panel_candidate(previous, candidate_col) *
                  source(previous, col);
      }
      target(candidate_col, col) -= update;
    }
  }
}

double BlockLdltDecomposition::FindCandidateColumnMax(int pivot,
                                                      int candidate) const {
  double omega = 0.0;
  for (int active_index = pivot; active_index < workspace_.Rows();
       ++active_index) {
    if (active_index == candidate) {
      continue;
    }
    omega = std::max(
        omega, std::abs(GetSymmetricGlobal(workspace_, active_index,
                                           candidate)));
  }
  return omega;
}

int BlockLdltDecomposition::SelectPivot(
    int panel, int local_pivot, const PivotCandidate& candidate) {
  const Block& diagonal_block = workspace_.GetBlock(panel, panel);
  const int pivot = diagonal_block.col0() + local_pivot;
  const double absajj = std::abs(diagonal_block(local_pivot, local_pivot));
  const double omega_r = FindCandidateColumnMax(pivot, candidate.global_col);
  const double absarr =
      std::abs(workspace_.GetBlock(candidate.block_col, candidate.block_col)(
          candidate.local_col, candidate.local_col));

  if (absajj * omega_r - kAlpha * candidate.omega * candidate.omega >= 0.0) {
    return kCurrentPivot1x1;
  }
  if (absarr - kAlpha * omega_r >= 0.0) {
    return kSwappedPivot1x1;
  }
  return kPivot2x2;
}

void BlockLdltDecomposition::SwapPivot1x1(
    int panel, int local_pivot, const PivotCandidate& candidate) {
  SwapPivot1x1InMatrix(&factors_, panel, local_pivot, candidate,
                       /*swap_previous_block_rows=*/true);
  SwapPivot1x1InMatrix(&workspace_, panel, local_pivot, candidate,
                       /*swap_previous_block_rows=*/false);

  const int pivot = factors_.GetBlock(panel, panel).col0() + local_pivot;
  std::swap(permutation_[pivot], permutation_[candidate.global_col]);
}

void BlockLdltDecomposition::SwapPivot2x2(
    int panel, int local_pivot, const PivotCandidate& candidate) {
  SwapPivot2x2InMatrix(&factors_, panel, local_pivot, candidate,
                       /*swap_previous_block_rows=*/true);
  SwapPivot2x2InMatrix(&workspace_, panel, local_pivot, candidate,
                       /*swap_previous_block_rows=*/false);

  const int pivot = factors_.GetBlock(panel, panel).col0() + local_pivot;
  std::swap(permutation_[pivot + 1], permutation_[candidate.global_col]);
}

void BlockLdltDecomposition::SwapPivot1x1InMatrix(
    BlockMatrix* matrix, int panel, int local_pivot,
    const PivotCandidate& candidate, bool swap_previous_block_rows) {
  const int candidate_block_col = candidate.block_col;
  const int candidate_col = candidate.local_col;
  const int num_blocks = matrix->num_blocks();
  const int panel_cols = matrix->GetBlock(panel, panel).cols();
  const int pivot = matrix->GetBlock(panel, panel).col0() + local_pivot;

  std::swap((*matrix).GetBlock(panel, panel)(local_pivot, local_pivot),
            (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                candidate_col, candidate_col));

  const int candidate_panel_cols =
      matrix->GetBlock(panel, candidate_block_col).cols();
  for (int col = candidate_col + 1; col < candidate_panel_cols; ++col) {
    std::swap((*matrix).GetBlock(panel, candidate_block_col)(local_pivot, col),
              (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                  candidate_col, col));
  }
  for (int block_col = candidate_block_col + 1; block_col < num_blocks;
       ++block_col) {
    const int block_cols = matrix->GetBlock(panel, block_col).cols();
    for (int col = 0; col < block_cols; ++col) {
      std::swap((*matrix).GetBlock(panel, block_col)(local_pivot, col),
                (*matrix).GetBlock(candidate_block_col, block_col)(
                    candidate_col, col));
    }
  }

  if (panel == candidate_block_col) {
    for (int col = local_pivot + 1; col < candidate_col; ++col) {
      std::swap((*matrix).GetBlock(panel, panel)(local_pivot, col),
                (*matrix).GetBlock(panel, panel)(col, candidate_col));
    }
  } else {
    for (int col = local_pivot + 1; col < panel_cols; ++col) {
      std::swap((*matrix).GetBlock(panel, panel)(local_pivot, col),
                (*matrix).GetBlock(panel, candidate_block_col)(col,
                                                               candidate_col));
    }
    for (int block_col = panel + 1; block_col < candidate_block_col;
         ++block_col) {
      const int block_cols = matrix->GetBlock(panel, block_col).cols();
      for (int col = 0; col < block_cols; ++col) {
        std::swap((*matrix).GetBlock(panel, block_col)(local_pivot, col),
                  (*matrix).GetBlock(block_col, candidate_block_col)(
                      col, candidate_col));
      }
    }
    for (int col = 0; col < candidate_col; ++col) {
      std::swap((*matrix).GetBlock(panel, candidate_block_col)(local_pivot,
                                                               col),
                (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                    col, candidate_col));
    }
  }

  if (swap_previous_block_rows) {
    for (int block_row = 0; block_row < panel; ++block_row) {
      const BlockColumnLocation pivot_location =
          matrix->GlobalColumnToBlockColumn(block_row, panel, pivot);
      const BlockColumnLocation candidate_location =
          matrix->GlobalColumnToBlockColumn(block_row, candidate_block_col,
                                            candidate.global_col);
      const int rows =
          matrix->GetBlock(pivot_location.block_row, pivot_location.block_col)
              .rows();
      for (int row = 0; row < rows; ++row) {
        std::swap((*matrix).GetBlock(pivot_location.block_row,
                                     pivot_location.block_col)(
                      row, pivot_location.local_col),
                  (*matrix).GetBlock(candidate_location.block_row,
                                     candidate_location.block_col)(
                      row, candidate_location.local_col));
      }
    }
  }

  for (int row = 0; row < local_pivot; ++row) {
    std::swap((*matrix).GetBlock(panel, panel)(row, local_pivot),
              (*matrix).GetBlock(panel, candidate_block_col)(row,
                                                             candidate_col));
  }
}

void BlockLdltDecomposition::SwapPivot2x2InMatrix(
    BlockMatrix* matrix, int panel, int local_pivot,
    const PivotCandidate& candidate, bool swap_previous_block_rows) {
  const int candidate_block_col = candidate.block_col;
  const int candidate_col = candidate.local_col;
  const int num_blocks = matrix->num_blocks();
  const int panel_cols = matrix->GetBlock(panel, panel).cols();
  const int pivot = matrix->GetBlock(panel, panel).col0() + local_pivot;
  const int next = local_pivot + 1;

  std::swap((*matrix).GetBlock(panel, panel)(next, next),
            (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                candidate_col, candidate_col));
  std::swap((*matrix).GetBlock(panel, panel)(local_pivot, next),
            (*matrix).GetBlock(panel, candidate_block_col)(local_pivot,
                                                           candidate_col));

  const int candidate_panel_cols =
      matrix->GetBlock(panel, candidate_block_col).cols();
  for (int col = candidate_col + 1; col < candidate_panel_cols; ++col) {
    std::swap((*matrix).GetBlock(panel, candidate_block_col)(next, col),
              (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                  candidate_col, col));
  }
  for (int block_col = candidate_block_col + 1; block_col < num_blocks;
       ++block_col) {
    const int block_cols = matrix->GetBlock(panel, block_col).cols();
    for (int col = 0; col < block_cols; ++col) {
      std::swap((*matrix).GetBlock(panel, block_col)(next, col),
                (*matrix).GetBlock(candidate_block_col, block_col)(
                    candidate_col, col));
    }
  }

  if (panel == candidate_block_col) {
    for (int col = local_pivot + 2; col < candidate_col; ++col) {
      std::swap((*matrix).GetBlock(panel, panel)(next, col),
                (*matrix).GetBlock(panel, panel)(col, candidate_col));
    }
  } else {
    for (int col = local_pivot + 2; col < panel_cols; ++col) {
      std::swap((*matrix).GetBlock(panel, panel)(next, col),
                (*matrix).GetBlock(panel, candidate_block_col)(col,
                                                               candidate_col));
    }
    for (int block_col = panel + 1; block_col < candidate_block_col;
         ++block_col) {
      const int block_cols = matrix->GetBlock(panel, block_col).cols();
      for (int col = 0; col < block_cols; ++col) {
        std::swap((*matrix).GetBlock(panel, block_col)(next, col),
                  (*matrix).GetBlock(block_col, candidate_block_col)(
                      col, candidate_col));
      }
    }
    for (int col = 0; col < candidate_col; ++col) {
      std::swap((*matrix).GetBlock(panel, candidate_block_col)(next, col),
                (*matrix).GetBlock(candidate_block_col, candidate_block_col)(
                    col, candidate_col));
    }
  }

  if (swap_previous_block_rows) {
    for (int block_row = 0; block_row < panel; ++block_row) {
      const BlockColumnLocation next_location =
          matrix->GlobalColumnToBlockColumn(block_row, panel, pivot + 1);
      const BlockColumnLocation candidate_location =
          matrix->GlobalColumnToBlockColumn(block_row, candidate_block_col,
                                            candidate.global_col);
      const int rows =
          matrix->GetBlock(next_location.block_row, next_location.block_col)
              .rows();
      for (int row = 0; row < rows; ++row) {
        std::swap((*matrix).GetBlock(next_location.block_row,
                                     next_location.block_col)(
                      row, next_location.local_col),
                  (*matrix).GetBlock(candidate_location.block_row,
                                     candidate_location.block_col)(
                      row, candidate_location.local_col));
      }
    }
  }

  for (int row = 0; row < local_pivot; ++row) {
    std::swap((*matrix).GetBlock(panel, panel)(row, next),
              (*matrix).GetBlock(panel, candidate_block_col)(row,
                                                             candidate_col));
  }
}

void BlockLdltDecomposition::ShiftBoundaryColumnToNextBlock(int panel) {
  const int next_panel = panel + 1;
  const int num_blocks = factors_.num_blocks();

  for (int block_col = next_panel; block_col < num_blocks; ++block_col) {
    factors_.ExpandBlock(next_panel, block_col);
  }
  factors_.AddColumnLeft(panel, next_panel);
  Block& factor_bridge = factors_.GetBlock(panel, next_panel);
  const Block& factor_diagonal = factors_.GetBlock(panel, panel);
  for (int row = 0; row < factor_bridge.rows(); ++row) {
    factor_bridge(row, 0) = factor_diagonal(row, factor_diagonal.cols() - 1);
  }
  for (int block_col = next_panel; block_col < num_blocks; ++block_col) {
    const Block& top = factors_.GetBlock(panel, block_col);
    Block& shifted = factors_.GetBlock(next_panel, block_col);
    for (int col = 0; col < top.cols(); ++col) {
      shifted(0, col) = top(top.rows() - 1, col);
    }
    factors_.ShrinkBlock(panel, block_col);
  }
  factors_.ShrinkBlock(panel, panel);

  for (int block_col = next_panel; block_col < num_blocks; ++block_col) {
    workspace_.ExpandBlock(next_panel, block_col);
  }
  workspace_.AddColumnLeft(panel, next_panel);
  Block& workspace_bridge = workspace_.GetBlock(panel, next_panel);
  const Block& workspace_diagonal = workspace_.GetBlock(panel, panel);
  for (int row = 0; row < workspace_bridge.rows(); ++row) {
    workspace_bridge(row, 0) =
        workspace_diagonal(row, workspace_diagonal.cols() - 1);
  }
  for (int block_col = next_panel; block_col < num_blocks; ++block_col) {
    const Block& top = workspace_.GetBlock(panel, block_col);
    Block& shifted = workspace_.GetBlock(next_panel, block_col);
    for (int col = 0; col < top.cols(); ++col) {
      shifted(0, col) = top(top.rows() - 1, col);
    }
    workspace_.ShrinkBlock(panel, block_col);
  }
  workspace_.ShrinkBlock(panel, panel);
}

void BlockLdltDecomposition::ScalePivot1x1(int panel, int local_pivot) {
  const int num_blocks = factors_.num_blocks();
  const Block& diagonal_block = factors_.GetBlock(panel, panel);
  const int pivot = diagonal_block.col0() + local_pivot;
  pivot_types_[pivot] = 1;
  const double diagonal =
      workspace_.GetBlock(panel, panel)(local_pivot, local_pivot);
  const double denominator = SafeDenominator(diagonal);

#if defined(NAIVE_BLOCK_LDLT_USE_OPENMP)
  const bool parallel_blocks = num_blocks - panel > kMinimumParallelBlocks;
#pragma omp parallel for schedule(static) if (parallel_blocks)
#endif
  for (int block_col = panel; block_col < num_blocks; ++block_col) {
    Block& target = factors_.GetBlock(panel, block_col);
    const Block& source = workspace_.GetBlock(panel, block_col);
    const int start_col = block_col == panel ? local_pivot + 1 : 0;
    for (int col = start_col; col < target.cols(); ++col) {
      target(local_pivot, col) =
          denominator == 0.0 ? 0.0 : source(local_pivot, col) / denominator;
    }
  }
  factors_.GetBlock(panel, panel)(local_pivot, local_pivot) = diagonal;
}

void BlockLdltDecomposition::ScalePivot2x2(int panel, int local_pivot) {
  const int num_blocks = factors_.num_blocks();
  const Block& diagonal_block = factors_.GetBlock(panel, panel);
  const int pivot = diagonal_block.col0() + local_pivot;
  const int next = pivot + 1;
  const int local_next = local_pivot + 1;
  pivot_types_[pivot] = 2;
  pivot_types_[next] = -2;

  const Block& workspace_diagonal = workspace_.GetBlock(panel, panel);
  const double d00 = workspace_diagonal(local_pivot, local_pivot);
  const double d01 = workspace_diagonal(local_pivot, local_next);
  const double d11 = workspace_diagonal(local_next, local_next);
  const double determinant = d00 * d11 - d01 * d01;

  const double inv_w00 = d00 / determinant;
  const double inv_w11 = d11 / determinant;
  const double inv_w01 = -d01 / determinant;
  const double inv_w10 = inv_w01;

#if defined(NAIVE_BLOCK_LDLT_USE_OPENMP)
  const bool parallel_blocks = num_blocks - panel > kMinimumParallelBlocks;
#pragma omp parallel for schedule(static) if (parallel_blocks)
#endif
  for (int block_col = panel; block_col < num_blocks; ++block_col) {
    Block& target = factors_.GetBlock(panel, block_col);
    const Block& source = workspace_.GetBlock(panel, block_col);
    const int start_col =
        block_col == panel && local_pivot < diagonal_block.cols() - 2
            ? local_pivot + 2
            : 0;
    for (int col = start_col; col < target.cols(); ++col) {
      target(local_pivot, col) =
          inv_w11 * source(local_pivot, col) + inv_w10 * source(local_next, col);
      target(local_next, col) =
          inv_w01 * source(local_pivot, col) + inv_w00 * source(local_next, col);
    }
  }

  Block& factor_diagonal = factors_.GetBlock(panel, panel);
  factor_diagonal(local_pivot, local_pivot) = d00;
  factor_diagonal(local_pivot, local_next) = d01;
  factor_diagonal(local_next, local_next) = d11;
}

void BlockLdltDecomposition::UpdateTrailingBlocks(int panel) {
  const int num_blocks = factors_.num_blocks();
  const int first_trailing_block = panel + 1;
  const int trailing_block_count = num_blocks - first_trailing_block;
  const int task_count =
      trailing_block_count * (trailing_block_count + 1) / 2;

#if defined(NAIVE_BLOCK_LDLT_USE_OPENMP)
  const bool parallel_tasks = task_count > kMinimumParallelBlocks;
  const int update_task_chunk_size = std::max(1, trailing_block_count);
  // Each task owns one trailing A block; panel factors and W are read-only here.
#pragma omp parallel for schedule(dynamic, update_task_chunk_size) if (parallel_tasks)
#endif
  for (int task = 0; task < task_count; ++task) {
    int relative_row = 0;
    int first_task_for_row = 0;
    int row_task_count = trailing_block_count;
    while (task >= first_task_for_row + row_task_count) {
      first_task_for_row += row_task_count;
      --row_task_count;
      ++relative_row;
    }
    const int relative_col = relative_row + task - first_task_for_row;
    const int block_row = first_trailing_block + relative_row;
    const int block_col = first_trailing_block + relative_col;

    const Block& u_panel = factors_.GetBlock(panel, block_row);
    const Block& w_panel = workspace_.GetBlock(panel, block_col);
    Block& target = factors_.GetBlock(block_row, block_col);
    if (u_panel.rows() == 0 || u_panel.cols() == 0 || w_panel.rows() == 0 ||
        w_panel.cols() == 0 || target.rows() == 0 || target.cols() == 0) {
      continue;
    }
    kernels::UpdateTrailingBlock(
        target.rows(), target.cols(), u_panel.rows(), u_panel.Data(),
        u_panel.leading_dimension(), w_panel.Data(),
        w_panel.leading_dimension(), target.Data(), target.leading_dimension());
  }
}

void BlockLdltDecomposition::ExtractRegularFactorsFromCompact() {
  l_factor_.Fill(0.0);
  d_factor_.Fill(0.0);

  for (int row = 0; row < factors_.Rows(); ++row) {
    l_factor_(row, row) = 1.0;
    d_factor_(row, row) = GetSymmetricGlobal(factors_, row, row);
    for (int col = row + 1; col < factors_.Cols(); ++col) {
      const double value = GetSymmetricGlobal(factors_, row, col);
      if (IsTwoByTwoPivotCoupling(row, col)) {
        d_factor_(row, col) = value;
        d_factor_(col, row) = value;
      } else {
        l_factor_(col, row) = value;
      }
    }
  }
}

double BlockLdltDecomposition::GetSymmetricGlobal(const BlockMatrix& matrix,
                                                  int row, int col) const {
  if (row > col) {
    std::swap(row, col);
  }
  return matrix(row, col);
}

bool BlockLdltDecomposition::IsTwoByTwoPivotCoupling(int row, int col) const {
  return col == row + 1 && pivot_types_[row] == 2;
}

}  // namespace naive_block_ldlt
