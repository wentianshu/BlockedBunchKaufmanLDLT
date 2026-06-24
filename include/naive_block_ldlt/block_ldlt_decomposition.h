#ifndef NAIVE_BLOCK_LDLT_BLOCK_LDLT_DECOMPOSITION_H_
#define NAIVE_BLOCK_LDLT_BLOCK_LDLT_DECOMPOSITION_H_

#include <vector>

#include "naive_block_ldlt/block_matrix.h"
#include "naive_block_ldlt/regular_matrix.h"

namespace naive_block_ldlt {

class BlockLdltDecomposition {
 public:
  enum class FactorizationMode {
    kFull,
    kCompactOnly,
  };

  explicit BlockLdltDecomposition(
      const BlockMatrix& matrix,
      FactorizationMode mode = FactorizationMode::kFull);

  void Factorize();
  RegularMatrix ReconstructOriginal() const;

  const RegularMatrix& LFactor() const { return l_factor_; }
  const RegularMatrix& DFactor() const { return d_factor_; }
  const BlockMatrix& CompactFactors() const { return factors_; }
  const BlockMatrix& Workspace() const { return workspace_; }
  const std::vector<int>& Permutation() const { return permutation_; }
  const std::vector<int>& PivotTypes() const { return pivot_types_; }

 private:
  struct PivotCandidate {
    double omega = 0.0;
    int block_col = 0;
    int local_col = 0;
    int global_col = 0;
  };

  void FactorPanel(int panel);
  void CopyCurrentRowToWorkspace(int panel, int local_pivot);
  void UpdateCurrentRowInWorkspace(int panel, int local_pivot);
  PivotCandidate FindCurrentRowMax(int panel, int local_pivot) const;
  void CopyCandidateToWorkspace(int panel, int local_pivot,
                                const PivotCandidate& candidate);
  void UpdateCandidateInWorkspace(int panel, int local_pivot,
                                  const PivotCandidate& candidate);
  double FindCandidateColumnMax(int pivot, int candidate) const;
  int SelectPivot(int panel, int local_pivot,
                  const PivotCandidate& candidate);
  void ShiftBoundaryColumnToNextBlock(int panel);
  void SwapPivot1x1(int panel, int local_pivot,
                    const PivotCandidate& candidate);
  void SwapPivot2x2(int panel, int local_pivot,
                    const PivotCandidate& candidate);
  void SwapPivot1x1InMatrix(BlockMatrix* matrix, int panel, int local_pivot,
                            const PivotCandidate& candidate,
                            bool swap_previous_block_rows);
  void SwapPivot2x2InMatrix(BlockMatrix* matrix, int panel, int local_pivot,
                            const PivotCandidate& candidate,
                            bool swap_previous_block_rows);
  void ScalePivot1x1(int panel, int local_pivot);
  void ScalePivot2x2(int panel, int local_pivot);
  void UpdateTrailingBlocks(int panel);
  void ExtractRegularFactorsFromCompact();
  double GetSymmetricGlobal(const BlockMatrix& matrix, int row, int col) const;
  bool IsTwoByTwoPivotCoupling(int row, int col) const;

  RegularMatrix original_;
  BlockMatrix factors_;
  BlockMatrix workspace_;
  RegularMatrix l_factor_;
  RegularMatrix d_factor_;
  std::vector<int> permutation_;
  std::vector<int> pivot_types_;
  FactorizationMode mode_ = FactorizationMode::kFull;
  bool factorized_ = false;
};

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_BLOCK_LDLT_DECOMPOSITION_H_
