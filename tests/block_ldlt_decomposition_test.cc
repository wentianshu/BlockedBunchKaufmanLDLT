#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
#include <mkl_lapacke.h>
#endif

#include "block_ldlt_decomposition.h"
#include "block_matrix.h"
#include "kernels.h"
#include "regular_matrix.h"

namespace {

using naive_block_ldlt::BlockLdltDecomposition;
using naive_block_ldlt::BlockMatrix;
using naive_block_ldlt::FrobeniusNorm;
using naive_block_ldlt::MatrixTriangle;
using naive_block_ldlt::RegularMatrix;
using naive_block_ldlt::Subtract;

constexpr double kTolerance = 1.0e-10;

struct StressCase {
  int size = 0;
  int block_size = 0;
  double value_scale = 1.0;
  int seed = 0;
};

const std::vector<int>& StandardStressSizes() {
  static const std::vector<int> sizes = {5, 10, 20, 50, 100};
  return sizes;
}

const std::vector<int>& CppExtendedStressSizes() {
  static const std::vector<int> sizes = {200, 500};
  return sizes;
}

const std::vector<int>& StandardBlockSizes() {
  static const std::vector<int> block_sizes = {1, 2, 3, 5, 8, 16, 21, 32};
  return block_sizes;
}

const std::vector<int>& CppExtendedBlockSizes() {
  static const std::vector<int> block_sizes = {1, 8, 16, 32, 64, 128};
  return block_sizes;
}

const std::vector<double>& WideValueScales() {
  static const std::vector<double> value_scales = {
      1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0,
      1.0e3,  1.0e6,  1.0e9,  1.0e12};
  return value_scales;
}

std::vector<StressCase> BuildStressCases(const std::vector<int>& sizes,
                                         const std::vector<int>& block_sizes,
                                         int base_seed) {
  std::vector<StressCase> cases;
  const std::vector<double>& value_scales = WideValueScales();
  for (int size : sizes) {
    for (int block_size : block_sizes) {
      if (block_size > size) {
        continue;
      }
      for (int scale_index = 0;
           scale_index < static_cast<int>(value_scales.size());
           ++scale_index) {
        cases.push_back({
            size,
            block_size,
            value_scales[scale_index],
            base_seed + 10000 * size + 100 * block_size + 10 * scale_index,
        });
      }
    }
  }
  return cases;
}

std::vector<StressCase> DefaultStressCases() {
  std::vector<StressCase> cases = BuildStressCases(
      StandardStressSizes(), StandardBlockSizes(), 910000);
  std::vector<StressCase> extended_cases = BuildStressCases(
      CppExtendedStressSizes(), CppExtendedBlockSizes(), 920000);
  cases.insert(cases.end(), extended_cases.begin(), extended_cases.end());
  return cases;
}

std::string ScaleToString(double value) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(0) << value;
  return stream.str();
}

std::string ResidualToString(double value) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(3) << value;
  return stream.str();
}

std::string CaseName(const StressCase& test_case) {
  return "size=" + std::to_string(test_case.size) +
         " block_size=" + std::to_string(test_case.block_size) +
         " value_scale=" + ScaleToString(test_case.value_scale) +
         " seed=" + std::to_string(test_case.seed);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

RegularMatrix MakeMatrix(int rows, int cols,
                         const std::vector<double>& values) {
  return RegularMatrix(rows, cols, values);
}

double RelativeError(const RegularMatrix& actual,
                     const RegularMatrix& expected) {
  const double denominator = std::max(FrobeniusNorm(expected), 1.0);
  return FrobeniusNorm(Subtract(actual, expected)) / denominator;
}

std::string MatrixToString(const RegularMatrix& matrix) {
  std::ostringstream stream;
  for (int i = 0; i < matrix.Rows(); ++i) {
    for (int j = 0; j < matrix.Cols(); ++j) {
      stream << matrix(i, j) << " ";
    }
    stream << "\n";
  }
  return stream.str();
}

RegularMatrix RandomSymmetricMatrix(int size, int seed,
                                    double value_scale = 1.0) {
  std::mt19937 generator(seed);
  std::normal_distribution<double> distribution(0.0, 1.0);
  RegularMatrix matrix(size, size);
  for (int i = 0; i < size; ++i) {
    for (int j = i; j < size; ++j) {
      const double value = value_scale * distribution(generator);
      matrix(i, j) = value;
      matrix(j, i) = value;
    }
  }
  return matrix;
}

RegularMatrix MakeTriangularInput(const RegularMatrix& matrix,
                                  MatrixTriangle triangle) {
  RegularMatrix triangular(matrix.Rows(), matrix.Cols());
  for (int row = 0; row < matrix.Rows(); ++row) {
    for (int col = 0; col < matrix.Cols(); ++col) {
      const bool in_selected_triangle =
          row == col ||
          (triangle == MatrixTriangle::kUpper ? row < col : row > col);
      triangular(row, col) =
          in_selected_triangle ? matrix(row, col) : 12345.0;
    }
  }
  return triangular;
}

#if defined(NAIVE_BLOCK_LDLT_USE_MKL)
std::vector<double> ToRowMajor(const RegularMatrix& matrix) {
  std::vector<double> values(
      static_cast<std::size_t>(matrix.Rows()) * matrix.Cols());
  for (int row = 0; row < matrix.Rows(); ++row) {
    for (int col = 0; col < matrix.Cols(); ++col) {
      values[static_cast<std::size_t>(row) * matrix.Cols() + col] =
          matrix(row, col);
    }
  }
  return values;
}

struct ReconstructionResult {
  RegularMatrix reconstructed;
  double relative_residual = 0.0;
};

std::vector<std::pair<int, int>> LapackLowerPivotSwaps(
    const std::vector<lapack_int>& ipiv) {
  std::vector<std::pair<int, int>> swaps;
  for (int pivot = 0; pivot < static_cast<int>(ipiv.size());) {
    if (ipiv[pivot] > 0) {
      swaps.push_back({pivot, static_cast<int>(ipiv[pivot]) - 1});
      ++pivot;
    } else {
      swaps.push_back({pivot + 1, static_cast<int>(-ipiv[pivot]) - 1});
      pivot += 2;
    }
  }
  return swaps;
}

std::vector<int> LapackPermutation(const std::vector<lapack_int>& ipiv,
                                   bool reverse_order) {
  std::vector<int> permutation(ipiv.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    permutation[i] = i;
  }

  const std::vector<std::pair<int, int>> swaps = LapackLowerPivotSwaps(ipiv);
  if (reverse_order) {
    for (auto swap_it = swaps.rbegin(); swap_it != swaps.rend(); ++swap_it) {
      std::swap(permutation[swap_it->first], permutation[swap_it->second]);
    }
  } else {
    for (const auto& [left, right] : swaps) {
      std::swap(permutation[left], permutation[right]);
    }
  }
  return permutation;
}

RegularMatrix ApplyPermutationByFactorPosition(const RegularMatrix& matrix,
                                               const std::vector<int>& perm) {
  RegularMatrix permuted(matrix.Rows(), matrix.Cols());
  for (int row = 0; row < matrix.Rows(); ++row) {
    for (int col = 0; col < matrix.Cols(); ++col) {
      permuted(perm[row], perm[col]) = matrix(row, col);
    }
  }
  return permuted;
}

RegularMatrix ApplyPermutationByOriginalPosition(const RegularMatrix& matrix,
                                                 const std::vector<int>& perm) {
  RegularMatrix permuted(matrix.Rows(), matrix.Cols());
  for (int row = 0; row < matrix.Rows(); ++row) {
    for (int col = 0; col < matrix.Cols(); ++col) {
      permuted(row, col) = matrix(perm[row], perm[col]);
    }
  }
  return permuted;
}

void KeepBestReconstruction(ReconstructionResult* best,
                            const RegularMatrix& candidate,
                            const RegularMatrix& expected) {
  const double residual = RelativeError(candidate, expected);
  if (residual < best->relative_residual) {
    best->reconstructed = candidate;
    best->relative_residual = residual;
  }
}

ReconstructionResult MklLdltReconstruction(const RegularMatrix& matrix) {
  const int size = matrix.Rows();
  std::vector<double> factors = ToRowMajor(matrix);
  std::vector<lapack_int> ipiv(size);
  const lapack_int factor_info =
      LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', size, factors.data(), size,
                     ipiv.data());
  if (factor_info < 0) {
    throw std::runtime_error("LAPACKE_dsytrf failed with info=" +
                             std::to_string(factor_info));
  }

  std::vector<double> off_diagonal(size, 0.0);
  const lapack_int convert_info =
      LAPACKE_dsyconv(LAPACK_ROW_MAJOR, 'L', 'C', size, factors.data(), size,
                      ipiv.data(), off_diagonal.data());
  if (convert_info != 0) {
    throw std::runtime_error("LAPACKE_dsyconv failed with info=" +
                             std::to_string(convert_info));
  }

  RegularMatrix lower(size, size);
  RegularMatrix diagonal(size, size);
  for (int row = 0; row < size; ++row) {
    lower(row, row) = 1.0;
    diagonal(row, row) = factors[static_cast<std::size_t>(row) * size + row];
    for (int col = 0; col < row; ++col) {
      lower(row, col) = factors[static_cast<std::size_t>(row) * size + col];
    }
  }
  for (int pivot = 0; pivot + 1 < size; ++pivot) {
    if (off_diagonal[pivot] != 0.0) {
      diagonal(pivot, pivot + 1) = off_diagonal[pivot];
      diagonal(pivot + 1, pivot) = off_diagonal[pivot];
    }
  }

  const RegularMatrix unpermuted = naive_block_ldlt::Multiply(
      naive_block_ldlt::Multiply(lower, diagonal),
      naive_block_ldlt::Transpose(lower));
  ReconstructionResult best = {unpermuted, RelativeError(unpermuted, matrix)};

  for (bool reverse_order : {false, true}) {
    const std::vector<int> permutation =
        LapackPermutation(ipiv, reverse_order);
    KeepBestReconstruction(
        &best, ApplyPermutationByFactorPosition(unpermuted, permutation),
        matrix);
    KeepBestReconstruction(
        &best, ApplyPermutationByOriginalPosition(unpermuted, permutation),
        matrix);
  }
  return best;
}

void CheckMklComparison(const RegularMatrix& matrix,
                        const RegularMatrix& reconstructed,
                        double naive_residual,
                        const std::string& case_name) {
  const ReconstructionResult mkl = MklLdltReconstruction(matrix);
  Expect(mkl.relative_residual < 1.0e-8,
         "MKL LDLT reconstruction residual exceeds tolerance");
  const double naive_vs_mkl_residual =
      RelativeError(reconstructed, mkl.reconstructed);
  Expect(naive_vs_mkl_residual < 1.0e-8,
         "naive and MKL LDLT reconstructions differ");
  std::cout << "mkl comparison " << case_name
            << " naive_relative_residual="
            << ResidualToString(naive_residual)
            << " mkl_ldlt_relative_residual="
            << ResidualToString(mkl.relative_residual)
            << " naive_vs_mkl_relative_residual="
            << ResidualToString(naive_vs_mkl_residual)
            << " residual_difference="
            << ResidualToString(
                   std::abs(naive_residual - mkl.relative_residual))
            << "\n";
}
#else
void CheckMklComparison(const RegularMatrix&, const RegularMatrix&, double,
                        const std::string&) {}
#endif

void CheckAlignedPointer(const double* pointer, const std::string& name) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  Expect(address % naive_block_ldlt::kDefaultAlignment == 0,
         name + " is not 64-byte aligned");
}

void TestAlignedStorageAndIndexing() {
  RegularMatrix matrix(4, 5);
  CheckAlignedPointer(matrix.Data(), "RegularMatrix data");
  matrix(3, 4) = 42.0;
  Expect(matrix.Data()[3 * matrix.LeadingDimension() + 4] == 42.0,
         "RegularMatrix operator() must use row-major indexing");

  BlockMatrix block_matrix(7, 3);
  block_matrix(5, 2) = -3.0;
  Expect(block_matrix(2, 5) == -3.0,
         "BlockMatrix operator() must map symmetric lower access");
  CheckAlignedPointer(block_matrix.GetBlock(0, 0).Data(),
                      "BlockMatrix first block");
}

void TestKernelShapes() {
  double source0[] = {2.0, 4.0, 6.0};
  double destination0[] = {0.0, 0.0, 0.0};
  naive_block_ldlt::kernels::ScaleOneByOnePivot(3, 2.0, source0, 1,
                                                destination0, 1);
  Expect(destination0[0] == 1.0 && destination0[2] == 3.0,
         "ScaleOneByOnePivot failed");

  double source1[] = {1.0, 2.0};
  double source2[] = {3.0, 4.0};
  double destination1[] = {0.0, 0.0};
  double destination2[] = {0.0, 0.0};
  naive_block_ldlt::kernels::ScaleTwoByTwoPivot(
      2, 2.0, 1.0, 3.0, source1, 1, source2, 1, destination1, 1,
      destination2, 1);
  Expect(std::abs(destination1[0]) < 1.0e-14 &&
             std::abs(destination2[0] - 1.0) < 1.0e-14,
         "ScaleTwoByTwoPivot failed");
}

void TestTriangularMatrixInput() {
  const RegularMatrix matrix =
      MakeMatrix(4, 4,
                 {
                     6.0, 12.0, 3.0, -6.0,
                     12.0, -8.0, -13.0, 4.0,
                     3.0, -13.0, -1.0, 1.0,
                     -6.0, 4.0, 1.0, 6.0,
                 });
  for (int block_size = 1; block_size <= matrix.Rows(); ++block_size) {
    const BlockMatrix upper_input(
        MakeTriangularInput(matrix, MatrixTriangle::kUpper), block_size,
        MatrixTriangle::kUpper);
    Expect(RelativeError(upper_input.ToRegular(), matrix) < kTolerance,
           "upper triangular input must ignore lower entries");
    BlockLdltDecomposition upper_decomposition(upper_input);
    upper_decomposition.Factorize();
    Expect(RelativeError(upper_decomposition.ReconstructOriginal(), matrix) <
               kTolerance,
           "upper triangular input factorization failed");

    const BlockMatrix lower_input(
        MakeTriangularInput(matrix, MatrixTriangle::kLower), block_size,
        MatrixTriangle::kLower);
    Expect(RelativeError(lower_input.ToRegular(), matrix) < kTolerance,
           "lower triangular input must ignore upper entries");
    BlockLdltDecomposition lower_decomposition(lower_input);
    lower_decomposition.Factorize();
    Expect(RelativeError(lower_decomposition.ReconstructOriginal(), matrix) <
               kTolerance,
           "lower triangular input factorization failed");
  }
}

void TestReferenceCompactStorage() {
  const RegularMatrix matrix =
      MakeMatrix(4, 4,
                 {
                     6.0, 12.0, 3.0, -6.0,
                     12.0, -8.0, -13.0, 4.0,
                     3.0, -13.0, -1.0, 1.0,
                     -6.0, 4.0, 1.0, 6.0,
                 });
  BlockLdltDecomposition decomposition(BlockMatrix(matrix, 2));
  decomposition.Factorize();

  const RegularMatrix expected_compact =
      MakeMatrix(4, 4,
                 {
                     6.0, 12.0, -0.6875, 0.0,
                     0.0, -8.0, 0.59375, -0.5,
                     0.0, 0.0, 8.78125, -0.6263345195729537,
                     0.0, 0.0, 0.0, 4.555160142348754,
                 });
  const double compact_error =
      RelativeError(decomposition.CompactFactors().ToUpperRegular(),
                    expected_compact);
  Expect(compact_error < kTolerance,
         "compact storage must match the reference compact factors");

  const std::vector<int> expected_pivots = {2, -2, 1, 1};
  Expect(decomposition.PivotTypes() == expected_pivots,
         "pivot types must match the reference pivots");
}

double AssertValidFactorization(const RegularMatrix& matrix, int block_size,
                                const std::string& case_name) {
  const std::string full_case_name =
      case_name + " block_size=" + std::to_string(block_size);
  BlockMatrix block_matrix(matrix, block_size);
  BlockLdltDecomposition decomposition(block_matrix);
  decomposition.Factorize();
  const RegularMatrix reconstructed = decomposition.ReconstructOriginal();
  const double error = RelativeError(reconstructed, matrix);
  if (error >= kTolerance) {
    throw std::runtime_error(full_case_name +
                             " factorization relative error " +
                             std::to_string(error) + " exceeds tolerance\n" +
                             "compact:\n" +
                             MatrixToString(
                                 decomposition.CompactFactors()
                                     .ToUpperRegular()) +
                              "reconstructed:\n" +
                              MatrixToString(reconstructed));
  }
  CheckMklComparison(matrix, reconstructed, error, full_case_name);
  return error;
}

void TestFixedFactorizations() {
  const std::vector<RegularMatrix> matrices = {
      MakeMatrix(4, 4,
                 {
                     6.0, 12.0, 3.0, -6.0,
                     12.0, -8.0, -13.0, 4.0,
                     3.0, -13.0, -1.0, 1.0,
                     -6.0, 4.0, 1.0, 6.0,
                 }),
      MakeMatrix(4, 4,
                 {
                     0.0, 1.0, 0.0, 0.0,
                     1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0,
                     0.0, 0.0, 1.0, 0.0,
                 }),
      MakeMatrix(4, 4,
                 {
                     2.0, 0.0, 0.0, 0.0,
                     0.0, -3.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 5.0,
                 }),
  };

  for (const RegularMatrix& matrix : matrices) {
    const int case_index = static_cast<int>(&matrix - matrices.data());
    for (int block_size = 1; block_size <= matrix.Rows(); ++block_size) {
      AssertValidFactorization(matrix, block_size,
                               "fixed case " + std::to_string(case_index));
    }
  }
}

void TestRandomFactorizations() {
  for (int size = 3; size <= 16; ++size) {
    for (int block_size = 1; block_size <= size; ++block_size) {
      for (int seed = 0; seed < 4; ++seed) {
        AssertValidFactorization(
            RandomSymmetricMatrix(size, 10000 * size + 100 * block_size + seed),
            block_size,
            "random factorization size=" + std::to_string(size) +
                " seed=" + std::to_string(seed));
      }
    }
  }
}

void TestLargeFactorizations() {
  for (const StressCase& test_case : DefaultStressCases()) {
    const double relative_residual = AssertValidFactorization(
        RandomSymmetricMatrix(test_case.size, test_case.seed,
                              test_case.value_scale),
        test_case.block_size, "default stress " + CaseName(test_case));
    std::cout << "default stress " << CaseName(test_case)
              << " relative_residual="
              << ResidualToString(relative_residual) << "\n";
  }
}

}  // namespace

int main() {
  try {
    TestAlignedStorageAndIndexing();
    TestKernelShapes();
    TestTriangularMatrixInput();
    TestReferenceCompactStorage();
    TestFixedFactorizations();
    TestRandomFactorizations();
    TestLargeFactorizations();
  } catch (const std::exception& error) {
    std::cerr << "C++ block LDLT tests failed: " << error.what() << "\n";
    return 1;
  }

  std::cout << "C++ block LDLT tests passed\n";
  return 0;
}
