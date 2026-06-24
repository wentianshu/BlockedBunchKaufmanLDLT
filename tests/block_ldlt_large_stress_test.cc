#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "block.h"
#include "block_ldlt_decomposition.h"
#include "block_matrix.h"

namespace {

using naive_block_ldlt::Block;
using naive_block_ldlt::BlockLdltDecomposition;
using naive_block_ldlt::BlockMatrix;

constexpr int kResidualSampleCount = 4096;
constexpr double kResidualTolerance = 1.0e-8;

struct StressCase {
  int size = 0;
  int block_size = 0;
  double value_scale = 1.0;
  int seed = 0;
};

struct Options {
  bool list = false;
  bool self_test = false;
  bool default_large = false;
  int max_size = 5000;
  int case_limit = std::numeric_limits<int>::max();
};

const std::vector<double>& WideValueScales() {
  static const std::vector<double> value_scales = {
      1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0,
      1.0e3,  1.0e6,  1.0e9,  1.0e12};
  return value_scales;
}

std::vector<StressCase> BuildHugeStressCases() {
  std::vector<StressCase> cases;
  const std::vector<double>& value_scales = WideValueScales();
  const std::vector<std::pair<int, std::vector<int>>> size_blocks = {
      {5000, {256, 512, 1024}},
      {10000, {512, 1024, 2048}},
  };
  for (const auto& [size, block_sizes] : size_blocks) {
    for (int block_size : block_sizes) {
      for (int scale_index = 0;
           scale_index < static_cast<int>(value_scales.size());
           ++scale_index) {
        cases.push_back({
            size,
            block_size,
            value_scales[scale_index],
            930000 + 10000 * size + 100 * block_size + 10 * scale_index,
        });
      }
    }
  }
  return cases;
}

std::vector<StressCase> SelfTestCases() {
  return {
      {20, 5, 1.0e-3, 931000},
      {20, 8, 1.0, 931100},
      {20, 16, 1.0e3, 931200},
  };
}

std::vector<StressCase> DefaultLargeStressCases() {
  return {
      {5000, 256, 1.0e-3, 931300},
      {5000, 512, 1.0, 931400},
      {5000, 1024, 1.0e3, 931500},
  };
}

std::string ScaleToString(double value) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(0) << value;
  return stream.str();
}

std::string CaseName(const StressCase& test_case) {
  return "size=" + std::to_string(test_case.size) +
         " block_size=" + std::to_string(test_case.block_size) +
         " value_scale=" + ScaleToString(test_case.value_scale) +
         " seed=" + std::to_string(test_case.seed);
}

std::string ResidualToString(double value) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(3) << value;
  return stream.str();
}

std::size_t EstimatedWorkingSetBytes(const StressCase& test_case) {
  const auto size = static_cast<std::size_t>(test_case.size);
  const std::size_t full_matrix_bytes = size * size * sizeof(double);
  const std::size_t upper_matrix_bytes = full_matrix_bytes / 2;
  return 3 * upper_matrix_bytes;
}

BlockMatrix RandomSymmetricBlockMatrix(const StressCase& test_case) {
  BlockMatrix matrix(test_case.size, test_case.block_size);
  std::mt19937 generator(test_case.seed);
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (int block_col = 0; block_col < matrix.num_blocks(); ++block_col) {
    for (int block_row = 0; block_row <= block_col; ++block_row) {
      Block& block = matrix.GetBlock(block_row, block_col);
      for (int row = 0; row < block.rows(); ++row) {
        const int global_row = block.row0() + row;
        for (int col = 0; col < block.cols(); ++col) {
          const int global_col = block.col0() + col;
          if (global_row > global_col) {
            continue;
          }
          block(row, col) = test_case.value_scale * distribution(generator);
        }
      }
    }
  }
  return matrix;
}

void ExpectFiniteCompactFactors(const BlockMatrix& matrix,
                                const std::string& case_name) {
  for (int block_col = 0; block_col < matrix.num_blocks(); ++block_col) {
    for (int block_row = 0; block_row <= block_col; ++block_row) {
      const Block& block = matrix.GetBlock(block_row, block_col);
      for (int row = 0; row < block.rows(); ++row) {
        const int global_row = block.row0() + row;
        for (int col = 0; col < block.cols(); ++col) {
          const int global_col = block.col0() + col;
          if (global_row > global_col) {
            continue;
          }
          if (!std::isfinite(block(row, col))) {
            throw std::runtime_error(case_name +
                                     " produced a non-finite compact factor");
          }
        }
      }
    }
  }
}

double CompactLValue(const BlockMatrix& compact_factors,
                     const std::vector<int>& pivot_types, int row, int col) {
  if (row == col) {
    return 1.0;
  }
  if (row < col) {
    return 0.0;
  }
  if (pivot_types[col] == 2 && row == col + 1) {
    return 0.0;
  }
  return compact_factors(col, row);
}

double ReconstructedPermutedEntry(const BlockMatrix& compact_factors,
                                  const std::vector<int>& pivot_types, int row,
                                  int col) {
  double value = 0.0;
  for (int pivot = 0; pivot < compact_factors.Rows(); ++pivot) {
    const double row_left = CompactLValue(compact_factors, pivot_types, row,
                                          pivot);
    const double col_left = CompactLValue(compact_factors, pivot_types, col,
                                          pivot);
    value += row_left * compact_factors(pivot, pivot) * col_left;
    if (pivot_types[pivot] == 2) {
      const double off_diagonal = compact_factors(pivot, pivot + 1);
      const double row_next = CompactLValue(compact_factors, pivot_types, row,
                                            pivot + 1);
      const double col_next = CompactLValue(compact_factors, pivot_types, col,
                                            pivot + 1);
      value += off_diagonal * (row_left * col_next + row_next * col_left);
    }
  }
  return value;
}

std::vector<int> InversePermutation(const std::vector<int>& permutation) {
  std::vector<int> inverse(permutation.size());
  for (int index = 0; index < static_cast<int>(permutation.size()); ++index) {
    inverse[permutation[index]] = index;
  }
  return inverse;
}

std::vector<std::pair<int, int>> ResidualSamples(int size, int seed) {
  std::vector<std::pair<int, int>> samples;
  samples.reserve(kResidualSampleCount + 4);
  samples.push_back({0, 0});
  samples.push_back({0, size - 1});
  samples.push_back({size / 2, size / 2});
  samples.push_back({size - 1, size - 1});

  std::mt19937 generator(seed + 170003);
  std::uniform_int_distribution<int> distribution(0, size - 1);
  while (static_cast<int>(samples.size()) < kResidualSampleCount) {
    samples.push_back({distribution(generator), distribution(generator)});
  }
  return samples;
}

double SampledRelativeResidual(const BlockMatrix& original,
                               const BlockLdltDecomposition& decomposition,
                               int seed) {
  const BlockMatrix& compact_factors = decomposition.CompactFactors();
  const std::vector<int>& pivot_types = decomposition.PivotTypes();
  const std::vector<int> inverse_permutation =
      InversePermutation(decomposition.Permutation());
  const std::vector<std::pair<int, int>> samples =
      ResidualSamples(original.Rows(), seed);

  double residual_squared = 0.0;
  double original_squared = 0.0;
  for (const auto& [original_row, original_col] : samples) {
    const int factor_row = inverse_permutation[original_row];
    const int factor_col = inverse_permutation[original_col];
    const double reconstructed = ReconstructedPermutedEntry(
        compact_factors, pivot_types, factor_row, factor_col);
    const double expected = original(original_row, original_col);
    const double difference = reconstructed - expected;
    residual_squared += difference * difference;
    original_squared += expected * expected;
  }
  return std::sqrt(residual_squared) / std::max(std::sqrt(original_squared),
                                                1.0);
}

void RunCase(const StressCase& test_case) {
  const std::string case_name = CaseName(test_case);
  std::cout << "running " << case_name << " estimated_working_set_gib="
            << std::fixed << std::setprecision(2)
            << static_cast<double>(EstimatedWorkingSetBytes(test_case)) /
                   (1024.0 * 1024.0 * 1024.0)
            << "\n";

  BlockMatrix matrix = RandomSymmetricBlockMatrix(test_case);
  BlockLdltDecomposition decomposition(
      matrix, BlockLdltDecomposition::FactorizationMode::kCompactOnly);
  decomposition.Factorize();
  ExpectFiniteCompactFactors(decomposition.CompactFactors(), case_name);
  const double relative_residual =
      SampledRelativeResidual(matrix, decomposition, test_case.seed);
  if (relative_residual >= kResidualTolerance) {
    throw std::runtime_error(case_name + " sampled relative residual " +
                             ResidualToString(relative_residual) +
                             " exceeds tolerance");
  }
  matrix = BlockMatrix();
  std::cout << "completed " << case_name
            << " relative_residual=" << ResidualToString(relative_residual)
            << " residual_sample_count=" << kResidualSampleCount
            << " residual_mode=sampled compact_only=true\n";
}

Options ParseArgs(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--list") {
      options.list = true;
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--default-large") {
      options.default_large = true;
    } else if (argument == "--max-size" && index + 1 < argc) {
      options.max_size = std::stoi(argv[++index]);
    } else if (argument == "--case-limit" && index + 1 < argc) {
      options.case_limit = std::stoi(argv[++index]);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
    }
  }
  return options;
}

std::vector<StressCase> SelectCases(const Options& options) {
  if (options.self_test && options.default_large) {
    throw std::invalid_argument(
        "--self-test and --default-large cannot be used together");
  }

  std::vector<StressCase> cases;
  if (options.self_test) {
    cases = SelfTestCases();
  } else if (options.default_large) {
    cases = DefaultLargeStressCases();
  } else {
    cases = BuildHugeStressCases();
  }
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [&](const StressCase& test_case) {
                               return test_case.size > options.max_size;
                             }),
              cases.end());
  if (static_cast<int>(cases.size()) > options.case_limit) {
    cases.resize(options.case_limit);
  }
  return cases;
}

int Run(int argc, char* argv[]) {
  const Options options = ParseArgs(argc, argv);
  const std::vector<StressCase> cases = SelectCases(options);

  if (options.list) {
    std::cout << "large C++ stress cases:\n";
    for (const StressCase& test_case : cases) {
      std::cout << "  " << CaseName(test_case) << " estimated_working_set_gib="
                << std::fixed << std::setprecision(2)
                << static_cast<double>(EstimatedWorkingSetBytes(test_case)) /
                       (1024.0 * 1024.0 * 1024.0)
                << "\n";
    }
    return 0;
  }

  for (const StressCase& test_case : cases) {
    RunCase(test_case);
  }
  std::cout << "C++ large stress tests passed\n";
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "C++ large stress tests failed: " << error.what() << "\n";
    return 1;
  }
}
