# BlockedBunchKaufmanLDLT

This repository contains a C++20 implementation of a blocked Bunch-Kaufman
`LDL^T` factorization for dense symmetric indefinite matrices.

The implementation is not a pure left-looking factorization. The outer blocked
algorithm factors panels from left to right and applies each completed panel to
the trailing block matrix. Within a panel, it uses workspace rows updated from
previous pivots in the panel before choosing Bunch-Kaufman `1 x 1` or `2 x 2`
pivots.

The code emphasizes:

- aligned row-major matrix storage
- an abstract `Matrix` base class
- `RegularMatrix` and `BlockMatrix` implementations
- upper-triangular block storage
- compact in-place LDLT factor storage
- Bunch-Kaufman threshold pivoting with symmetric row/column swaps
- pointer/leading-dimension kernel boundaries
- optional OpenMP and Intel MKL-backed BLAS/LAPACK kernels and comparisons

## CMake

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --verbose --output-on-failure -C Release
```

The CTest output includes detailed case logs. Reconstruction stress cases print
`value_scale` and `relative_residual`. The large-stress executable runs its
`--default-large` suite under CTest, which includes selected `5000 x 5000`
compact-only cases. Full `5000 x 5000` and `10000 x 10000` compact-only sweeps
are available explicitly.
When MKL is available, the test script enables it automatically. In MKL
builds, every reconstructing factorization test case is compared with MKL's
LAPACKE `dsytrf` LDLT path and logs `naive_relative_residual`,
`mkl_ldlt_relative_residual`, `naive_vs_mkl_relative_residual`, and
`residual_difference`. Full compact-only large-stress cases compute a sampled
residual from the compact LDLT factors and fail if it exceeds tolerance.

## Matrix Input

`BlockMatrix` represents symmetric matrices in upper-triangular block storage.
When constructing from a `RegularMatrix`, only one triangle is read. The upper
triangle is the default; pass `MatrixTriangle::kLower` for lower-triangle input:

```cpp
naive_block_ldlt::BlockMatrix from_upper(
    matrix, block_size, naive_block_ldlt::MatrixTriangle::kUpper);
naive_block_ldlt::BlockMatrix from_lower(
    matrix, block_size, naive_block_ldlt::MatrixTriangle::kLower);
```

## Test Script

The repository keeps one Bash test helper for Git Bash on Windows, macOS, and
Linux:

```bash
bash scripts/run_cpp_tests.sh
```

The script uses CMake, auto-selects Ninja or MinGW Makefiles when useful, and
writes the live output to `logs/bash_tests.log`. It auto-enables MKL comparisons
when an MKL root is provided through `MKL_ROOT`, `MKLROOT`, or
`NAIVE_BLOCK_LDLT_MKL_ROOT` and the detected toolchain is compatible. On Windows
with MinGW, MKL stays off by default because Intel MKL's Windows import
libraries are MSVC/Intel-toolchain oriented.

Optional Bash settings:

```bash
bash scripts/run_cpp_tests.sh --config Debug
bash scripts/run_cpp_tests.sh --generator Ninja
bash scripts/run_cpp_tests.sh --openmp
bash scripts/run_cpp_tests.sh --mkl-root /path/to/mkl
```

On Windows/Git Bash, MKL auto-detection or `--mkl` also prepends the MKL,
Intel compiler, and TBB runtime DLL folders to `PATH` for the test run.

## Options

OpenMP:

```bash
cmake -S . -B build-openmp -DNAIVE_BLOCK_LDLT_USE_OPENMP=ON
```

Intel MKL kernels and factorization comparisons:

```bash
cmake -S . -B build-mkl -DNAIVE_BLOCK_LDLT_USE_MKL=ON
cmake --build build-mkl --parallel
ctest --test-dir build-mkl --verbose --output-on-failure
```

CMake first tries Intel's MKL package config through normal CMake package
discovery. It then falls back to finding `mkl_lapacke.h` and `mkl_rt` from
`MKLROOT`, `NAIVE_BLOCK_LDLT_MKL_ROOT`, or standard system search paths. If MKL
is not found automatically, set `MKLROOT`, `MKL_DIR`, `CMAKE_PREFIX_PATH`, or
pass `-DNAIVE_BLOCK_LDLT_MKL_ROOT=/path/to/mkl`.

## Large Stress Tests

The large-stress executable can run selected `5000 x 5000` and `10000 x 10000`
compact-only stress cases. These cases print a computed sampled residual, not a
placeholder:
`relative_residual`, `residual_sample_count`, and `residual_mode=sampled`.

After building with CMake:

```bash
build/block_ldlt_large_stress_test --list
build/block_ldlt_large_stress_test --default-large
```

For multi-config generators:

```bash
build/Release/block_ldlt_large_stress_test.exe --list
build/Release/block_ldlt_large_stress_test.exe --default-large
```

Run full `5000 x 5000` and selected `10000 x 10000` cases explicitly when
needed:

```bash
build/block_ldlt_large_stress_test
build/block_ldlt_large_stress_test --max-size 10000
build/block_ldlt_large_stress_test --max-size 10000 --case-limit 1
```

## Notes

The implementation is a correctness-oriented research prototype. It is useful
for validating block metadata, pivot behavior, compact storage, and future
kernel boundaries, but it is not a production sparse or out-of-core solver.
