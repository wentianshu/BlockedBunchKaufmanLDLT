cmake_minimum_required(VERSION 3.20)

project(BlockedBunchKaufmanLDLT VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

option(NAIVE_BLOCK_LDLT_USE_OPENMP "Enable OpenMP block-level kernels" OFF)
option(NAIVE_BLOCK_LDLT_USE_MKL "Enable Intel MKL-backed BLAS/LAPACK kernels" OFF)
set(
  NAIVE_BLOCK_LDLT_MKL_ROOT
  ""
  CACHE PATH
  "Intel MKL installation root"
)

if(NAIVE_BLOCK_LDLT_USE_MKL AND WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(
    FATAL_ERROR
    "NAIVE_BLOCK_LDLT_USE_MKL=ON is not supported with MinGW/GNU on Windows. "
    "Intel MKL's Windows import libraries are MSVC/Intel-toolchain oriented; "
    "use USE_MKL=OFF, omit --mkl, or build with an MSVC/Intel-compatible "
    "toolchain."
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
  blocked_bunch_kaufman_ldlt
  src/block_ldlt_decomposition.cc
  src/block_matrix.cc
  src/kernels.cc
  src/regular_matrix.cc
)
add_library(naive_block_ldlt ALIAS blocked_bunch_kaufman_ldlt)

target_include_directories(
  blocked_bunch_kaufman_ldlt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(NAIVE_BLOCK_LDLT_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(blocked_bunch_kaufman_ldlt PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(
    blocked_bunch_kaufman_ldlt
    PUBLIC
      NAIVE_BLOCK_LDLT_USE_OPENMP
  )
endif()

if(NAIVE_BLOCK_LDLT_USE_MKL)
  set(_naive_block_ldlt_mkl_hints)
  if(NAIVE_BLOCK_LDLT_MKL_ROOT)
    list(APPEND _naive_block_ldlt_mkl_hints "${NAIVE_BLOCK_LDLT_MKL_ROOT}")
  endif()
  if(DEFINED ENV{MKLROOT})
    list(APPEND _naive_block_ldlt_mkl_hints "$ENV{MKLROOT}")
  endif()
  set(_naive_block_ldlt_mkl_include_dirs)
  set(_naive_block_ldlt_mkl_libraries)

  if(_naive_block_ldlt_mkl_hints)
    find_package(MKL QUIET CONFIG HINTS ${_naive_block_ldlt_mkl_hints})
  else()
    find_package(MKL QUIET CONFIG)
  endif()
  if(MKL_FOUND)
    list(APPEND _naive_block_ldlt_mkl_libraries MKL::MKL)
  else()
    find_path(
      NAIVE_BLOCK_LDLT_MKL_INCLUDE_DIR
      NAMES mkl_lapacke.h
      HINTS ${_naive_block_ldlt_mkl_hints}
      PATH_SUFFIXES include
    )
    find_library(
      NAIVE_BLOCK_LDLT_MKL_RT_LIBRARY
      NAMES mkl_rt
      HINTS ${_naive_block_ldlt_mkl_hints}
      PATH_SUFFIXES lib lib/intel64 lib/intel64_win
    )
    if(
      NOT NAIVE_BLOCK_LDLT_MKL_INCLUDE_DIR
      OR NOT NAIVE_BLOCK_LDLT_MKL_RT_LIBRARY
    )
      message(
        FATAL_ERROR
        "Intel MKL was requested but CMake could not find MKLConfig.cmake, "
        "mkl_lapacke.h, or mkl_rt. Set NAIVE_BLOCK_LDLT_MKL_ROOT, MKLROOT, "
        "MKL_DIR, or CMAKE_PREFIX_PATH to your Intel oneAPI MKL install."
      )
    endif()

    list(APPEND _naive_block_ldlt_mkl_include_dirs
         "${NAIVE_BLOCK_LDLT_MKL_INCLUDE_DIR}")
    list(APPEND _naive_block_ldlt_mkl_libraries
         "${NAIVE_BLOCK_LDLT_MKL_RT_LIBRARY}")
    mark_as_advanced(
      NAIVE_BLOCK_LDLT_MKL_INCLUDE_DIR
      NAIVE_BLOCK_LDLT_MKL_RT_LIBRARY
    )
  endif()

  target_include_directories(
    blocked_bunch_kaufman_ldlt
    PRIVATE
      ${_naive_block_ldlt_mkl_include_dirs}
  )
  target_link_libraries(
    blocked_bunch_kaufman_ldlt
    PRIVATE
      ${_naive_block_ldlt_mkl_libraries}
  )
  target_compile_definitions(
    blocked_bunch_kaufman_ldlt
    PRIVATE
      NAIVE_BLOCK_LDLT_USE_MKL
  )
endif()

install(
  TARGETS blocked_bunch_kaufman_ldlt
  EXPORT BlockedBunchKaufmanLDLTTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT BlockedBunchKaufmanLDLTTargets
  NAMESPACE BlockedBunchKaufmanLDLT::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BlockedBunchKaufmanLDLT
)

if(BUILD_TESTING)
  add_executable(
    block_ldlt_decomposition_test
    tests/block_ldlt_decomposition_test.cc
  )
  target_link_libraries(block_ldlt_decomposition_test PRIVATE blocked_bunch_kaufman_ldlt)
  if(NAIVE_BLOCK_LDLT_USE_MKL)
    target_include_directories(
      block_ldlt_decomposition_test
      PRIVATE
        ${_naive_block_ldlt_mkl_include_dirs}
    )
    target_link_libraries(
      block_ldlt_decomposition_test
      PRIVATE
        ${_naive_block_ldlt_mkl_libraries}
    )
    target_compile_definitions(
      block_ldlt_decomposition_test
      PRIVATE
        NAIVE_BLOCK_LDLT_USE_MKL
    )
  endif()

  add_executable(
    block_ldlt_large_stress_test
    tests/block_ldlt_large_stress_test.cc
  )
  target_link_libraries(block_ldlt_large_stress_test PRIVATE blocked_bunch_kaufman_ldlt)

  add_test(
    NAME block_ldlt_decomposition_test
    COMMAND block_ldlt_decomposition_test
  )
  add_test(
    NAME block_ldlt_large_stress_test
    COMMAND block_ldlt_large_stress_test --default-large
  )
endif()
