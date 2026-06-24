#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG="${CONFIG:-${BUILD_TYPE:-Release}}"
LOG_DIR="${CPP_DIR}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bash_tests.log}"
USE_OPENMP="${USE_OPENMP:-OFF}"
USE_MKL="${USE_MKL:-AUTO}"
MKL_ROOT="${MKL_ROOT:-}"
GENERATOR="${GENERATOR:-}"
BUILD_DIR="${BUILD_DIR:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cpp_tests.sh [options]

Configure, build, and run BlockedBunchKaufmanLDLT C++ tests with CMake.
MKL comparisons are enabled automatically when an MKL root is provided and the
toolchain is compatible.

Options:
  --build-dir PATH       CMake build directory.
  --config NAME          Build configuration, default: Release.
  --generator NAME       CMake generator, e.g. Ninja or MinGW Makefiles.
  --openmp              Configure with NAIVE_BLOCK_LDLT_USE_OPENMP=ON.
  --mkl                 Force NAIVE_BLOCK_LDLT_USE_MKL=ON.
  --mkl-root PATH        Intel MKL root.
  --help                Show this help.

Environment overrides:
  BUILD_DIR, CONFIG, GENERATOR, USE_OPENMP, USE_MKL, MKL_ROOT,
  NAIVE_BLOCK_LDLT_MKL_ROOT, MKLROOT, LOG_FILE
  Set USE_MKL=OFF to force a non-MKL test run.
EOF
}

to_cmake_bool() {
  case "${1}" in
    1|ON|on|On|TRUE|true|True|YES|yes|Yes) echo "ON" ;;
    *) echo "OFF" ;;
  esac
}

platform_name() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    Darwin*) echo "macos" ;;
    Linux*) echo "linux" ;;
    *) echo "unknown" ;;
  esac
}

detect_generator() {
  local platform="$1"
  if [[ -n "${GENERATOR}" ]]; then
    echo "${GENERATOR}"
    return
  fi
  if command -v ninja >/dev/null 2>&1; then
    echo "Ninja"
    return
  fi
  if [[ "${platform}" == "windows" ]]; then
    if command -v mingw32-make >/dev/null 2>&1 && has_cxx_compiler; then
      echo "MinGW Makefiles"
      return
    fi
    echo \
      "Git Bash on Windows needs ninja or mingw32-make plus a C++ compiler on PATH." \
      >&2
    exit 127
  fi
}

has_cxx_compiler() {
  command -v c++ >/dev/null 2>&1 ||
    command -v g++ >/dev/null 2>&1 ||
    command -v clang++ >/dev/null 2>&1
}

check_required_tool() {
  local tool_name="$1"
  if ! command -v "${tool_name}" >/dev/null 2>&1; then
    echo "Required executable was not found on PATH: ${tool_name}" >&2
    exit 127
  fi
}

check_generator_tools() {
  case "${1}" in
    "")
      return
      ;;
    Ninja)
      check_required_tool ninja
      ;;
    "MinGW Makefiles")
      check_required_tool mingw32-make
      if ! has_cxx_compiler; then
        echo "MinGW Makefiles needs c++, g++, or clang++ on PATH." >&2
        exit 127
      fi
      ;;
  esac
}

mkl_supported_by_toolchain() {
  local platform="$1"
  local generator="$2"
  if [[ "${platform}" != "windows" ]]; then
    return 0
  fi
  if [[ "${generator}" == "MinGW Makefiles" ]]; then
    return 1
  fi
  if [[ -n "${CXX:-}" ]]; then
    case "$(basename "${CXX}")" in
      cl|cl.exe|icx|icx.exe|icx-cl|icx-cl.exe) return 0 ;;
      *) return 1 ;;
    esac
  fi
  command -v cl >/dev/null 2>&1 ||
    command -v icx >/dev/null 2>&1 ||
    command -v icx-cl >/dev/null 2>&1
}

generator_suffix() {
  case "${1}" in
    "") echo "" ;;
    "MinGW Makefiles") echo "-mingw" ;;
    *) echo "-$(echo "${1}" | tr '[:upper:] ' '[:lower:]-')" ;;
  esac
}

detect_mkl_root() {
  if [[ -n "${MKL_ROOT}" ]]; then
    echo "${MKL_ROOT}"
    return
  fi
  if [[ -n "${NAIVE_BLOCK_LDLT_MKL_ROOT:-}" ]]; then
    echo "${NAIVE_BLOCK_LDLT_MKL_ROOT}"
    return
  fi
  if [[ -n "${MKLROOT:-}" ]]; then
    echo "${MKLROOT}"
    return
  fi
}

prepend_mkl_runtime_path() {
  local mkl_root="$1"
  if [[ -z "${mkl_root}" ]]; then
    return
  fi

  local oneapi_root=""
  oneapi_root="$(cd "${mkl_root}/../.." >/dev/null 2>&1 && pwd -P || true)"
  local runtime_paths=()
  if [[ -d "${mkl_root}/bin" ]]; then
    runtime_paths+=("${mkl_root}/bin")
  fi
  if [[ -n "${oneapi_root}" ]]; then
    local runtime_path
    for runtime_path in \
      "${oneapi_root}"/compiler/*/bin \
      "${oneapi_root}"/tbb/*/bin \
      "${oneapi_root}"/tbb/*/bin/vc_mt; do
      if [[ -d "${runtime_path}" ]]; then
        runtime_paths+=("${runtime_path}")
      fi
    done
  fi

  if [[ "${#runtime_paths[@]}" -gt 0 ]]; then
    local joined_paths
    joined_paths="$(IFS=:; echo "${runtime_paths[*]}")"
    export PATH="${joined_paths}:${PATH}"
  fi
}

log() {
  echo "$*" | tee -a "${LOG_FILE}"
}

log_command() {
  printf '$ ' | tee -a "${LOG_FILE}"
  printf '%q ' "$@" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"
}

run_logged() {
  log
  log_command "$@"
  set +e
  "$@" 2>&1 | tee -a "${LOG_FILE}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --generator)
      GENERATOR="$2"
      shift 2
      ;;
    --openmp)
      USE_OPENMP="ON"
      shift
      ;;
    --mkl)
      USE_MKL="ON"
      shift
      ;;
    --mkl-root)
      MKL_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

USE_OPENMP="$(to_cmake_bool "${USE_OPENMP}")"
PLATFORM="$(platform_name)"
GENERATOR="$(detect_generator "${PLATFORM}")"
check_required_tool cmake
check_required_tool ctest
check_generator_tools "${GENERATOR}"
MKL_RUNTIME_ROOT=""
case "${USE_MKL}" in
  AUTO|auto|Auto|"")
    MKL_RUNTIME_ROOT="$(detect_mkl_root)"
    if [[ -n "${MKL_RUNTIME_ROOT}" ]] &&
       mkl_supported_by_toolchain "${PLATFORM}" "${GENERATOR}"; then
      USE_MKL="ON"
    else
      USE_MKL="OFF"
    fi
    ;;
  *)
    USE_MKL="$(to_cmake_bool "${USE_MKL}")"
    ;;
esac
if [[ "${USE_MKL}" == "ON" ]] &&
   ! mkl_supported_by_toolchain "${PLATFORM}" "${GENERATOR}"; then
  echo \
    "Intel MKL is not supported by this script with MinGW/GNU on Windows. Use USE_MKL=OFF, omit --mkl, or build with an MSVC/Intel-compatible toolchain." \
    >&2
  exit 2
fi
if [[ "${USE_MKL}" == "ON" ]]; then
  if [[ -z "${MKL_RUNTIME_ROOT}" ]]; then
    MKL_RUNTIME_ROOT="$(detect_mkl_root)"
  fi
  prepend_mkl_runtime_path "${MKL_RUNTIME_ROOT}"
fi
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${CPP_DIR}/build-${PLATFORM}$(generator_suffix "${GENERATOR}")"
fi

mkdir -p "${BUILD_DIR}" "${LOG_DIR}"
: > "${LOG_FILE}"

configure_command=(
  cmake
  -S "${CPP_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${CONFIG}"
  -DNAIVE_BLOCK_LDLT_USE_OPENMP="${USE_OPENMP}"
  -DNAIVE_BLOCK_LDLT_USE_MKL="${USE_MKL}"
)
if [[ -n "${GENERATOR}" ]]; then
  configure_command+=(-G "${GENERATOR}")
fi
if [[ "${USE_MKL}" == "ON" && -n "${MKL_RUNTIME_ROOT}" ]]; then
  configure_command+=(-DNAIVE_BLOCK_LDLT_MKL_ROOT="${MKL_RUNTIME_ROOT}")
fi

build_command=(cmake --build "${BUILD_DIR}" --config "${CONFIG}" --parallel)
test_command=(
  ctest
  --test-dir "${BUILD_DIR}"
  --verbose
  --output-on-failure
  -C "${CONFIG}"
)

log "BlockedBunchKaufmanLDLT Bash CMake test run"
log "Platform: ${PLATFORM}"
log "Generator: ${GENERATOR:-CMake default}"
log "Config: ${CONFIG}"
log "OpenMP: ${USE_OPENMP}"
log "MKL: ${USE_MKL}"
if [[ "${USE_MKL}" == "ON" ]]; then
  log "MKL runtime root: ${MKL_RUNTIME_ROOT:-not found}"
fi
log "Build dir: ${BUILD_DIR}"
log "Log: ${LOG_FILE}"

run_logged "${configure_command[@]}"
run_logged "${build_command[@]}"
run_logged "${test_command[@]}"

log
log "C++ tests passed."
log "Log saved to: ${LOG_FILE}"
