#ifndef NAIVE_BLOCK_LDLT_ALIGNED_BUFFER_H_
#define NAIVE_BLOCK_LDLT_ALIGNED_BUFFER_H_

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace naive_block_ldlt {

inline constexpr std::size_t kDefaultAlignment = 64;

class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(std::size_t size)
      : AlignedBuffer(size, kDefaultAlignment) {}

  AlignedBuffer(std::size_t size, std::size_t alignment)
      : size_(size), alignment_(alignment) {
    Allocate();
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept { MoveFrom(std::move(other)); }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  ~AlignedBuffer() { Free(); }

  void Resize(std::size_t size) {
    if (size == size_) {
      return;
    }
    Free();
    size_ = size;
    Allocate();
  }

  void Fill(double value) {
    for (std::size_t i = 0; i < size_; ++i) {
      data_[i] = value;
    }
  }

  double* data() { return data_; }
  const double* data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t alignment() const { return alignment_; }

 private:
  void Allocate() {
    if (size_ == 0) {
      data_ = nullptr;
      return;
    }

#if defined(_WIN32)
    data_ = static_cast<double*>(
        _aligned_malloc(size_ * sizeof(double), alignment_));
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
#else
    void* pointer = nullptr;
    if (posix_memalign(&pointer, alignment_, size_ * sizeof(double)) != 0) {
      throw std::bad_alloc();
    }
    data_ = static_cast<double*>(pointer);
#endif
    std::memset(data_, 0, size_ * sizeof(double));
  }

  void Free() {
    if (data_ == nullptr) {
      return;
    }
#if defined(_WIN32)
    _aligned_free(data_);
#else
    std::free(data_);
#endif
    data_ = nullptr;
    size_ = 0;
  }

  void MoveFrom(AlignedBuffer&& other) {
    data_ = other.data_;
    size_ = other.size_;
    alignment_ = other.alignment_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  double* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = kDefaultAlignment;
};

}  // namespace naive_block_ldlt

#endif  // NAIVE_BLOCK_LDLT_ALIGNED_BUFFER_H_
