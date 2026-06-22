#pragma once

#include <complex>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <span>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>
#include <cstring>

namespace sar {

using Real    = float;
using Complex = std::complex<Real>;
using Int8    = int8_t;
using Int16   = int16_t;
using Int32   = int32_t;
using Int64   = int64_t;
using UInt8   = uint8_t;
using UInt16  = uint16_t;
using UInt32  = uint32_t;
using UInt64  = uint64_t;

constexpr UInt64 SAR_CACHE_LINE = 64;

template<typename T>
using Matrix2D = std::vector<std::vector<T>>;

template<typename T>
using Ptr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

enum class SampleFormat {
    COMPLEX_INT8,
    COMPLEX_INT16,
    COMPLEX_INT32,
    COMPLEX_FLOAT32,
    COMPLEX_FLOAT64,
    RAW_PACKED_BITS
};

enum class Polarization {
    HH,
    HV,
    VH,
    VV
};

struct SARMetadata {
    UInt64  azimuth_lines      = 0;
    UInt64  range_samples      = 0;
    Real    carrier_freq_hz    = 0.0;
    Real    bandwidth_hz       = 0.0;
    Real    pulse_duration_s   = 0.0;
    Real    prf_hz             = 0.0;
    Real    range_sampling_hz  = 0.0;
    Real    platform_velocity  = 0.0;
    Real    wavelength_m       = 0.0;
    Real    near_range_m       = 0.0;
    Real    range_spacing_m    = 0.0;
    Real    azimuth_spacing_m  = 0.0;
    Real    doppler_centroid   = 0.0;
    Polarization polarization  = Polarization::HH;
    SampleFormat sample_format = SampleFormat::COMPLEX_INT16;
    UInt32  bits_per_sample    = 16;
    std::string scene_id;
    std::string acquisition_date;
};

inline UInt64 align_up(UInt64 v, UInt64 align) noexcept {
    return (v + align - 1) & ~(align - 1);
}

inline bool is_cache_aligned(const void* p) noexcept {
    return (reinterpret_cast<UInt64>(p) & (SAR_CACHE_LINE - 1)) == 0;
}

template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        UInt64 total = static_cast<UInt64>(n) * sizeof(T);
        UInt64 aligned_total = align_up(total, SAR_CACHE_LINE);
        void* ptr = ::operator new(aligned_total, std::align_val_t{SAR_CACHE_LINE});
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        ::operator delete(ptr, std::align_val_t{SAR_CACHE_LINE});
    }

    template<typename U>
    struct rebind { using other = AlignedAllocator<U>; };
};

template<typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept { return true; }

template<typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept { return false; }

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

template<typename T>
class Tensor2D {
public:
    Tensor2D() = default;

    Tensor2D(UInt64 rows, UInt64 cols)
        : rows_(rows), cols_(cols)
    {
        allocate_aligned();
    }

    Tensor2D(UInt64 rows, UInt64 cols, T init_val)
        : rows_(rows), cols_(cols)
    {
        allocate_aligned();
        std::fill(data_.begin(), data_.end(), init_val);
    }

    Tensor2D(const Tensor2D& other)
        : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_), data_(other.data_)
        , write_watermark_(other.write_watermark_.load(std::memory_order_relaxed))
    {}

    Tensor2D& operator=(const Tensor2D& other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            data_ = other.data_;
            write_watermark_.store(other.write_watermark_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    Tensor2D(Tensor2D&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_), data_(std::move(other.data_))
        , write_watermark_(other.write_watermark_.load(std::memory_order_relaxed))
    {
        other.rows_ = 0;
        other.cols_ = 0;
        other.stride_ = 0;
        other.write_watermark_.store(0, std::memory_order_relaxed);
    }

    Tensor2D& operator=(Tensor2D&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            data_ = std::move(other.data_);
            write_watermark_.store(other.write_watermark_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.rows_ = 0;
            other.cols_ = 0;
            other.stride_ = 0;
            other.write_watermark_.store(0, std::memory_order_relaxed);
        }
        return *this;
    }

    inline UInt64 rows() const noexcept { return rows_; }
    inline UInt64 cols() const noexcept { return cols_; }
    inline UInt64 size() const noexcept { return data_.size(); }
    inline UInt64 stride() const noexcept { return stride_; }

    inline T& at(UInt64 row, UInt64 col) {
        return data_[row * stride_ + col];
    }

    inline const T& at(UInt64 row, UInt64 col) const {
        return data_[row * stride_ + col];
    }

    inline T* data() noexcept { return data_.data(); }
    inline const T* data() const noexcept { return data_.data(); }

    inline std::span<T> row(UInt64 r) {
        return std::span<T>(data_.data() + r * stride_, cols_);
    }

    inline std::span<const T> row(UInt64 r) const {
        return std::span<const T>(data_.data() + r * stride_, cols_);
    }

    inline bool is_row_aligned(UInt64 r) const noexcept {
        return is_cache_aligned(data_.data() + r * stride_);
    }

    inline void resize(UInt64 rows, UInt64 cols) {
        rows_ = rows;
        cols_ = cols;
        allocate_aligned();
    }

    inline void fill(T val) {
        std::fill(data_.begin(), data_.end(), val);
    }

    inline void clear() {
        rows_ = 0;
        cols_ = 0;
        stride_ = 0;
        data_.clear();
        data_.shrink_to_fit();
    }

    void ensure_row_watermark(UInt64 safe_row) {
        UInt64 expected = safe_row + 1;
        UInt64 current = write_watermark_.load(std::memory_order_acquire);
        while (current < expected) {
            if (write_watermark_.compare_exchange_weak(current, expected,
                    std::memory_order_release, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    void wait_until_row_ready(UInt64 row) const {
        while (write_watermark_.load(std::memory_order_acquire) <= row) {
            std::this_thread::yield();
        }
    }

    UInt64 write_watermark() const noexcept {
        return write_watermark_.load(std::memory_order_acquire);
    }

    void reset_watermark() noexcept {
        write_watermark_.store(0, std::memory_order_release);
    }

    typename AlignedVector<T>::iterator begin() { return data_.begin(); }
    typename AlignedVector<T>::iterator end() { return data_.end(); }
    typename AlignedVector<T>::const_iterator begin() const { return data_.begin(); }
    typename AlignedVector<T>::const_iterator end() const { return data_.end(); }

private:
    void allocate_aligned() {
        stride_ = align_up(cols_, SAR_CACHE_LINE / sizeof(T));
        if (stride_ == 0) stride_ = cols_;
        if (stride_ < cols_) stride_ = cols_;
        data_.resize(static_cast<size_t>(rows_ * stride_));
    }

    UInt64 rows_ = 0;
    UInt64 cols_ = 0;
    UInt64 stride_ = 0;
    AlignedVector<T> data_;
    std::atomic<UInt64> write_watermark_{0};
};

using ComplexMatrix = Tensor2D<Complex>;
using RealMatrix    = Tensor2D<Real>;
using Int16Matrix   = Tensor2D<Int16>;

}
