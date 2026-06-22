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

template<typename T>
class Tensor2D {
public:
    Tensor2D() = default;

    Tensor2D(UInt64 rows, UInt64 cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    Tensor2D(UInt64 rows, UInt64 cols, T init_val)
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    inline UInt64 rows() const noexcept { return rows_; }
    inline UInt64 cols() const noexcept { return cols_; }
    inline UInt64 size() const noexcept { return data_.size(); }

    inline T& at(UInt64 row, UInt64 col) {
        return data_[row * cols_ + col];
    }

    inline const T& at(UInt64 row, UInt64 col) const {
        return data_[row * cols_ + col];
    }

    inline T* data() noexcept { return data_.data(); }
    inline const T* data() const noexcept { return data_.data(); }

    inline std::span<T> row(UInt64 r) {
        return std::span<T>(data_.data() + r * cols_, cols_);
    }

    inline std::span<const T> row(UInt64 r) const {
        return std::span<const T>(data_.data() + r * cols_, cols_);
    }

    inline void resize(UInt64 rows, UInt64 cols) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols);
    }

    inline void fill(T val) {
        std::fill(data_.begin(), data_.end(), val);
    }

    inline void clear() {
        rows_ = 0;
        cols_ = 0;
        data_.clear();
    }

    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

private:
    UInt64 rows_ = 0;
    UInt64 cols_ = 0;
    std::vector<T> data_;
};

using ComplexMatrix = Tensor2D<Complex>;
using RealMatrix    = Tensor2D<Real>;
using Int16Matrix   = Tensor2D<Int16>;

}
