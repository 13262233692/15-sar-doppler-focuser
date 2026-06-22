#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <sar/common/math_utils.hpp>
#include <sar/common/progress.hpp>
#include <sar/fft/fft_engine.hpp>
#include <vector>
#include <span>

namespace sar::focusing {

class RangeCompressor {
public:
    RangeCompressor() = default;

    RangeCompressor(const SARMetadata& meta,
                    fft::FFTEngine* fft_engine,
                    bool apply_window = true,
                    Real window_beta = 25.0);

    void configure(const SARMetadata& meta,
                   fft::FFTEngine* fft_engine,
                   bool apply_window = true,
                   Real window_beta = 25.0);

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void compress(ComplexMatrix& echo_data) const;

    const std::vector<Complex>& matched_filter() const noexcept { return range_filter_; }

    Int64 filter_size() const noexcept { return static_cast<Int64>(range_filter_.size()); }

private:
    void build_matched_filter();
    void build_window();

    SARMetadata    meta_;
    fft::FFTEngine* fft_engine_ = nullptr;
    bool            apply_window_ = true;
    Real            window_beta_  = 25.0;
    std::vector<Complex> range_filter_;
    std::vector<Real>    window_coeffs_;
    ProgressReporter*    progress_ = nullptr;
};

class SincInterpolator {
public:
    static constexpr Int32 DEFAULT_KERNEL_SIZE = 8;

    SincInterpolator();
    explicit SincInterpolator(Int32 kernel_size, Int32 oversampling = 1024);

    Int32 kernel_size() const noexcept { return kernel_size_; }
    Int32 oversampling() const noexcept { return oversampling_; }

    Complex interpolate(std::span<const Complex> data, Real frac_index) const;

    Complex interpolate_checked(std::span<const Complex> data, Real frac_index) const;

    void interpolate_range_line(std::span<const Complex> in_line,
                                std::span<Complex> out_line,
                                std::span<const Real> shifts) const;

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    const std::vector<Real>& kernel_table() const noexcept { return kernel_table_; }

private:
    void build_kernel_table();

    Int32 kernel_size_  = DEFAULT_KERNEL_SIZE;
    Int32 oversampling_ = 1024;
    Int32 half_kernel_  = DEFAULT_KERNEL_SIZE / 2;
    std::vector<Real> kernel_table_;
    ProgressReporter* progress_ = nullptr;
};

class RangeCellMigrationCorrector {
public:
    RangeCellMigrationCorrector() = default;

    RangeCellMigrationCorrector(const SARMetadata& meta,
                                SincInterpolator* interpolator,
                                Int32 kernel_size = 8);

    void configure(const SARMetadata& meta,
                   SincInterpolator* interpolator,
                   Int32 kernel_size = 8);

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void correct(ComplexMatrix& range_compressed) const;

    std::vector<std::vector<Real>> compute_shifts(Real doppler_centroid) const;

    const SincInterpolator& interpolator() const noexcept { return *interpolator_; }

private:
    Real compute_range_shift_m(Real range_m, Real doppler_centroid, Real azimuth_freq) const;

    SARMetadata       meta_;
    SincInterpolator* interpolator_ = nullptr;
    Int32             kernel_size_  = 8;
    ProgressReporter* progress_     = nullptr;
};

class DopplerEstimator {
public:
    DopplerEstimator() = default;
    explicit DopplerEstimator(const SARMetadata& meta);

    void configure(const SARMetadata& meta);

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    Real estimate_centroid(const ComplexMatrix& range_compressed,
                           fft::FFTEngine* fft_engine) const;

    std::vector<Real> estimate_spectrum(const ComplexMatrix& range_compressed,
                                        fft::FFTEngine* fft_engine) const;

private:
    SARMetadata    meta_;
    ProgressReporter* progress_ = nullptr;
};

class AzimuthCompressor {
public:
    AzimuthCompressor() = default;

    AzimuthCompressor(const SARMetadata& meta,
                      fft::FFTEngine* fft_engine,
                      bool apply_window = true,
                      Real window_beta = 25.0);

    void configure(const SARMetadata& meta,
                   fft::FFTEngine* fft_engine,
                   bool apply_window = true,
                   Real window_beta = 25.0);

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void set_doppler_centroid(Real doppler_hz) { doppler_centroid_ = doppler_hz; }
    Real doppler_centroid() const noexcept { return doppler_centroid_; }

    void compress(ComplexMatrix& range_rcmc_data);

    void build_reference_function(std::vector<Complex>& ref_func,
                                  Real range_m,
                                  Int64 azimuth_fft_size) const;

private:
    void build_azimuth_window(Int64 n);

    SARMetadata    meta_;
    fft::FFTEngine* fft_engine_ = nullptr;
    bool            apply_window_ = true;
    Real            window_beta_  = 25.0;
    Real            doppler_centroid_ = 0.0;
    std::vector<Real> azimuth_window_;
    ProgressReporter* progress_ = nullptr;
};

}
