#include <sar/focusing/rd_focuser.hpp>
#include <sar/common/math_utils.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>

namespace sar::focusing {

using namespace sar::math;

RangeCompressor::RangeCompressor(const SARMetadata& meta,
                                 fft::FFTEngine* fft_engine,
                                 bool apply_window,
                                 Real window_beta) {
    configure(meta, fft_engine, apply_window, window_beta);
}

void RangeCompressor::configure(const SARMetadata& meta,
                                fft::FFTEngine* fft_engine,
                                bool apply_window,
                                Real window_beta) {
    meta_ = meta;
    fft_engine_ = fft_engine;
    apply_window_ = apply_window;
    window_beta_ = window_beta;
    build_matched_filter();
    build_window();
}

void RangeCompressor::build_matched_filter() {
    const Int64 N = static_cast<Int64>(meta_.range_samples);
    const Int64 Nfft = next_pow2(N);
    const Real T = meta_.pulse_duration_s;
    const Real B = meta_.bandwidth_hz;
    const Real fs = meta_.range_sampling_hz;

    range_filter_.resize(static_cast<size_t>(Nfft), Complex(0, 0));

    if (T <= 0 || B <= 0 || fs <= 0) {
        for (auto& c : range_filter_) c = Complex(1, 0);
        return;
    }

    std::vector<Complex> chirp(static_cast<size_t>(N), Complex(0, 0));
    const Real K = B / T;
    const Int64 N0 = std::min(N, static_cast<Int64>(T * fs));
    const Real t0 = -T / 2.0f;
    const Real dt = 1.0f / fs;

    for (Int64 i = 0; i < N0; ++i) {
        const Real t = t0 + static_cast<Real>(i) * dt;
        const Real phase = PI * K * t * t;
        Real w = 1.0f;
        if (apply_window_) {
            w = blackman_window(i, N0);
        }
        chirp[static_cast<size_t>(i)] = Complex(std::cos(phase), std::sin(phase)) * w;
    }

    std::vector<Complex> chirp_fft(static_cast<size_t>(Nfft), Complex(0, 0));
    std::copy(chirp.begin(), chirp.end(), chirp_fft.begin());

    if (fft_engine_) {
        fft_engine_->fft_1d(chirp_fft, fft::FFTDirection::FORWARD);
    }

    for (Int64 k = 0; k < Nfft; ++k) {
        const Complex& H = chirp_fft[static_cast<size_t>(k)];
        const Real mag2 = std::norm(H);
        if (mag2 > 1e-20f) {
            range_filter_[static_cast<size_t>(k)] = std::conj(H) / mag2;
        }
    }
}

void RangeCompressor::build_window() {
    if (!apply_window_) return;
    const Int64 N = static_cast<Int64>(meta_.range_samples);
    window_coeffs_.resize(static_cast<size_t>(N));
    for (Int64 i = 0; i < N; ++i) {
        window_coeffs_[static_cast<size_t>(i)] = blackman_window(i, N);
    }
}

void RangeCompressor::compress(ComplexMatrix& echo_data) const {
    if (!fft_engine_) {
        throw InvalidParameterException("FFT engine not configured");
    }
    if (echo_data.rows() == 0 || echo_data.cols() == 0) {
        throw InvalidParameterException("Empty echo data");
    }

    const Int64 rows = static_cast<Int64>(echo_data.rows());
    const Int64 cols = static_cast<Int64>(echo_data.cols());
    const Int64 cols_pow2 = next_pow2(cols);

    if (progress_) {
        progress_->start_stage("Range compression", static_cast<UInt64>(rows));
    }

    echo_data.reset_watermark();

    #pragma omp parallel for schedule(dynamic) num_threads(fft_engine_->num_threads())
    for (Int64 r = 0; r < rows; ++r) {
        auto row_span = echo_data.row(static_cast<UInt64>(r));
        std::vector<Complex> buf(static_cast<size_t>(cols_pow2), Complex(0, 0));

        for (Int64 c = 0; c < cols; ++c) {
            Real w = apply_window_ ? window_coeffs_[static_cast<size_t>(c)] : 1.0f;
            buf[static_cast<size_t>(c)] = row_span[static_cast<size_t>(c)] * w;
        }

        fft_engine_->fft_1d(buf, fft::FFTDirection::FORWARD);

        const Int64 filter_size = static_cast<Int64>(range_filter_.size());
        for (Int64 k = 0; k < cols_pow2; ++k) {
            const Int64 kf = k < filter_size ? k : k % filter_size;
            buf[static_cast<size_t>(k)] *= range_filter_[static_cast<size_t>(kf)];
        }

        fft_engine_->fft_1d(buf, fft::FFTDirection::INVERSE);

        for (Int64 c = 0; c < cols; ++c) {
            row_span[static_cast<size_t>(c)] = buf[static_cast<size_t>(c)];
        }

        echo_data.ensure_row_watermark(static_cast<UInt64>(r));

        if (progress_) progress_->update();
    }

    echo_data.ensure_row_watermark(static_cast<UInt64>(rows - 1));

    if (progress_) {
        progress_->finish_stage();
    }
}

SincInterpolator::SincInterpolator() {
    build_kernel_table();
}

SincInterpolator::SincInterpolator(Int32 kernel_size, Int32 oversampling)
    : kernel_size_(kernel_size)
    , oversampling_(oversampling)
    , half_kernel_(kernel_size / 2) {
    if (kernel_size <= 0 || kernel_size % 2 != 0) {
        throw InvalidParameterException("Sinc kernel size must be positive even integer");
    }
    build_kernel_table();
}

void SincInterpolator::build_kernel_table() {
    const Int32 total_points = (oversampling_ + 1) * kernel_size_;
    kernel_table_.resize(static_cast<size_t>(total_points));

    const Real inv_oversample = 1.0f / static_cast<Real>(oversampling_);

    for (Int32 frac_idx = 0; frac_idx <= oversampling_; ++frac_idx) {
        const Real frac = static_cast<Real>(frac_idx) * inv_oversample;
        Real sum = 0.0f;
        for (Int32 k = 0; k < kernel_size_; ++k) {
            const Real x = static_cast<Real>(k - half_kernel_ + 1) - frac;
            Real w = 1.0f;
            if (kernel_size_ >= 4) {
                const Real win_x = static_cast<Real>(k) / static_cast<Real>(kernel_size_ - 1);
                w = 0.5f - 0.5f * std::cos(TWO_PI * win_x);
            }
            const Real s = sinc(x) * w;
            kernel_table_[static_cast<size_t>(frac_idx * kernel_size_ + k)] = s;
            sum += s;
        }
        if (std::abs(sum) > 1e-10f) {
            const Real inv_sum = 1.0f / sum;
            for (Int32 k = 0; k < kernel_size_; ++k) {
                kernel_table_[static_cast<size_t>(frac_idx * kernel_size_ + k)] *= inv_sum;
            }
        }
    }
}

Complex SincInterpolator::interpolate(std::span<const Complex> data, Real frac_index) const {
    const Int64 n = static_cast<Int64>(data.size());
    const Real fi = std::floor(frac_index);
    const Real frac = frac_index - fi;
    const Int64 center = static_cast<Int64>(fi);
    const Int32 frac_idx = static_cast<Int32>(frac * static_cast<Real>(oversampling_) + 0.5f);
    const Real* const kernel = kernel_table_.data() + static_cast<size_t>(frac_idx * kernel_size_);

    Complex acc(0, 0);
    for (Int32 k = 0; k < kernel_size_; ++k) {
        const Int64 idx = center + k - half_kernel_ + 1;
        const Complex& sample = (idx >= 0 && idx < n) ? data[static_cast<size_t>(idx)] : Complex(0, 0);
        acc += sample * kernel[k];
    }
    return acc;
}

Complex SincInterpolator::interpolate_checked(std::span<const Complex> data, Real frac_index) const {
    if (frac_index < 0 || frac_index > static_cast<Real>(data.size())) {
        throw InvalidParameterException("Fractional index out of range");
    }
    return interpolate(data, frac_index);
}

void SincInterpolator::interpolate_range_line(
    std::span<const Complex> in_line,
    std::span<Complex> out_line,
    std::span<const Real> shifts) const {

    const Int64 n = static_cast<Int64>(in_line.size());
    const Int64 m = static_cast<Int64>(out_line.size());
    const Int64 len = (std::min)({n, m, static_cast<Int64>(shifts.size())});

    for (Int64 i = 0; i < len; ++i) {
        const Real target_idx = static_cast<Real>(i) + shifts[static_cast<size_t>(i)];
        out_line[static_cast<size_t>(i)] = interpolate(in_line, target_idx);
    }
}

RangeCellMigrationCorrector::RangeCellMigrationCorrector(
    const SARMetadata& meta,
    SincInterpolator* interpolator,
    Int32 kernel_size) {
    configure(meta, interpolator, kernel_size);
}

void RangeCellMigrationCorrector::configure(
    const SARMetadata& meta,
    SincInterpolator* interpolator,
    Int32 kernel_size) {
    meta_ = meta;
    interpolator_ = interpolator;
    kernel_size_ = kernel_size;
    if (!interpolator_) {
        throw InvalidParameterException("Sinc interpolator must not be null");
    }
}

Real RangeCellMigrationCorrector::compute_range_shift_m(
    Real range_m,
    Real doppler_centroid,
    Real azimuth_freq) const {
    const Real v = meta_.platform_velocity;
    const Real lambda = meta_.wavelength_m;
    if (v <= 0 || lambda <= 0 || range_m <= 0) return 0.0f;

    const Real fd = doppler_centroid + azimuth_freq;
    const Real R_ref = range_m;
    const Real term = lambda * lambda * fd * fd / (8.0f * v * v);
    return R_ref * term;
}

std::vector<std::vector<Real>> RangeCellMigrationCorrector::compute_shifts(Real doppler_centroid) const {
    const Int64 Naz = static_cast<Int64>(meta_.azimuth_lines);
    const Int64 Nr  = static_cast<Int64>(meta_.range_samples);
    const Real dr = meta_.range_spacing_m;
    const Real dfa = meta_.prf_hz / static_cast<Real>(Naz);

    std::vector<std::vector<Real>> shifts(
        static_cast<size_t>(Naz),
        std::vector<Real>(static_cast<size_t>(Nr))
    );

    #pragma omp parallel for schedule(dynamic)
    for (Int64 a = 0; a < Naz; ++a) {
        Real fa = static_cast<Real>(a) * dfa;
        if (fa > meta_.prf_hz / 2.0f) {
            fa -= meta_.prf_hz;
        }
        auto& row = shifts[static_cast<size_t>(a)];
        for (Int64 r = 0; r < Nr; ++r) {
            const Real range_m = meta_.near_range_m + static_cast<Real>(r) * dr;
            const Real shift_m = compute_range_shift_m(range_m, doppler_centroid, fa);
            row[static_cast<size_t>(r)] = shift_m / dr;
        }
    }
    return shifts;
}

void RangeCellMigrationCorrector::correct(ComplexMatrix& range_compressed) const {
    if (!interpolator_) {
        throw InvalidParameterException("Interpolator not configured");
    }

    const Int64 Naz = static_cast<Int64>(range_compressed.rows());
    const Int64 Nr  = static_cast<Int64>(range_compressed.cols());

    if (progress_) {
        progress_->start_stage("Range Cell Migration Correction", static_cast<UInt64>(Naz * Nr));
    }

    ComplexMatrix az_freq_data(Naz, Nr);
    ComplexMatrix result(Naz, Nr);
    result.reset_watermark();

    for (Int64 r = 0; r < Nr; ++r) {
        std::vector<Complex> col(static_cast<size_t>(Naz));
        for (Int64 a = 0; a < Naz; ++a) {
            col[static_cast<size_t>(a)] = range_compressed.at(static_cast<UInt64>(a), static_cast<UInt64>(r));
        }
        for (Int64 a = 0; a < Naz; ++a) {
            az_freq_data.at(static_cast<UInt64>(a), static_cast<UInt64>(r)) = col[static_cast<size_t>(a)];
        }
    }

    fft::FFTEngine engine;
    engine.fft_cols(az_freq_data, fft::FFTDirection::FORWARD);

    const Real dr = meta_.range_spacing_m;
    const Real dfa = meta_.prf_hz / static_cast<Real>(Naz);
    const Int64 Naz_half = Naz / 2;

    #pragma omp parallel for schedule(dynamic)
    for (Int64 a = 0; a < Naz; ++a) {
        std::vector<Complex> tmp_in(static_cast<size_t>(Nr));
        std::vector<Complex> tmp_out(static_cast<size_t>(Nr));
        std::vector<Real> shifts(static_cast<size_t>(Nr));

        const Int64 a_wrap = (a < Naz_half) ? a : a - Naz;
        Real fa = static_cast<Real>(a_wrap) * dfa;
        const Real dc = meta_.doppler_centroid;

        for (Int64 r = 0; r < Nr; ++r) {
            const Real range_m = meta_.near_range_m + static_cast<Real>(r) * dr;
            const Real shift_m = compute_range_shift_m(range_m, dc, fa);
            shifts[static_cast<size_t>(r)] = shift_m / dr;
        }

        for (Int64 r = 0; r < Nr; ++r) {
            tmp_in[static_cast<size_t>(r)] = az_freq_data.at(static_cast<UInt64>(a), static_cast<UInt64>(r));
        }

        interpolator_->interpolate_range_line(
            std::span<const Complex>(tmp_in.data(), Nr),
            std::span<Complex>(tmp_out.data(), Nr),
            std::span<const Real>(shifts.data(), Nr)
        );

        for (Int64 r = 0; r < Nr; ++r) {
            result.at(static_cast<UInt64>(a), static_cast<UInt64>(r)) = tmp_out[static_cast<size_t>(r)];
        }

        result.ensure_row_watermark(static_cast<UInt64>(a));

        if (progress_) progress_->update(static_cast<UInt64>(Nr));
    }

    result.ensure_row_watermark(static_cast<UInt64>(Naz - 1));

    engine.fft_cols(result, fft::FFTDirection::INVERSE);
    range_compressed = std::move(result);

    if (progress_) {
        progress_->finish_stage();
    }
}

DopplerEstimator::DopplerEstimator(const SARMetadata& meta) {
    configure(meta);
}

void DopplerEstimator::configure(const SARMetadata& meta) {
    meta_ = meta;
}

Real DopplerEstimator::estimate_centroid(
    const ComplexMatrix& range_compressed,
    fft::FFTEngine*) const {

    const Int64 Naz = static_cast<Int64>(range_compressed.rows());
    const Int64 Nr  = static_cast<Int64>(range_compressed.cols());

    Real acf_re = 0.0f, acf_im = 0.0f;
    Real acf_re2 = 0.0f, acf_im2 = 0.0f;
    const Int64 start_lag = 1;
    const Int64 end_lag = 8;
    const Int64 samples = (std::min)(static_cast<Int64>(1024), Naz);

    #pragma omp parallel for reduction(+:acf_re,acf_im,acf_re2,acf_im2) schedule(dynamic)
    for (Int64 a = 0; a < samples - end_lag; ++a) {
        for (Int64 r = 0; r < Nr; r += 16) {
            const Complex& s0 = range_compressed.at(static_cast<UInt64>(a), static_cast<UInt64>(r));
            const Complex& s1 = range_compressed.at(static_cast<UInt64>(a + start_lag), static_cast<UInt64>(r));
            const Complex& s2 = range_compressed.at(static_cast<UInt64>(a + end_lag), static_cast<UInt64>(r));
            const Complex prod1 = s0 * std::conj(s1);
            const Complex prod2 = s0 * std::conj(s2);
            acf_re += prod1.real();
            acf_im += prod1.imag();
            acf_re2 += prod2.real();
            acf_im2 += prod2.imag();
        }
    }

    const Complex acf_sum(acf_re, acf_im);
    const Complex acf_sum2(acf_re2, acf_im2);

    const Real phase1 = std::arg(acf_sum);
    (void)phase1;
    const Real phaseN = std::arg(acf_sum2);
    const Real dphi = phaseN / static_cast<Real>(end_lag);
    Real fd = dphi * meta_.prf_hz / TWO_PI;

    fd = std::fmod(fd + meta_.prf_hz / 2.0f, meta_.prf_hz) - meta_.prf_hz / 2.0f;

    if (progress_) {
        progress_->start_stage("Doppler estimation", 1);
        progress_->update(1);
        progress_->finish_stage();
    }

    return fd;
}

std::vector<Real> DopplerEstimator::estimate_spectrum(
    const ComplexMatrix& range_compressed,
    fft::FFTEngine* fft_engine) const {

    const Int64 Naz = static_cast<Int64>(range_compressed.rows());
    const Int64 Nr  = static_cast<Int64>(range_compressed.cols());
    const Int64 Nfft = next_pow2(Naz);

    std::vector<Complex> buf(static_cast<size_t>(Nfft), Complex(0, 0));
    std::vector<Real> spectrum(static_cast<size_t>(Nfft), 0.0f);

    const Int64 r_step = std::max<Int64>(1, Nr / 128);
    for (Int64 r = 0; r < Nr; r += r_step) {
        for (Int64 a = 0; a < Naz; ++a) {
            buf[static_cast<size_t>(a)] = range_compressed.at(static_cast<UInt64>(a), static_cast<UInt64>(r));
        }
        fft_engine->fft_1d(buf, fft::FFTDirection::FORWARD);
        for (Int64 k = 0; k < Nfft; ++k) {
            spectrum[static_cast<size_t>(k)] += std::norm(buf[static_cast<size_t>(k)]);
        }
    }
    return spectrum;
}

AzimuthCompressor::AzimuthCompressor(
    const SARMetadata& meta,
    fft::FFTEngine* fft_engine,
    bool apply_window,
    Real window_beta) {
    configure(meta, fft_engine, apply_window, window_beta);
}

void AzimuthCompressor::configure(
    const SARMetadata& meta,
    fft::FFTEngine* fft_engine,
    bool apply_window,
    Real window_beta) {
    meta_ = meta;
    fft_engine_ = fft_engine;
    apply_window_ = apply_window;
    window_beta_ = window_beta;
    doppler_centroid_ = meta.doppler_centroid;
}

void AzimuthCompressor::build_azimuth_window(Int64 n) {
    azimuth_window_.resize(static_cast<size_t>(n));
    for (Int64 i = 0; i < n; ++i) {
        azimuth_window_[static_cast<size_t>(i)] = blackman_window(i, n);
    }
}

void AzimuthCompressor::build_reference_function(
    std::vector<Complex>& ref_func,
    Real range_m,
    Int64 azimuth_fft_size) const {

    ref_func.resize(static_cast<size_t>(azimuth_fft_size), Complex(0, 0));
    if (range_m <= 0 || meta_.platform_velocity <= 0 || meta_.wavelength_m <= 0) {
        for (auto& c : ref_func) c = Complex(1, 0);
        return;
    }

    const Real v = meta_.platform_velocity;
    const Real lambda = meta_.wavelength_m;
    const Real R0 = range_m;
    const Real Ka = 2.0f * v * v / (lambda * R0);
    const Real prf = meta_.prf_hz;
    const Real dfa = prf / static_cast<Real>(azimuth_fft_size);
    const Real fd_c = doppler_centroid_;

    const Int64 Naz_half = azimuth_fft_size / 2;
    for (Int64 k = 0; k < azimuth_fft_size; ++k) {
        const Int64 k_wrap = (k < Naz_half) ? k : k - azimuth_fft_size;
        const Real fa = static_cast<Real>(k_wrap) * dfa;
        const Real fa_centered = fa - fd_c;
        const Real phase = -PI * fa_centered * fa_centered / Ka;
        ref_func[static_cast<size_t>(k)] = Complex(std::cos(phase), std::sin(phase));
    }
}

void AzimuthCompressor::compress(ComplexMatrix& range_rcmc_data) {
    if (!fft_engine_) {
        throw InvalidParameterException("FFT engine not configured");
    }

    const Int64 Naz = static_cast<Int64>(range_rcmc_data.rows());
    const Int64 Nr  = static_cast<Int64>(range_rcmc_data.cols());
    const Int64 Naz_fft = next_pow2(Naz);

    if (apply_window_) {
        build_azimuth_window(Naz);
    }

    if (progress_) {
        progress_->start_stage("Azimuth compression", static_cast<UInt64>(Nr));
    }

    range_rcmc_data.reset_watermark();

    #pragma omp parallel for schedule(dynamic) num_threads(fft_engine_->num_threads())
    for (Int64 r = 0; r < Nr; ++r) {
        std::vector<Complex> ref_func(static_cast<size_t>(Naz_fft));
        std::vector<Complex> col_buf(static_cast<size_t>(Naz_fft), Complex(0, 0));

        const Real range_m = meta_.near_range_m + static_cast<Real>(r) * meta_.range_spacing_m;

        for (Int64 a = 0; a < Naz; ++a) {
            Complex val = range_rcmc_data.at(static_cast<UInt64>(a), static_cast<UInt64>(r));
            if (apply_window_) {
                val *= azimuth_window_[static_cast<size_t>(a)];
            }
            col_buf[static_cast<size_t>(a)] = val;
        }
        for (Int64 a = Naz; a < Naz_fft; ++a) {
            col_buf[static_cast<size_t>(a)] = Complex(0, 0);
        }

        build_reference_function(ref_func, range_m, Naz_fft);

        fft_engine_->fft_1d(col_buf, fft::FFTDirection::FORWARD);

        for (Int64 k = 0; k < Naz_fft; ++k) {
            col_buf[static_cast<size_t>(k)] *= ref_func[static_cast<size_t>(k)];
        }

        fft_engine_->fft_1d(col_buf, fft::FFTDirection::INVERSE);

        for (Int64 a = 0; a < Naz; ++a) {
            range_rcmc_data.at(static_cast<UInt64>(a), static_cast<UInt64>(r)) = col_buf[static_cast<size_t>(a)];
        }

        if (progress_) progress_->update();
    }

    range_rcmc_data.ensure_row_watermark(static_cast<UInt64>(Naz - 1));

    if (progress_) {
        progress_->finish_stage();
    }
}

}
