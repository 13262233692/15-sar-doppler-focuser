#pragma once

#include <sar/common/types.hpp>
#include <sar/common/config.hpp>
#include <sar/common/progress.hpp>
#include <sar/common/exceptions.hpp>
#include <sar/common/math_utils.hpp>
#include <array>
#include <vector>
#include <complex>
#include <cmath>
#include <mutex>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace sar::polsar {

using namespace sar::math;

struct PolSARQuadChannel {
    ComplexMatrix hh;
    ComplexMatrix hv;
    ComplexMatrix vh;
    ComplexMatrix vv;
};

struct HalphaPixel {
    Real entropy_H      = 0.0f;
    Real anisotropy_A   = 0.0f;
    Real alpha_mean     = 0.0f;
    Real lambda_1       = 0.0f;
    Real lambda_2       = 0.0f;
    Real lambda_3       = 0.0f;
    bool oil_suspected  = false;
};

using HalphaMatrix = Tensor2D<HalphaPixel>;

struct OilSpillAlert {
    UInt64 row          = 0;
    UInt64 col          = 0;
    Real   entropy_H    = 0.0f;
    Real   alpha_mean   = 0.0f;
    Real   delta_H      = 0.0f;
    Real   delta_alpha  = 0.0f;
    Real   lambda_1     = 0.0f;
    Real   lambda_2     = 0.0f;
    Real   lambda_3     = 0.0f;
};

struct InterventionVector {
    std::vector<UInt64> oil_rows;
    std::vector<UInt64> oil_cols;
    std::vector<Real>   oil_H;
    std::vector<Real>   oil_alpha;
    mutable std::mutex  mutex;
    std::atomic<bool>   has_oil{false};
    std::atomic<UInt64> total_oil_pixels{0};

    void add(UInt64 r, UInt64 c, Real H, Real alpha) {
        std::lock_guard<std::mutex> lock(mutex);
        oil_rows.push_back(r);
        oil_cols.push_back(c);
        oil_H.push_back(H);
        oil_alpha.push_back(alpha);
        has_oil.store(true, std::memory_order_release);
        total_oil_pixels.fetch_add(1, std::memory_order_relaxed);
    }

    UInt64 size() const { return total_oil_pixels.load(std::memory_order_acquire); }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        oil_rows.clear();
        oil_cols.clear();
        oil_H.clear();
        oil_alpha.clear();
        has_oil.store(false, std::memory_order_release);
        total_oil_pixels.store(0, std::memory_order_release);
    }
};

class CloudePottierDecomposer {
public:
    static constexpr Real OIL_ENTROPY_THRESHOLD = 0.80f;
    static constexpr Real OIL_ALPHA_THRESHOLD   = 0.80f;

    CloudePottierDecomposer() = default;
    explicit CloudePottierDecomposer(const ProcessingConfig& config);

    void configure(const ProcessingConfig& config);
    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void decompose(const PolSARQuadChannel& channels,
                   HalphaMatrix& out_halpha,
                   InterventionVector& intervention) const;

    static void compute_coherency_matrix(
        const Complex& S_hh, const Complex& S_hv,
        const Complex& S_vh, const Complex& S_vv,
        std::array<std::array<Complex, 3>, 3>& T3) noexcept;

    static bool eigen_decompose_3x3_hermitian(
        const std::array<std::array<Complex, 3>, 3>& T3,
        std::array<Real, 3>& eigenvalues,
        std::array<std::array<Complex, 3>, 3>& eigenvectors) noexcept;

    static Real compute_entropy(const std::array<Real, 3>& eigenvalues) noexcept;
    static Real compute_anisotropy(const std::array<Real, 3>& eigenvalues) noexcept;
    static Real compute_mean_alpha(
        const std::array<Real, 3>& eigenvalues,
        const std::array<std::array<Complex, 3>, 3>& eigenvectors) noexcept;

    static bool is_oil_spill_candidate(Real entropy_H, Real alpha_mean) noexcept {
        return entropy_H >= OIL_ENTROPY_THRESHOLD && alpha_mean >= OIL_ALPHA_THRESHOLD;
    }

    static std::string format_alert(const OilSpillAlert& alert);

private:
    ProcessingConfig config_;
    ProgressReporter* progress_ = nullptr;
};

class PolSARPipeline {
public:
    PolSARPipeline() = default;
    explicit PolSARPipeline(const SARConfig& config);

    void configure(const SARConfig& config);
    void set_progress_reporter(ProgressReporter* reporter);

    void load_quad_polar_channels(
        const std::string& hh_file,
        const std::string& hv_file,
        const std::string& vh_file,
        const std::string& vv_file,
        UInt64 header_bytes = 0);

    void run_decomposition();

    void render_oil_overlay(
        std::vector<UInt8>& r_band,
        std::vector<UInt8>& g_band,
        std::vector<UInt8>& b_band,
        UInt64 rows, UInt64 cols) const;

    const HalphaMatrix& halpha() const noexcept { return halpha_; }
    const InterventionVector& intervention() const noexcept { return intervention_; }
    InterventionVector& intervention() noexcept { return intervention_; }
    const PolSARQuadChannel& channels() const noexcept { return channels_; }

    UInt64 oil_pixel_count() const { return intervention_.size(); }
    bool has_oil_spill() const { return intervention_.has_oil.load(std::memory_order_acquire); }

    void emit_blocking_alerts(std::ostream& os = std::cerr) const;

private:
    SARConfig config_;
    ProgressReporter* progress_ = nullptr;
    CloudePottierDecomposer decomposer_;

    PolSARQuadChannel channels_;
    HalphaMatrix halpha_;
    InterventionVector intervention_;
    bool channels_loaded_ = false;
    bool decomposed_ = false;
};

}
