#include <sar/polsar/polsar_decomposer.hpp>
#include <sar/common/math_utils.hpp>
#include <sar/io/mapped_file_reader.hpp>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace sar::polsar {

using namespace sar::math;

CloudePottierDecomposer::CloudePottierDecomposer(const ProcessingConfig& config) {
    configure(config);
}

void CloudePottierDecomposer::configure(const ProcessingConfig& config) {
    config_ = config;
}

void CloudePottierDecomposer::compute_coherency_matrix(
    const Complex& S_hh, const Complex& S_hv,
    const Complex& S_vh, const Complex& S_vv,
    std::array<std::array<Complex, 3>, 3>& T3) noexcept {

    const Complex k1 = S_hh + S_vv;
    const Complex k2 = S_hh - S_vv;
    const Complex k3 = Complex(2.0f, 0.0f) * S_hv;

    const std::array<Complex, 3> k = {k1, k2, k3};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T3[i][j] = k[i] * std::conj(k[j]);
        }
    }
}

bool CloudePottierDecomposer::eigen_decompose_3x3_hermitian(
    const std::array<std::array<Complex, 3>, 3>& T3,
    std::array<Real, 3>& eigenvalues,
    std::array<std::array<Complex, 3>, 3>& eigenvectors) noexcept {

    const Real a11 = T3[0][0].real();
    const Real a22 = T3[1][1].real();
    const Real a33 = T3[2][2].real();
    const Complex a12 = T3[0][1];
    const Complex a13 = T3[0][2];
    const Complex a23 = T3[1][2];

    const Real a12r = a12.real(), a12i = a12.imag();
    const Real a13r = a13.real(), a13i = a13.imag();
    const Real a23r = a23.real(), a23i = a23.imag();

    const Real trace = a11 + a22 + a33;
    const Real a11a22 = a11 * a22;
    const Real a11a33 = a11 * a33;
    const Real a22a33 = a22 * a33;
    const Real mag12 = a12r*a12r + a12i*a12i;
    const Real mag13 = a13r*a13r + a13i*a13i;
    const Real mag23 = a23r*a23r + a23i*a23i;
    const Real S2 = a11a22 + a11a33 + a22a33 - mag12 - mag13 - mag23;

    const Real detA =
        a11*(a22*a33 - mag23)
      - a12r*(a12r*a33 - a13r*a23r - a13i*a23i)
      + a12i*(a12i*a33 - a13r*a23i + a13i*a23r)
      - a13r*(a13r*a22 - a12r*a23r + a12i*a23i)
      + a13i*(a13i*a22 - a12r*a23i - a12i*a23r);

    const Real p1 = mag12 + mag13 + mag23;
    const Real p = (trace*trace - 3.0f*S2) / 9.0f;
    const Real q = (2.0f*trace*trace*trace - 9.0f*trace*S2 + 27.0f*detA) / 54.0f;
    const Real discriminant = q*q - p*p*p;

    Real phi = 0.0f;
    if (discriminant <= 0.0f || p <= 1e-20f) {
        if (p <= 1e-20f) {
            phi = 0.0f;
        } else {
            Real ratio = q / std::sqrt(p*p*p);
            ratio = std::max(-1.0f, std::min(1.0f, ratio));
            phi = std::acos(ratio) / 3.0f;
        }
    } else {
        phi = 0.0f;
    }

    const Real sqrt_p = std::sqrt(std::max(0.0f, p));
    const Real c1 = 2.0f * sqrt_p;
    const Real c2 = trace / 3.0f;

    Real l1 = c2 + c1 * std::cos(phi);
    Real l2 = c2 + c1 * std::cos(phi + TWO_PI / 3.0f);
    Real l3 = c2 + c1 * std::cos(phi + 2.0f * TWO_PI / 3.0f);

    l1 = std::max(0.0f, l1);
    l2 = std::max(0.0f, l2);
    l3 = std::max(0.0f, l3);

    if (l1 < l2) std::swap(l1, l2);
    if (l1 < l3) std::swap(l1, l3);
    if (l2 < l3) std::swap(l2, l3);

    eigenvalues[0] = l1;
    eigenvalues[1] = l2;
    eigenvalues[2] = l3;

    auto compute_eigenvector = [&](Real lambda, std::array<Complex, 3>& ev) -> bool {
        std::array<std::array<Complex, 3>, 3> M = T3;
        for (int i = 0; i < 3; ++i) {
            M[i][i] -= Complex(lambda, 0.0f);
        }

        int pivot = -1;
        Real max_mag = 0.0f;
        for (int i = 0; i < 3; ++i) {
            Real m = std::abs(M[0][i]);
            if (m > max_mag) { max_mag = m; pivot = i; }
        }
        if (max_mag < 1e-15f) {
            for (int i = 1; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Real m = std::abs(M[i][j]);
                    if (m > max_mag) { max_mag = m; pivot = j; }
                }
            }
            if (max_mag < 1e-15f) return false;
        }

        int row_idx = 0;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(M[i][pivot]) > 1e-15f) { row_idx = i; break; }
        }

        int c1 = (pivot + 1) % 3;
        int c2 = (pivot + 2) % 3;

        Complex m_pc1 = M[row_idx][c1];
        Complex m_pc2 = M[row_idx][c2];
        Complex m_pp  = M[row_idx][pivot];

        if (std::abs(m_pp) < 1e-20f) {
            ev[pivot] = Complex(1.0f, 0.0f);
            ev[c1] = Complex(0.0f, 0.0f);
            ev[c2] = Complex(0.0f, 0.0f);
        } else {
            ev[c1] = Complex(1.0f, 0.0f);
            ev[c2] = Complex(0.0f, 1.0f);
            ev[pivot] = -(m_pc1 * ev[c1] + m_pc2 * ev[c2]) / m_pp;
        }

        Real nrm = std::sqrt(std::norm(ev[0]) + std::norm(ev[1]) + std::norm(ev[2]));
        if (nrm < 1e-20f) {
            ev[0] = Complex(1.0f, 0.0f);
            ev[1] = Complex(0.0f, 0.0f);
            ev[2] = Complex(0.0f, 0.0f);
            nrm = 1.0f;
        }
        for (int i = 0; i < 3; ++i) ev[i] /= nrm;

        if (std::abs(ev[0]) > 1e-20f) {
            Real phase = std::arg(ev[0]);
            Complex phasor = std::exp(Complex(0.0f, -phase));
            for (int i = 0; i < 3; ++i) ev[i] *= phasor;
        }

        return true;
    };

    bool ok = true;
    ok &= compute_eigenvector(l1, eigenvectors[0]);
    ok &= compute_eigenvector(l2, eigenvectors[1]);
    ok &= compute_eigenvector(l3, eigenvectors[2]);

    return ok;
}

Real CloudePottierDecomposer::compute_entropy(const std::array<Real, 3>& eigenvalues) noexcept {
    Real total = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    if (total <= 1e-30f) return 0.0f;

    const Real p1 = eigenvalues[0] / total;
    const Real p2 = eigenvalues[1] / total;
    const Real p3 = eigenvalues[2] / total;

    Real H = 0.0f;
    if (p1 > 1e-20f) H -= p1 * std::log2(p1);
    if (p2 > 1e-20f) H -= p2 * std::log2(p2);
    if (p3 > 1e-20f) H -= p3 * std::log2(p3);
    H /= std::log2(3.0f);

    return clamp(H, 0.0f, 1.0f);
}

Real CloudePottierDecomposer::compute_anisotropy(const std::array<Real, 3>& eigenvalues) noexcept {
    const Real l2 = eigenvalues[1];
    const Real l3 = eigenvalues[2];
    const Real denom = l2 + l3;
    if (denom <= 1e-30f) return 0.0f;
    return clamp((l2 - l3) / denom, 0.0f, 1.0f);
}

Real CloudePottierDecomposer::compute_mean_alpha(
    const std::array<Real, 3>& eigenvalues,
    const std::array<std::array<Complex, 3>, 3>& eigenvectors) noexcept {

    Real total = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    if (total <= 1e-30f) return 0.0f;

    Real alpha_mean = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const Real p = eigenvalues[i] / total;
        const Complex t1i = eigenvectors[i][0];
        const Real mag = std::abs(t1i);
        const Real alpha_i = std::acos(clamp(mag, 0.0f, 1.0f));
        alpha_mean += p * alpha_i;
    }

    alpha_mean *= 180.0f / PI;
    return clamp(alpha_mean / 90.0f, 0.0f, 1.0f);
}

std::string CloudePottierDecomposer::format_alert(const OilSpillAlert& alert) {
    std::ostringstream oss;
    oss << "[POLSAR-OIL-SPILL-BLOCKING-ALERT]"
        << " pixel(" << alert.row << "," << alert.col << ")"
        << " H=" << std::fixed << std::setprecision(4) << alert.entropy_H
        << " alpha=" << std::fixed << std::setprecision(4) << alert.alpha_mean
        << " dH=" << std::fixed << std::setprecision(4) << alert.delta_H
        << " dAlpha=" << std::fixed << std::setprecision(4) << alert.delta_alpha
        << " lambda=[" << alert.lambda_1 << "," << alert.lambda_2 << "," << alert.lambda_3 << "]";
    return oss.str();
}

void CloudePottierDecomposer::decompose(
    const PolSARQuadChannel& channels,
    HalphaMatrix& out_halpha,
    InterventionVector& intervention) const {

    const UInt64 rows = channels.hh.rows();
    const UInt64 cols = channels.hh.cols();

    if (channels.hv.rows() != rows || channels.hv.cols() != cols ||
        channels.vh.rows() != rows || channels.vh.cols() != cols ||
        channels.vv.rows() != rows || channels.vv.cols() != cols) {
        throw InvalidParameterException("PolSAR decomposition requires all four polarization channels to have identical dimensions");
    }

    out_halpha.resize(rows, cols);
    out_halpha.reset_watermark();
    intervention.clear();

    if (progress_) {
        progress_->start_stage("Cloude-Pottier H-alpha decomposition", rows * cols);
    }

    std::atomic<UInt64> alert_counter{0};

    #pragma omp parallel
    {
        std::vector<OilSpillAlert> local_alerts;
        local_alerts.reserve(64);

        #pragma omp for schedule(dynamic) nowait
        for (Int64 r = 0; r < static_cast<Int64>(rows); ++r) {
            const UInt64 ur = static_cast<UInt64>(r);
            auto hh_row = channels.hh.row(ur);
            auto hv_row = channels.hv.row(ur);
            auto vh_row = channels.vh.row(ur);
            auto vv_row = channels.vv.row(ur);

            for (UInt64 c = 0; c < cols; ++c) {
                const Int64 WINDOW = 1;
                std::array<std::array<Complex, 3>, 3> T3_avg{};
                Int64 count = 0;
                for (Int64 dr = -WINDOW; dr <= WINDOW; ++dr) {
                    const Int64 rr = static_cast<Int64>(ur) + dr;
                    if (rr < 0 || rr >= static_cast<Int64>(rows)) continue;
                    auto hh_nr = channels.hh.row(static_cast<UInt64>(rr));
                    auto hv_nr = channels.hv.row(static_cast<UInt64>(rr));
                    auto vh_nr = channels.vh.row(static_cast<UInt64>(rr));
                    auto vv_nr = channels.vv.row(static_cast<UInt64>(rr));
                    for (Int64 dc = -WINDOW; dc <= WINDOW; ++dc) {
                        const Int64 cc = static_cast<Int64>(c) + dc;
                        if (cc < 0 || cc >= static_cast<Int64>(cols)) continue;
                        std::array<std::array<Complex, 3>, 3> T3_loc{};
                        compute_coherency_matrix(
                            hh_nr[static_cast<UInt64>(cc)],
                            hv_nr[static_cast<UInt64>(cc)],
                            vh_nr[static_cast<UInt64>(cc)],
                            vv_nr[static_cast<UInt64>(cc)],
                            T3_loc);
                        for (int ii = 0; ii < 3; ++ii)
                            for (int jj = 0; jj < 3; ++jj)
                                T3_avg[ii][jj] += T3_loc[ii][jj];
                        ++count;
                    }
                }
                if (count > 0) {
                    const Real inv_count = 1.0f / static_cast<Real>(count);
                    for (int ii = 0; ii < 3; ++ii)
                        for (int jj = 0; jj < 3; ++jj)
                            T3_avg[ii][jj] *= inv_count;
                }

                std::array<Real, 3> eigenvalues{};
                std::array<std::array<Complex, 3>, 3> eigenvectors{};
                eigen_decompose_3x3_hermitian(T3_avg, eigenvalues, eigenvectors);

                const Real H = compute_entropy(eigenvalues);
                const Real A = compute_anisotropy(eigenvalues);
                const Real alpha = compute_mean_alpha(eigenvalues, eigenvectors);

                HalphaPixel& pix = out_halpha.at(ur, c);
                pix.entropy_H    = H;
                pix.anisotropy_A = A;
                pix.alpha_mean   = alpha;
                pix.lambda_1     = eigenvalues[0];
                pix.lambda_2     = eigenvalues[1];
                pix.lambda_3     = eigenvalues[2];

                if (is_oil_spill_candidate(H, alpha)) {
                    pix.oil_suspected = true;
                    OilSpillAlert alert;
                    alert.row        = ur;
                    alert.col        = c;
                    alert.entropy_H  = H;
                    alert.alpha_mean = alpha;
                    alert.delta_H    = H - OIL_ENTROPY_THRESHOLD;
                    alert.delta_alpha = alpha - OIL_ALPHA_THRESHOLD;
                    alert.lambda_1   = eigenvalues[0];
                    alert.lambda_2   = eigenvalues[1];
                    alert.lambda_3   = eigenvalues[2];
                    local_alerts.push_back(alert);
                    intervention.add(ur, c, H, alpha);
                }
            }
            out_halpha.ensure_row_watermark(ur);
            if (progress_) progress_->update(cols);

            for (const auto& a : local_alerts) {
                std::cerr << format_alert(a) << std::endl;
            }
            alert_counter.fetch_add(local_alerts.size(), std::memory_order_relaxed);
            local_alerts.clear();
        }
    }

    if (out_halpha.rows() > 0) {
        out_halpha.ensure_row_watermark(out_halpha.rows() - 1);
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

PolSARPipeline::PolSARPipeline(const SARConfig& config) {
    configure(config);
}

void PolSARPipeline::configure(const SARConfig& config) {
    config_ = config;
    decomposer_.configure(config.processing);
    channels_loaded_ = false;
    decomposed_ = false;
    intervention_.clear();
}

void PolSARPipeline::set_progress_reporter(ProgressReporter* reporter) {
    progress_ = reporter;
    decomposer_.set_progress_reporter(reporter);
}

void PolSARPipeline::load_quad_polar_channels(
    const std::string& hh_file,
    const std::string& hv_file,
    const std::string& vh_file,
    const std::string& vv_file,
    UInt64 header_bytes) {

    const UInt64 rows = config_.metadata.azimuth_lines;
    const UInt64 cols = config_.metadata.range_samples;

    struct ChannelLoadSpec {
        const std::string* path;
        ComplexMatrix* dest;
        Polarization pol;
    };

    std::array<ChannelLoadSpec, 4> specs = {{
        {&hh_file, &channels_.hh, Polarization::HH},
        {&hv_file, &channels_.hv, Polarization::HV},
        {&vh_file, &channels_.vh, Polarization::VH},
        {&vv_file, &channels_.vv, Polarization::VV},
    }};

    #pragma omp parallel for schedule(dynamic)
    for (Int64 i = 0; i < 4; ++i) {
        const auto& spec = specs[static_cast<size_t>(i)];
        if (spec.path->empty()) {
            spec.dest->resize(rows, cols);
            spec.dest->fill(Complex(0.0f, 0.0f));
            continue;
        }
        io::MappedFileReader reader;
        reader.open(*spec.path);
        const UInt64 needed_bytes = header_bytes + rows * cols * sizeof(Complex);
        if (reader.size() < needed_bytes) {
            throw IOException("PolSAR channel file too small for specified dimensions: " + *spec.path);
        }
        spec.dest->resize(rows, cols);
        spec.dest->reset_watermark();
        const Complex* src_ptr = reinterpret_cast<const Complex*>(reader.safe_ptr(header_bytes, rows * cols * sizeof(Complex)));
        for (UInt64 r = 0; r < rows; ++r) {
            auto row = spec.dest->row(r);
            for (UInt64 c = 0; c < cols; ++c) {
                row[c] = src_ptr[r * cols + c];
            }
            spec.dest->ensure_row_watermark(r);
        }
        if (spec.dest->rows() > 0) {
            spec.dest->ensure_row_watermark(spec.dest->rows() - 1);
        }
    }

    channels_loaded_ = true;

    if (config_.processing.verbose) {
        std::cout << "[POLSAR] Quad-polarization channels loaded: "
                  << rows << "x" << cols << " (HH+HV+VH+VV)" << std::endl;
    }
}

void PolSARPipeline::run_decomposition() {
    if (!channels_loaded_) {
        throw InvalidParameterException("PolSAR quad-polarization channels not loaded");
    }

    decomposer_.decompose(channels_, halpha_, intervention_);
    decomposed_ = true;

    const UInt64 oil_count = intervention_.size();
    if (config_.processing.verbose) {
        std::cout << "[POLSAR] H-alpha decomposition complete. "
                  << "Oil-suspected pixels: " << oil_count << std::endl;
    }
    if (oil_count > 0) {
        emit_blocking_alerts(std::cerr);
    }
}

void PolSARPipeline::render_oil_overlay(
    std::vector<UInt8>& r_band,
    std::vector<UInt8>& g_band,
    std::vector<UInt8>& b_band,
    UInt64 rows, UInt64 cols) const {

    if (!decomposed_ || !intervention_.has_oil.load(std::memory_order_acquire)) {
        return;
    }

    const UInt64 grid_spacing = 16;

    std::lock_guard<std::mutex> lock(intervention_.mutex);
    for (size_t i = 0; i < intervention_.oil_rows.size(); ++i) {
        const UInt64 orow = intervention_.oil_rows[i];
        const UInt64 ocol = intervention_.oil_cols[i];
        if (orow >= rows || ocol >= cols) continue;

        const UInt64 idx = orow * cols + ocol;
        r_band[idx] = 255;
        g_band[idx] = 0;
        b_band[idx] = 0;

        if (orow % grid_spacing == 0 || ocol % grid_spacing == 0) {
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    const Int64 tr = static_cast<Int64>(orow) + dr;
                    const Int64 tc = static_cast<Int64>(ocol) + dc;
                    if (tr >= 0 && tr < static_cast<Int64>(rows) &&
                        tc >= 0 && tc < static_cast<Int64>(cols)) {
                        const UInt64 tidx = static_cast<UInt64>(tr) * cols + static_cast<UInt64>(tc);
                        r_band[tidx] = 255;
                        g_band[tidx] = 40;
                        b_band[tidx] = 40;
                    }
                }
            }
        }
    }
}

void PolSARPipeline::emit_blocking_alerts(std::ostream& os) const {
    if (!intervention_.has_oil.load(std::memory_order_acquire)) return;

    const UInt64 count = intervention_.size();
    Real max_H = 0.0f, max_alpha = 0.0f;
    Real sum_H = 0.0f, sum_alpha = 0.0f;
    UInt64 max_row = 0, max_col = 0;

    std::lock_guard<std::mutex> lock(intervention_.mutex);
    for (size_t i = 0; i < intervention_.oil_H.size(); ++i) {
        const Real H = intervention_.oil_H[i];
        const Real A = intervention_.oil_alpha[i];
        sum_H += H;
        sum_alpha += A;
        if (H > max_H || (H == max_H && A > max_alpha)) {
            max_H = H;
            max_alpha = A;
            max_row = intervention_.oil_rows[i];
            max_col = intervention_.oil_cols[i];
        }
    }

    os << "\n";
    os << "==============================================================\n";
    os << "  [POLSAR] BLOCKING OIL SPILL INTERVENTION REPORT\n";
    os << "==============================================================\n";
    os << "  Total oil-suspected pixels : " << count << "\n";
    os << "  Mean entropy H             : " << std::fixed << std::setprecision(4)
       << (sum_H / static_cast<Real>(count)) << " (threshold: "
       << CloudePottierDecomposer::OIL_ENTROPY_THRESHOLD << ")\n";
    os << "  Mean alpha angle           : " << std::fixed << std::setprecision(4)
       << (sum_alpha / static_cast<Real>(count)) << " (threshold: "
       << CloudePottierDecomposer::OIL_ALPHA_THRESHOLD << ")\n";
    os << "  Peak pixel                 : (" << max_row << "," << max_col << ")\n";
    os << "  Peak H / alpha             : " << std::fixed << std::setprecision(4)
       << max_H << " / " << max_alpha << "\n";
    os << "  Action                     : Backscatter calculation BLOCKED\n";
    os << "  Overlay rendered           : Bright red oil-spill calibration grid\n";
    os << "==============================================================\n" << std::flush;
}

}
