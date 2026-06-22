#include <sar/common/types.hpp>
#include <sar/common/config.hpp>
#include <sar/polsar/polsar_decomposer.hpp>
#include <sar/sar_focuser.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include <complex>
#include <vector>

using namespace sar;
using namespace sar::polsar;

static void test_coherency_matrix() {
    std::cout << "  Testing coherency matrix construction..." << std::endl;

    const Complex S_hh(1.0f, 0.0f);
    const Complex S_hv(0.2f, 0.1f);
    const Complex S_vh(0.2f, 0.1f);
    const Complex S_vv(0.8f, 0.0f);

    std::array<std::array<Complex, 3>, 3> T3{};
    CloudePottierDecomposer::compute_coherency_matrix(S_hh, S_hv, S_vh, S_vv, T3);

    const Complex k1 = S_hh + S_vv;
    const Complex T11_expected = k1 * std::conj(k1);
    const Real T11_diff = std::abs(T3[0][0] - T11_expected);
    assert(T11_diff < 1e-5f);

    const Real trace = T3[0][0].real() + T3[1][1].real() + T3[2][2].real();
    assert(trace > 0.0f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Real diff = std::abs(T3[i][j] - std::conj(T3[j][i]));
            assert(diff < 1e-5f);
        }
    }

    std::cout << "    Coherency matrix OK" << std::endl;
}

static void test_eigen_decomposition() {
    std::cout << "  Testing 3x3 Hermitian eigen decomposition..." << std::endl;

    std::array<std::array<Complex, 3>, 3> T3{};
    T3[0][0] = Complex(3.0f, 0.0f);
    T3[1][1] = Complex(2.0f, 0.0f);
    T3[2][2] = Complex(1.0f, 0.0f);
    T3[0][1] = Complex(0.0f, 0.0f);
    T3[0][2] = Complex(0.0f, 0.0f);
    T3[1][2] = Complex(0.0f, 0.0f);

    std::array<Real, 3> eigenvalues{};
    std::array<std::array<Complex, 3>, 3> eigenvectors{};
    const bool ok = CloudePottierDecomposer::eigen_decompose_3x3_hermitian(T3, eigenvalues, eigenvectors);
    assert(ok);

    assert(eigenvalues[0] >= eigenvalues[1] - 1e-3f);
    assert(eigenvalues[1] >= eigenvalues[2] - 1e-3f);

    const Real trace_in = 3.0f + 2.0f + 1.0f;
    const Real trace_out = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    assert(std::abs(trace_in - trace_out) < 1e-2f);

    std::cout << "    Eigen decomposition OK (trace=" << trace_out << ")" << std::endl;
}

static void test_entropy_alpha() {
    std::cout << "  Testing entropy and mean alpha computation..." << std::endl;

    std::array<Real, 3> equal_eigs = {1.0f, 1.0f, 1.0f};
    const Real H_equal = CloudePottierDecomposer::compute_entropy(equal_eigs);
    assert(std::abs(H_equal - 1.0f) < 1e-3f);

    std::array<Real, 3> dominant_eigs = {100.0f, 0.001f, 0.001f};
    const Real H_dom = CloudePottierDecomposer::compute_entropy(dominant_eigs);
    assert(H_dom < 0.2f);

    std::cout << "    Entropy OK (equal=" << H_equal << ", dominant=" << H_dom << ")" << std::endl;
}

static void test_oil_spill_detection() {
    std::cout << "  Testing oil spill threshold detection..." << std::endl;

    assert(CloudePottierDecomposer::is_oil_spill_candidate(0.90f, 0.85f));
    assert(!CloudePottierDecomposer::is_oil_spill_candidate(0.50f, 0.50f));
    assert(!CloudePottierDecomposer::is_oil_spill_candidate(0.90f, 0.50f));
    assert(!CloudePottierDecomposer::is_oil_spill_candidate(0.50f, 0.90f));

    std::cout << "    Oil spill threshold OK" << std::endl;
}

static void test_full_polsar_pipeline() {
    std::cout << "  Testing full PolSAR pipeline with synthetic oil-spill scene..." << std::endl;

    const UInt64 rows = 64;
    const UInt64 cols = 64;

    PolSARQuadChannel channels;
    channels.hh.resize(rows, cols);
    channels.hv.resize(rows, cols);
    channels.vh.resize(rows, cols);
    channels.vv.resize(rows, cols);
    channels.hh.reset_watermark();
    channels.hv.reset_watermark();
    channels.vh.reset_watermark();
    channels.vv.reset_watermark();

    unsigned int seed = 12345u;
    auto rand_real = [&]() -> Real {
        seed = seed * 1103515245u + 12345u;
        return static_cast<Real>(seed) / static_cast<Real>(UINT32_MAX);
    };

    for (UInt64 r = 0; r < rows; ++r) {
        for (UInt64 c = 0; c < cols; ++c) {
            const Real dist = std::sqrt(
                std::pow(static_cast<Real>(r) - static_cast<Real>(rows)/2.0f, 2) +
                std::pow(static_cast<Real>(c) - static_cast<Real>(cols)/2.0f, 2)
            );
            const bool in_oil = dist < 15.0f;

            if (in_oil) {
                const Real amp = 0.3f;
                const Real ph1 = rand_real() * 6.2831853f;
                const Real ph2 = rand_real() * 6.2831853f;
                const Real ph3 = rand_real() * 6.2831853f;
                const Real ph4 = rand_real() * 6.2831853f;
                channels.hh.at(r, c) = Complex(amp * std::cos(ph1), amp * std::sin(ph1));
                channels.hv.at(r, c) = Complex(amp * std::cos(ph2), amp * std::sin(ph2));
                channels.vh.at(r, c) = Complex(amp * std::cos(ph3), amp * std::sin(ph3));
                channels.vv.at(r, c) = Complex(amp * std::cos(ph4), amp * std::sin(ph4));
            } else {
                channels.hh.at(r, c) = Complex(1.0f, 0.0f);
                channels.hv.at(r, c) = Complex(0.0f, 0.0f);
                channels.vh.at(r, c) = Complex(0.0f, 0.0f);
                channels.vv.at(r, c) = Complex(0.8f, 0.0f);
            }
        }
        channels.hh.ensure_row_watermark(r);
        channels.hv.ensure_row_watermark(r);
        channels.vh.ensure_row_watermark(r);
        channels.vv.ensure_row_watermark(r);
    }
    channels.hh.ensure_row_watermark(rows - 1);
    channels.hv.ensure_row_watermark(rows - 1);
    channels.vh.ensure_row_watermark(rows - 1);
    channels.vv.ensure_row_watermark(rows - 1);

    CloudePottierDecomposer decomposer;
    HalphaMatrix halpha;
    InterventionVector intervention;
    decomposer.decompose(channels, halpha, intervention);

    std::cout << "    Oil-suspected pixels detected: " << intervention.size() << std::endl;

    UInt64 oil_count_center = 0;
    UInt64 non_oil_edge = 0;
    for (UInt64 r = 0; r < rows; ++r) {
        for (UInt64 c = 0; c < cols; ++c) {
            const Real dist = std::sqrt(
                std::pow(static_cast<Real>(r) - static_cast<Real>(rows)/2.0f, 2) +
                std::pow(static_cast<Real>(c) - static_cast<Real>(cols)/2.0f, 2)
            );
            const bool in_oil = dist < 15.0f;
            const Real H = halpha.at(r, c).entropy_H;
            if (in_oil && H > 0.75f) oil_count_center++;
            if (!in_oil && H < 0.3f) non_oil_edge++;
        }
    }
    assert(non_oil_edge > 100);
    std::cout << "    High-H oil-region pixels: " << oil_count_center
              << ", low-H non-oil pixels: " << non_oil_edge << std::endl;

    if (intervention.size() > 0) {
        std::cout << "    Intervention triggered with " << intervention.size()
                  << " pixels (H >= 0.8 && alpha >= 0.8)" << std::endl;
    }

    std::cout << "    Full PolSAR pipeline OK" << std::endl;
}

static void test_intervention_overlay() {
    std::cout << "  Testing intervention vector overlay rendering..." << std::endl;

    const UInt64 rows = 8;
    const UInt64 cols = 8;
    const UInt64 n = rows * cols;
    std::vector<UInt8> r(n, 128);
    std::vector<UInt8> g(n, 128);
    std::vector<UInt8> b(n, 128);

    SARConfig cfg;
    cfg.metadata.azimuth_lines = rows;
    cfg.metadata.range_samples = cols;
    PolSARPipeline pipeline(cfg);

    pipeline.intervention().add(4, 4, 0.9f, 0.85f);
    assert(pipeline.has_oil_spill());
    assert(pipeline.oil_pixel_count() == 1);

    pipeline.render_oil_overlay(r, g, b, rows, cols);

    const UInt64 idx = 4 * cols + 4;
    assert(r[idx] == 255);

    std::cout << "    Intervention overlay OK" << std::endl;
}

int main() {
    std::cout << "=== Testing PolSAR H-alpha Decomposition ===" << std::endl;

    try {
        test_coherency_matrix();
        test_eigen_decomposition();
        test_entropy_alpha();
        test_oil_spill_detection();
        test_full_polsar_pipeline();
        test_intervention_overlay();
    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "All PolSAR tests passed!" << std::endl;
    return 0;
}
