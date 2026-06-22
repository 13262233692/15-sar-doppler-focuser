#include <sar/fft/fft_engine.hpp>
#include <sar/common/math_utils.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>

static void test_fft_1d_roundtrip() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::fft;

    std::cout << "  Testing 1D FFT round-trip..." << std::endl;

    FFTEngine engine(1);

    const Int64 N = 256;
    std::vector<Complex> orig(static_cast<size_t>(N));
    for (Int64 i = 0; i < N; ++i) {
        const Real t = TWO_PI * static_cast<Real>(i) / static_cast<Real>(N);
        orig[static_cast<size_t>(i)] = Complex(std::cos(t) + 0.5 * std::cos(3 * t),
                                                std::sin(t));
    }

    std::vector<Complex> data = orig;
    engine.fft_1d(data, FFTDirection::FORWARD);
    engine.fft_1d(data, FFTDirection::INVERSE);

    Real max_err = 0.0f;
    for (Int64 i = 0; i < N; ++i) {
        const Real err = std::abs(data[static_cast<size_t>(i)] - orig[static_cast<size_t>(i)]);
        if (err > max_err) max_err = err;
    }

    std::cout << "    Max round-trip error: " << max_err << std::endl;
    assert(max_err < 1e-4);
    std::cout << "    OK" << std::endl;
}

static void test_fft_dirac_delta() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::fft;

    std::cout << "  Testing FFT of Dirac delta..." << std::endl;

    FFTEngine engine(1);
    const Int64 N = 64;
    std::vector<Complex> data(static_cast<size_t>(N), Complex(0, 0));
    data[0] = Complex(1, 0);

    engine.fft_1d(data, FFTDirection::FORWARD);

    for (Int64 i = 0; i < N; ++i) {
        const Real err = std::abs(data[static_cast<size_t>(i)] - Complex(1, 0));
        assert(err < 1e-5);
    }
    std::cout << "    OK" << std::endl;
}

static void test_fft_matrix_rows() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::fft;

    std::cout << "  Testing FFT matrix rows..." << std::endl;

    FFTEngine engine(2);
    const Int64 rows = 8;
    const Int64 cols = 64;
    ComplexMatrix mat(static_cast<UInt64>(rows), static_cast<UInt64>(cols));

    for (Int64 r = 0; r < rows; ++r) {
        for (Int64 c = 0; c < cols; ++c) {
            const Real t = TWO_PI * static_cast<Real>(c) * static_cast<Real>(r + 1) / static_cast<Real>(cols);
            mat.at(static_cast<UInt64>(r), static_cast<UInt64>(c)) = Complex(std::cos(t), std::sin(t));
        }
    }

    ComplexMatrix original = mat;
    engine.fft_rows(mat, FFTDirection::FORWARD);
    engine.fft_rows(mat, FFTDirection::INVERSE);

    Real max_err = 0.0f;
    for (Int64 r = 0; r < rows; ++r) {
        for (Int64 c = 0; c < cols; ++c) {
            const Real err = std::abs(
                mat.at(static_cast<UInt64>(r), static_cast<UInt64>(c)) -
                original.at(static_cast<UInt64>(r), static_cast<UInt64>(c))
            );
            if (err > max_err) max_err = err;
        }
    }
    std::cout << "    Max matrix rows round-trip error: " << max_err << std::endl;
    assert(max_err < 1e-4);
    std::cout << "    OK" << std::endl;
}

int main() {
    std::cout << "=== Testing FFT Engine ===" << std::endl;

    test_fft_1d_roundtrip();
    test_fft_dirac_delta();
    test_fft_matrix_rows();

    std::cout << "All FFT tests passed!" << std::endl;
    return 0;
}
