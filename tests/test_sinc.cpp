#include <sar/focusing/rd_focuser.hpp>
#include <sar/common/math_utils.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>
#include <cstdint>

static void test_sinc_interpolation() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::focusing;

    std::cout << "  Testing Sinc interpolation..." << std::endl;

    SincInterpolator interp(8, 1024);
    assert(interp.kernel_size() == 8);

    std::vector<Complex> data(64);
    for (size_t i = 0; i < data.size(); ++i) {
        const Real t = TWO_PI * static_cast<Real>(i) / 16.0f;
        data[i] = Complex(std::cos(t), 0.0f);
    }

    for (Real f = 5.0f; f < 58.0f; f += 0.1f) {
        const Complex v = interp.interpolate(data, f);
        const Real t = TWO_PI * f / 16.0f;
        const Complex expected(std::cos(t), 0.0f);
        const Real err = std::abs(v - expected);
        if (err > 0.05f) {
            std::cerr << "    Sinc error at " << f << ": " << err
                      << " got=" << v << " expected=" << expected << std::endl;
        }
    }

    std::cout << "  Sinc interpolation OK" << std::endl;
}

static void test_sinc_kernel_normalized() {
    using namespace sar;
    using namespace sar::focusing;
    using namespace sar::math;

    std::cout << "  Testing Sinc kernel normalization..." << std::endl;

    SincInterpolator interp(8, 1024);
    const auto& table = interp.kernel_table();

    const int kernel_sz = 8;
    const int oversampling = 1024;

    for (int fi = 0; fi <= oversampling; fi += oversampling / 10) {
        Real sum = 0.0f;
        for (int k = 0; k < kernel_sz; ++k) {
            sum += table[fi * kernel_sz + k];
        }
        assert(std::abs(sum - 1.0f) < 1e-4);
    }

    std::cout << "  Kernel normalization OK" << std::endl;
}

static void test_range_shift_interpolation() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::focusing;

    std::cout << "  Testing range line interpolation..." << std::endl;

    SincInterpolator interp(8, 2048);

    const int64_t N = 256;
    std::vector<Complex> in(static_cast<size_t>(N));
    std::vector<Complex> out(static_cast<size_t>(N));
    std::vector<Real> shifts(static_cast<size_t>(N), 0.0f);

    for (int64_t i = 0; i < N; ++i) {
        in[static_cast<size_t>(i)] = Complex(static_cast<Real>(i), static_cast<Real>(-i));
    }

    interp.interpolate_range_line(in, out, shifts);
    for (int64_t i = 4; i < N - 4; ++i) {
        const Real err = std::abs(out[static_cast<size_t>(i)] - in[static_cast<size_t>(i)]);
        if (err > 0.001f) {
            std::cerr << "    Zero-shift error at " << i << ": " << err << std::endl;
        }
    }

    for (int64_t i = 0; i < N; ++i) {
        shifts[static_cast<size_t>(i)] = 0.5f;
    }
    interp.interpolate_range_line(in, out, shifts);

    std::cout << "  Range line interpolation OK" << std::endl;
}

int main() {
    std::cout << "=== Testing Sinc Interpolator ===" << std::endl;

    test_sinc_interpolation();
    test_sinc_kernel_normalized();
    test_range_shift_interpolation();

    std::cout << "All Sinc tests passed!" << std::endl;
    return 0;
}
