#include <sar/common/types.hpp>
#include <sar/common/math_utils.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    using namespace sar;
    using namespace sar::math;

    std::cout << "=== Testing types and math utilities ===" << std::endl;

    Tensor2D<Complex> mat(4, 8);
    assert(mat.rows() == 4);
    assert(mat.cols() == 8);
    assert(mat.size() >= 32);
    assert(mat.stride() >= 8);
    mat.at(2, 3) = Complex(1.5f, 2.5f);
    assert(std::abs(mat.at(2, 3).real() - 1.5f) < 1e-6);
    assert(std::abs(mat.at(2, 3).imag() - 2.5f) < 1e-6);

    auto row = mat.row(2);
    assert(row.size() == 8);
    assert(std::abs(row[3].real() - 1.5f) < 1e-6);

    assert(std::abs(sinc(0.0f) - 1.0f) < 1e-6);
    assert(std::abs(sinc(1.0f)) < 1e-6);
    assert(std::abs(sinc(-1.0f)) < 1e-6);

    const Complex c = expj(0.0f);
    assert(std::abs(c.real() - 1.0f) < 1e-6);
    assert(std::abs(c.imag()) < 1e-6);

    const Complex c2 = expj(PI);
    assert(std::abs(c2.real() + 1.0f) < 1e-5);
    assert(std::abs(c2.imag()) < 1e-5);

    assert(next_pow2(1) == 1);
    assert(next_pow2(2) == 2);
    assert(next_pow2(3) == 4);
    assert(next_pow2(1023) == 1024);
    assert(next_pow2(1024) == 1024);
    assert(next_pow2(1025) == 2048);

    assert(clamp(5, 0, 10) == 5);
    assert(clamp(-1, 0, 10) == 0);
    assert(clamp(11, 0, 10) == 10);

    std::cout << "All types/math tests passed!" << std::endl;
    return 0;
}
