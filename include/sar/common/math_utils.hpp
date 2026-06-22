#pragma once

#include <cmath>
#include <numbers>
#include <complex>
#include <sar/common/types.hpp>

namespace sar::math {

inline constexpr Real PI        = std::numbers::pi_v<Real>;
inline constexpr Real TWO_PI    = 2.0 * PI;
inline constexpr Real HALF_PI   = PI / 2.0;
inline constexpr Real INV_PI    = 1.0 / PI;
inline constexpr Real INV_TWO_PI = 1.0 / TWO_PI;
inline constexpr Real SPEED_OF_LIGHT = 299792458.0;

inline Complex expj(Real phase) noexcept {
    return Complex(std::cos(phase), std::sin(phase));
}

inline Real sinc(Real x) noexcept {
    if (std::abs(x) < 1e-10) {
        return 1.0;
    }
    const Real px = PI * x;
    return std::sin(px) / px;
}

inline Real hann_window(Int32 n, Int32 N) noexcept {
    return 0.5 - 0.5 * std::cos(TWO_PI * static_cast<Real>(n) / static_cast<Real>(N - 1));
}

inline Real hamming_window(Int32 n, Int32 N) noexcept {
    return 0.54 - 0.46 * std::cos(TWO_PI * static_cast<Real>(n) / static_cast<Real>(N - 1));
}

inline Real blackman_window(Int32 n, Int32 N) noexcept {
    const Real a0 = 0.42;
    const Real a1 = 0.50;
    const Real a2 = 0.08;
    const Real t  = TWO_PI * static_cast<Real>(n) / static_cast<Real>(N - 1);
    return a0 - a1 * std::cos(t) + a2 * std::cos(2.0 * t);
}

template<typename T>
inline T clamp(T v, T lo, T hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline Int64 next_pow2(Int64 n) noexcept {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

inline Real db2amp(Real db) noexcept {
    return std::pow(Real(10.0), db / Real(20.0));
}

inline Real amp2db(Real amp) noexcept {
    return Real(20.0) * std::log10(amp + Real(1e-12));
}

inline Real pow2db(Real pwr) noexcept {
    return Real(10.0) * std::log10(pwr + Real(1e-12));
}

}
