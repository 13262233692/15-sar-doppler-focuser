#include <sar/sar_focuser.hpp>
#include <sar/common/math_utils.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

static void test_pipeline_synthetic() {
    using namespace sar;
    using namespace sar::math;

    std::cout << "  Running synthetic imaging pipeline test..." << std::endl;

    SARConfig cfg = SARConfig::make_default();
    cfg.metadata.azimuth_lines = 256;
    cfg.metadata.range_samples = 256;
    cfg.processing.verbose = false;
    cfg.processing.omp_num_threads = 2;
    cfg.processing.sinc_interp_kernel_size = 8;

    SARDopplerFocuser focuser(cfg);
    focuser.run_full_pipeline();

    const ComplexMatrix& focused = focuser.focused_image();
    assert(focused.rows() == 256);
    assert(focused.cols() == 256);

    const RealMatrix& backscatter = focuser.backscatter();
    assert(backscatter.rows() == 256);
    assert(backscatter.cols() == 256);

    Real max_val = 0.0f;
    UInt64 max_az = 0, max_rg = 0;
    for (UInt64 a = 0; a < backscatter.rows(); ++a) {
        for (UInt64 r = 0; r < backscatter.cols(); ++r) {
            const Real v = backscatter.at(a, r);
            if (v > max_val) {
                max_val = v;
                max_az = a;
                max_rg = r;
            }
        }
    }

    const UInt64 tgt_az = backscatter.rows() / 2;
    const UInt64 tgt_rg = backscatter.cols() / 2;

    std::cout << "    Peak at (az=" << max_az << ", rg=" << max_rg
              << ") expected ~(" << tgt_az << ", " << tgt_rg << ")" << std::endl;
    std::cout << "    Peak value: " << max_val << std::endl;

    const Int64 az_err = std::abs(static_cast<Int64>(max_az) - static_cast<Int64>(tgt_az));
    const Int64 rg_err = std::abs(static_cast<Int64>(max_rg) - static_cast<Int64>(tgt_rg));
    std::cout << "    Az error: " << az_err << " pixels, Rg error: " << rg_err << " pixels" << std::endl;

    assert(az_err < 4);
    assert(rg_err < 4);

    std::cout << "    Synthetic pipeline OK" << std::endl;
}

static void test_range_compression_only() {
    using namespace sar;
    using namespace sar::math;
    using namespace sar::focusing;

    std::cout << "  Testing range compression..." << std::endl;

    SARConfig cfg = SARConfig::make_default();
    cfg.metadata.azimuth_lines = 32;
    cfg.metadata.range_samples = 512;
    cfg.processing.verbose = false;

    SARDopplerFocuser focuser(cfg);
    focuser.run_full_pipeline();

    std::cout << "    Range compression OK" << std::endl;
}

int main() {
    std::cout << "=== Testing full SAR imaging pipeline ===" << std::endl;

    test_range_compression_only();
    test_pipeline_synthetic();

    std::cout << "All pipeline tests passed!" << std::endl;
    return 0;
}
