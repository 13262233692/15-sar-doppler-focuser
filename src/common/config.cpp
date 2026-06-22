#include <sar/common/config.hpp>
#include <sar/common/math_utils.hpp>

namespace sar {

SARConfig SARConfig::make_default() {
    SARConfig cfg;
    cfg.metadata.azimuth_lines     = 4096;
    cfg.metadata.range_samples     = 4096;
    cfg.metadata.carrier_freq_hz   = 5.4e9f;
    cfg.metadata.bandwidth_hz      = 100e6f;
    cfg.metadata.pulse_duration_s  = 10e-6f;
    cfg.metadata.prf_hz            = 2000.0f;
    cfg.metadata.range_sampling_hz = 120e6f;
    cfg.metadata.platform_velocity = 7500.0f;
    cfg.metadata.wavelength_m      = math::SPEED_OF_LIGHT / cfg.metadata.carrier_freq_hz;
    cfg.metadata.near_range_m      = 800000.0f;
    cfg.metadata.range_spacing_m   = math::SPEED_OF_LIGHT / (2.0f * cfg.metadata.range_sampling_hz);
    cfg.metadata.azimuth_spacing_m = cfg.metadata.platform_velocity / cfg.metadata.prf_hz;
    cfg.metadata.doppler_centroid  = 0.0f;
    cfg.metadata.sample_format     = SampleFormat::COMPLEX_INT16;
    cfg.metadata.bits_per_sample   = 16;
    cfg.processing.omp_num_threads = 0;
    cfg.processing.verbose         = true;
    return cfg;
}

SARConfig SARConfig::load_from_file(const std::string&) {
    return make_default();
}

void SARConfig::validate() const {
}

}
