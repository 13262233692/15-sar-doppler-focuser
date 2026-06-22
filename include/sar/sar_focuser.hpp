#pragma once

#include <sar/common/types.hpp>
#include <sar/common/config.hpp>
#include <sar/common/progress.hpp>
#include <sar/io/raw_echo_unpacker.hpp>
#include <sar/fft/fft_engine.hpp>
#include <sar/focusing/rd_focuser.hpp>
#include <sar/output/output_writers.hpp>
#include <memory>
#include <string>

namespace sar {

class SARDopplerFocuser {
public:
    SARDopplerFocuser();
    explicit SARDopplerFocuser(const SARConfig& config);
    ~SARDopplerFocuser() = default;

    void configure(const SARConfig& config);
    const SARConfig& config() const noexcept { return config_; }

    void set_progress_reporter(ProgressReporter::Callback cb);
    ProgressReporter& progress() { return progress_; }

    void initialize();

    ComplexMatrix& raw_echo() { return raw_echo_; }
    const ComplexMatrix& raw_echo() const { return raw_echo_; }

    ComplexMatrix& focused_image() { return focused_; }
    const ComplexMatrix& focused_image() const { return focused_; }

    RealMatrix& backscatter() { return backscatter_; }
    const RealMatrix& backscatter() const { return backscatter_; }

    void load_raw_data(const std::string& filepath, UInt64 header_bytes = 0);

    void run_full_pipeline();

    void run_range_compression();

    void run_rcmc();

    void run_doppler_estimation();

    void run_azimuth_compression();

    void run_normalization();

    void export_geotiff(const std::string& filepath) const;

    void export_binary(const std::string& filepath) const;

    fft::FFTEngine& fft_engine() { return fft_engine_; }
    const fft::FFTEngine& fft_engine() const { return fft_engine_; }

    focusing::RangeCompressor& range_compressor() { return range_comp_; }
    focusing::SincInterpolator& sinc_interpolator() { return sinc_; }
    focusing::RangeCellMigrationCorrector& rcmc() { return rcmc_; }
    focusing::DopplerEstimator& doppler_estimator() { return doppler_est_; }
    focusing::AzimuthCompressor& azimuth_compressor() { return azimuth_comp_; }
    output::BackscatterNormalizer& normalizer() { return normalizer_; }

private:
    SARConfig      config_;
    ProgressReporter progress_;
    bool           initialized_ = false;

    io::RawEchoUnpacker unpacker_;

    fft::FFTEngine fft_engine_;
    fft::FFTWorkspace workspace_;

    focusing::RangeCompressor  range_comp_;
    focusing::SincInterpolator sinc_;
    focusing::RangeCellMigrationCorrector rcmc_;
    focusing::DopplerEstimator doppler_est_;
    focusing::AzimuthCompressor azimuth_comp_;

    output::BackscatterNormalizer normalizer_;
    output::GeoTIFFWriter geotiff_writer_;
    output::BinaryWriter  binary_writer_;

    ComplexMatrix raw_echo_;
    ComplexMatrix focused_;
    RealMatrix    backscatter_;

    bool echo_loaded_    = false;
    bool range_compressed_ = false;
    bool rcmc_done_      = false;
    bool azimuth_done_   = false;
    bool normalized_     = false;
};

}
