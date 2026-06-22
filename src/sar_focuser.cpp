#include <sar/sar_focuser.hpp>
#include <sar/common/math_utils.hpp>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace sar {

SARDopplerFocuser::SARDopplerFocuser() {
    config_ = SARConfig::make_default();
}

SARDopplerFocuser::SARDopplerFocuser(const SARConfig& config)
    : config_(config) {
    initialize();
}

void SARDopplerFocuser::configure(const SARConfig& config) {
    config_ = config;
    initialized_ = false;
    initialize();
}

void SARDopplerFocuser::set_progress_reporter(ProgressReporter::Callback cb) {
    progress_.set_callback(std::move(cb));
}

void SARDopplerFocuser::initialize() {
    if (config_.processing.omp_num_threads > 0) {
        fft_engine_.set_num_threads(config_.processing.omp_num_threads);
        omp_set_num_threads(static_cast<int>(config_.processing.omp_num_threads));
    }

    sinc_ = focusing::SincInterpolator(
        config_.processing.sinc_interp_kernel_size, 1024
    );

    range_comp_.configure(
        config_.metadata,
        &fft_engine_,
        config_.processing.apply_range_window,
        config_.processing.range_window_beta
    );

    rcmc_.configure(
        config_.metadata,
        &sinc_,
        config_.processing.sinc_interp_kernel_size
    );

    doppler_est_.configure(config_.metadata);

    azimuth_comp_.configure(
        config_.metadata,
        &fft_engine_,
        config_.processing.apply_azimuth_window,
        config_.processing.azimuth_window_beta
    );

    normalizer_.configure(config_.processing);

    range_comp_.set_progress_reporter(&progress_);
    rcmc_.set_progress_reporter(&progress_);
    doppler_est_.set_progress_reporter(&progress_);
    azimuth_comp_.set_progress_reporter(&progress_);
    normalizer_.set_progress_reporter(&progress_);
    geotiff_writer_.set_progress_reporter(&progress_);
    binary_writer_.set_progress_reporter(&progress_);
    polsar_pipeline_.set_progress_reporter(&progress_);

    polsar_pipeline_.configure(config_);

    const UInt64 rows = config_.metadata.azimuth_lines;
    const UInt64 cols = config_.metadata.range_samples;
    workspace_.reserve(rows, cols);

    initialized_ = true;
    echo_loaded_ = false;
    range_compressed_ = false;
    rcmc_done_ = false;
    azimuth_done_ = false;
    normalized_ = false;
}

void SARDopplerFocuser::load_raw_data(const std::string& filepath, UInt64 header_bytes) {
    if (!initialized_) initialize();

    unpacker_.open(config_.metadata, filepath, header_bytes);
    unpacker_.set_progress_reporter(&progress_);
    unpacker_.unpack_all(raw_echo_);
    echo_loaded_ = true;

    if (config_.processing.verbose) {
        std::cout << "[SAR] Raw data loaded: "
                  << raw_echo_.rows() << " azimuth x "
                  << raw_echo_.cols() << " range samples" << std::endl;
    }
}

void SARDopplerFocuser::run_range_compression() {
    if (!echo_loaded_) {
        throw InvalidParameterException("No raw echo data loaded");
    }

    focused_ = raw_echo_;
    range_comp_.compress(focused_);
    range_compressed_ = true;
}

void SARDopplerFocuser::run_rcmc() {
    if (!range_compressed_) {
        run_range_compression();
    }

    rcmc_.correct(focused_);
    rcmc_done_ = true;
}

void SARDopplerFocuser::run_doppler_estimation() {
    if (!range_compressed_) {
        run_range_compression();
    }

    if (config_.processing.do_doppler_estimation) {
        Real fd = doppler_est_.estimate_centroid(focused_, &fft_engine_);
        config_.metadata.doppler_centroid = fd;
        azimuth_comp_.set_doppler_centroid(fd);

        if (config_.processing.verbose) {
            std::cout << "[SAR] Estimated Doppler centroid: "
                      << std::fixed << std::setprecision(2)
                      << fd << " Hz" << std::endl;
        }
    } else {
        azimuth_comp_.set_doppler_centroid(config_.processing.fixed_doppler_centroid);
    }
}

void SARDopplerFocuser::run_azimuth_compression() {
    if (!rcmc_done_) {
        run_rcmc();
    }

    run_doppler_estimation();

    azimuth_comp_.compress(focused_);
    azimuth_done_ = true;
}

void SARDopplerFocuser::run_normalization() {
    if (!azimuth_done_) {
        run_azimuth_compression();
    }

    normalizer_.normalize_to_db(focused_, backscatter_, true);
    normalized_ = true;
}

void SARDopplerFocuser::run_full_pipeline() {
    using namespace std::chrono;
    const auto t_start = high_resolution_clock::now();

    if (!initialized_) initialize();

    if (!echo_loaded_ && !config_.input_file.empty()) {
        load_raw_data(config_.input_file);
    }

    if (!echo_loaded_) {
        std::cerr << "[SAR] WARNING: No raw echo data loaded, generating synthetic test data" << std::endl;
        const UInt64 Naz = config_.metadata.azimuth_lines;
        const UInt64 Nr  = config_.metadata.range_samples;
        raw_echo_.resize(Naz, Nr);
        raw_echo_.fill(Complex(0, 0));
        echo_loaded_ = true;

        const Real v = config_.metadata.platform_velocity;
        const Real lambda = config_.metadata.wavelength_m;
        const Real R0 = config_.metadata.near_range_m;
        const Real dr = config_.metadata.range_spacing_m;
        const Real prf = config_.metadata.prf_hz;
        const Real T_pulse = config_.metadata.pulse_duration_s;
        const Real B = config_.metadata.bandwidth_hz;
        const Real fs_r = config_.metadata.range_sampling_hz;
        const Real K = B / T_pulse;

        const UInt64 tgt_az = Naz / 2;
        const UInt64 tgt_r  = Nr / 2;
        const Real tgt_R = R0 + static_cast<Real>(tgt_r) * dr;

        for (UInt64 a = 0; a < Naz; ++a) {
            const Real ta = static_cast<Real>(static_cast<Int64>(a) - static_cast<Int64>(tgt_az)) / prf;
            const Real R_a = tgt_R + (v * v * ta * ta) / (2.0 * tgt_R);
            const Real tau = 2.0 * R_a / math::SPEED_OF_LIGHT;

            for (UInt64 r = 0; r < Nr; ++r) {
                const Real tr = static_cast<Real>(r) / fs_r;
                const Real dt = tr - tau;
                if (std::abs(dt) < T_pulse / 2.0) {
                    const Real phase_range = math::PI * K * dt * dt;
                    const Real phase_az = -2.0 * math::TWO_PI * R_a / lambda;
                    const Real W_range = 0.5f - 0.5f * std::cos(
                        math::TWO_PI * (dt + T_pulse / 2.0f) / T_pulse);
                    raw_echo_.at(a, r) = Complex(
                        std::cos(phase_range + phase_az),
                        std::sin(phase_range + phase_az)
                    ) * W_range;
                }
            }
        }
    }

    if (config_.processing.verbose) {
        std::cout << "[SAR] === Starting Range-Doppler Imaging Pipeline ===" << std::endl;
    }

    run_range_compression();
    if (config_.processing.verbose) std::cout << "[SAR] Range compression complete" << std::endl;

    run_rcmc();
    if (config_.processing.verbose) std::cout << "[SAR] RCMC complete" << std::endl;

    run_doppler_estimation();

    run_azimuth_compression();
    if (config_.processing.verbose) std::cout << "[SAR] Azimuth compression complete" << std::endl;

    if (polsar_enabled_) {
        if (config_.processing.verbose) {
            std::cout << "[SAR] Running Cloude-Pottier H-alpha PolSAR decomposition..." << std::endl;
        }
        polsar_pipeline_.run_decomposition();
        const UInt64 oil_count = polsar_pipeline_.oil_pixel_count();
        if (oil_count > 0) {
            if (config_.processing.verbose) {
                std::cout << "[SAR] POLSAR INTERVENTION: Detected " << oil_count
                          << " oil-suspected pixels. Blocking backscatter for these regions." << std::endl;
            }
            const auto& intervention = polsar_pipeline_.intervention();
            #pragma omp parallel for schedule(dynamic)
            for (Int64 i = 0; i < static_cast<Int64>(intervention.oil_rows.size()); ++i) {
                const UInt64 r = intervention.oil_rows[static_cast<size_t>(i)];
                const UInt64 c = intervention.oil_cols[static_cast<size_t>(i)];
                if (r < focused_.rows() && c < focused_.cols()) {
                    focused_.at(r, c) = Complex(0.0f, 0.0f);
                }
            }
        }
    }

    run_normalization();
    if (config_.processing.verbose) std::cout << "[SAR] Normalization complete" << std::endl;

    if (!config_.output_geotiff.empty()) {
        export_geotiff(config_.output_geotiff);
        if (config_.processing.verbose) {
            std::cout << "[SAR] GeoTIFF exported to: " << config_.output_geotiff << std::endl;
        }
    }
    if (!config_.output_binary.empty()) {
        export_binary(config_.output_binary);
        if (config_.processing.verbose) {
            std::cout << "[SAR] Binary exported to: " << config_.output_binary << std::endl;
        }
    }

    const auto t_end = high_resolution_clock::now();
    const Real elapsed = duration<Real>(t_end - t_start).count();
    if (config_.processing.verbose) {
        std::cout << "[SAR] Total processing time: "
                  << std::fixed << std::setprecision(2)
                  << elapsed << " s" << std::endl;
    }
}

void SARDopplerFocuser::export_geotiff(const std::string& filepath) const {
    if (!normalized_) {
        throw InvalidParameterException("Image not yet normalized");
    }
    if (output::GeoTIFFWriter::is_available() && polsar_enabled_ && polsar_pipeline_.has_oil_spill()) {
        const UInt64 rows = focused_.rows();
        const UInt64 cols = focused_.cols();
        std::vector<UInt8> r(rows * cols);
        std::vector<UInt8> g(rows * cols);
        std::vector<UInt8> b(rows * cols);
        output::BackscatterNormalizer norm(config_.processing);
        std::vector<UInt8> gray;
        norm.normalize_to_uint8(focused_, gray, true);
        #pragma omp parallel for schedule(dynamic)
        for (Int64 i = 0; i < static_cast<Int64>(rows * cols); ++i) {
            const UInt8 v = gray[static_cast<size_t>(i)];
            r[static_cast<size_t>(i)] = v;
            g[static_cast<size_t>(i)] = v;
            b[static_cast<size_t>(i)] = v;
        }
        polsar_pipeline_.render_oil_overlay(r, g, b, rows, cols);
        geotiff_writer_.write_rgb(r, g, b, rows, cols, filepath, config_.georef);
    } else if (output::GeoTIFFWriter::is_available()) {
        geotiff_writer_.write_float32(backscatter_, filepath, config_.georef);
    } else {
        std::vector<UInt8> u8;
        output::BackscatterNormalizer norm(config_.processing);
        norm.normalize_to_uint8(focused_, u8, true);
        if (polsar_enabled_ && polsar_pipeline_.has_oil_spill()) {
            const UInt64 rows = focused_.rows();
            const UInt64 cols = focused_.cols();
            std::vector<UInt8> r(rows * cols);
            std::vector<UInt8> g(rows * cols);
            std::vector<UInt8> b(rows * cols);
            #pragma omp parallel for schedule(dynamic)
            for (Int64 i = 0; i < static_cast<Int64>(rows * cols); ++i) {
                const UInt8 v = u8[static_cast<size_t>(i)];
                r[static_cast<size_t>(i)] = v;
                g[static_cast<size_t>(i)] = v;
                b[static_cast<size_t>(i)] = v;
            }
            polsar_pipeline_.render_oil_overlay(r, g, b, rows, cols);
            geotiff_writer_.write_rgb(r, g, b, rows, cols, filepath, config_.georef);
        } else {
            geotiff_writer_.write_uint8(u8, focused_.rows(), focused_.cols(), filepath, config_.georef);
        }
    }
}

void SARDopplerFocuser::export_binary(const std::string& filepath) const {
    binary_writer_.write_real_matrix(backscatter_, filepath);
}

void SARDopplerFocuser::load_polsar_channels(
    const std::string& hh_file,
    const std::string& hv_file,
    const std::string& vh_file,
    const std::string& vv_file,
    UInt64 header_bytes) {
    if (!initialized_) initialize();
    polsar_pipeline_.load_quad_polar_channels(hh_file, hv_file, vh_file, vv_file, header_bytes);
    polsar_enabled_ = true;
}

void SARDopplerFocuser::run_polsar_decomposition() {
    if (!initialized_) initialize();
    polsar_pipeline_.run_decomposition();
}

}
