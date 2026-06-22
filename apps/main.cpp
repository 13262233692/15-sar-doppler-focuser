#include <sar/sar_focuser.hpp>
#include <sar/common/config.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

static void print_usage(const char* prog) {
    std::cout << "SAR Doppler Focuser - Synthetic Aperture Radar Range-Doppler Imaging Engine\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i <input_file    Input raw echo file (raw binary)\n";
    std::cout << "  -o <output>      Output GeoTIFF file (.tif)\n";
    std::cout << "  -b <binary>     Output binary matrix file (.bin)\n";
    std::cout << "  -az <lines>     Number of azimuth lines (default: 4096)\n";
    std::cout << "  -rg <samples>  Number of range samples (default: 4096)\n";
    std::cout << "  -fc <freq>     Carrier frequency Hz (default: 5.4e9)\n";
    std::cout << "  -bw <bw>       Bandwidth Hz (default: 100e6)\n";
    std::cout << "  -tp <sec>      Pulse duration seconds (default: 10e-6)\n";
    std::cout << "  -prf <hz>      PRF Hz (default: 2000)\n";
    std::cout << "  -fs <hz>       Range sampling rate Hz (default: 120e6)\n";
    std::cout << "  -v <m/s>       Platform velocity (default: 7500)\n";
    std::cout << "  -t <threads>   Number of OpenMP threads (default: all)\n";
    std::cout << "  -sinc <n>      Sinc interpolation kernel size (default: 8)\n";
    std::cout << "  -hdr <bytes>    File header bytes (default: 0)\n";
    std::cout << "  -dop <hz>      Fixed Doppler centroid Hz (default: estimate)\n";
    std::cout << "  -polsar        Enable PolSAR H-alpha decomposition (requires quad-pol channels)\n";
    std::cout << "  -hh <file>     HH polarization channel file\n";
    std::cout << "  -hv <file>     HV polarization channel file\n";
    std::cout << "  -vh <file>     VH polarization channel file\n";
    std::cout << "  -vv <file>     VV polarization channel file\n";
    std::cout << "  -h             Show this help\n";
}

int main(int argc, char* argv[]) {
    using namespace sar;

    if (argc < 2) {
        std::cout << "No arguments provided, running default synthetic test...\n\n";
        SARConfig cfg = SARConfig::make_default();
        cfg.metadata.azimuth_lines = 512;
        cfg.metadata.range_samples = 512;
        cfg.processing.verbose = true;
        cfg.output_geotiff = "sar_output.tif";

        SARDopplerFocuser focuser(cfg);
        focuser.set_progress_reporter([](const std::string& stage, Real pct, UInt64, UInt64) {
            if (pct >= 0.0 && pct <= 1.0) {
                const int bar = 30;
                const int filled = static_cast<int>(pct * bar);
                std::cout << "\r  [" << stage << "] [";
                for (int i = 0; i < filled; ++i) std::cout << '=';
                for (int i = filled; i < bar; ++i) std::cout << ' ';
                std::cout << "] " << std::fixed << std::setprecision(1) << (pct * 100.0f) << "%" << std::flush;
                if (pct >= 1.0f) std::cout << std::endl;
            }
        });

        try {
            focuser.run_full_pipeline();
            std::cout << "\nImaging complete. Output written to sar_output.tif\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    SARConfig cfg = SARConfig::make_default();
    std::string input_file;
    std::string hh_file, hv_file, vh_file, vv_file;
    bool polsar_enabled = false;
    UInt64 header_bytes = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-i") input_file = next("-i");
        else if (arg == "-o") cfg.output_geotiff = next("-o");
        else if (arg == "-b") cfg.output_binary = next("-b");
        else if (arg == "-az") cfg.metadata.azimuth_lines = std::stoull(next("-az"));
        else if (arg == "-rg") cfg.metadata.range_samples = std::stoull(next("-rg"));
        else if (arg == "-fc") cfg.metadata.carrier_freq_hz = std::stod(next("-fc"));
        else if (arg == "-bw") cfg.metadata.bandwidth_hz = std::stod(next("-bw"));
        else if (arg == "-tp") cfg.metadata.pulse_duration_s = std::stod(next("-tp"));
        else if (arg == "-prf") cfg.metadata.prf_hz = std::stod(next("-prf"));
        else if (arg == "-fs") cfg.metadata.range_sampling_hz = std::stod(next("-fs"));
        else if (arg == "-v") cfg.metadata.platform_velocity = std::stod(next("-v"));
        else if (arg == "-t") cfg.processing.omp_num_threads = std::stoul(next("-t"));
        else if (arg == "-sinc") cfg.processing.sinc_interp_kernel_size = std::stoi(next("-sinc"));
        else if (arg == "-hdr") header_bytes = std::stoull(next("-hdr"));
        else if (arg == "-dop") {
            cfg.processing.do_doppler_estimation = false;
            cfg.processing.fixed_doppler_centroid = std::stod(next("-dop"));
        } else if (arg == "-polsar") {
            polsar_enabled = true;
        } else if (arg == "-hh") hh_file = next("-hh");
        else if (arg == "-hv") hv_file = next("-hv");
        else if (arg == "-vh") vh_file = next("-vh");
        else if (arg == "-vv") vv_file = next("-vv");
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    cfg.metadata.wavelength_m = math::SPEED_OF_LIGHT / cfg.metadata.carrier_freq_hz;
    cfg.metadata.range_spacing_m = math::SPEED_OF_LIGHT / (2.0f * cfg.metadata.range_sampling_hz);
    cfg.metadata.azimuth_spacing_m = cfg.metadata.platform_velocity / cfg.metadata.prf_hz;
    cfg.processing.verbose = true;

    if (!input_file.empty()) {
        cfg.input_file = input_file;
    }

    if (!hh_file.empty() || !hv_file.empty() || !vh_file.empty() || !vv_file.empty()) {
        polsar_enabled = true;
    }

    std::cout << "SAR Doppler Focuser\n";
    std::cout << "  Azimuth lines: " << cfg.metadata.azimuth_lines << "\n";
    std::cout << "  Range samples: " << cfg.metadata.range_samples << "\n";
    std::cout << "  Carrier freq: " << cfg.metadata.carrier_freq_hz << " Hz\n";
    std::cout << "  Bandwidth: " << cfg.metadata.bandwidth_hz << " Hz\n";
    std::cout << "  Platform velocity: " << cfg.metadata.platform_velocity << " m/s\n";
    std::cout << "  Wavelength: " << cfg.metadata.wavelength_m << " m\n";
    std::cout << "  Threads: " << (cfg.processing.omp_num_threads == 0 ? omp_get_max_threads() : cfg.processing.omp_num_threads) << "\n";
    if (polsar_enabled) {
        std::cout << "  PolSAR H-alpha decomposition: ENABLED\n";
    }
    std::cout << "\n";

    SARDopplerFocuser focuser(cfg);

    try {
        if (polsar_enabled) {
            focuser.enable_polsar(true);
            if (!hh_file.empty() || !hv_file.empty() || !vh_file.empty() || !vv_file.empty()) {
                focuser.load_polsar_channels(hh_file, hv_file, vh_file, vv_file, header_bytes);
            }
        }
        focuser.run_full_pipeline();
        std::cout << "\nImaging pipeline completed successfully.\n";
        if (!cfg.output_geotiff.empty()) {
            std::cout << "  GeoTIFF output: " << cfg.output_geotiff << "\n";
        }
        if (!cfg.output_binary.empty()) std::cout << "  Binary output: " << cfg.output_binary << "\n";
        if (polsar_enabled && focuser.polsar_pipeline().has_oil_spill()) {
            std::cout << "  [POLSAR] Oil spill suspected pixels: "
                      << focuser.polsar_pipeline().oil_pixel_count() << "\n";
            std::cout << "  [POLSAR] Red calibration grid overlaid on GeoTIFF\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
