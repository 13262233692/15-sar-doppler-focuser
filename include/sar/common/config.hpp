#pragma once

#include <sar/common/types.hpp>
#include <string>
#include <optional>

namespace sar {

struct ProcessingConfig {
    bool   apply_range_window      = true;
    bool   apply_azimuth_window    = true;
    Real   range_window_beta       = 25.0;
    Real   azimuth_window_beta     = 25.0;
    Int32  sinc_interp_kernel_size = 8;
    bool   do_doppler_estimation   = true;
    Real   fixed_doppler_centroid  = 0.0;
    bool   output_backscatter_db   = true;
    Real   output_min_db           = -40.0;
    Real   output_max_db           = 10.0;
    UInt32 omp_num_threads         = 0;
    bool   verbose                 = true;
};

struct GeoReferenceConfig {
    bool   enable_georef    = false;
    double origin_lon       = 0.0;
    double origin_lat       = 0.0;
    double pixel_spacing_lon = 1.0e-5;
    double pixel_spacing_lat = 1.0e-5;
    std::string projection  = "EPSG:4326";
};

class SARConfig {
public:
    SARMetadata       metadata;
    ProcessingConfig  processing;
    GeoReferenceConfig georef;
    std::string       input_file;
    std::string       output_geotiff;
    std::string       output_binary;

    static SARConfig load_from_file(const std::string& path);
    static SARConfig make_default();

    void validate() const;
};

}
