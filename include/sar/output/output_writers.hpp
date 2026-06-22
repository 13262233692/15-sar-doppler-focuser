#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <sar/common/progress.hpp>
#include <sar/common/config.hpp>
#include <string>
#include <vector>
#include <fstream>

namespace sar::output {

class BackscatterNormalizer {
public:
    BackscatterNormalizer() = default;

    explicit BackscatterNormalizer(const ProcessingConfig& config);

    void configure(const ProcessingConfig& config);

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void normalize_to_amplitude(const ComplexMatrix& focused,
                                RealMatrix& amplitude) const;

    void normalize_to_db(const ComplexMatrix& focused,
                         RealMatrix& backscatter_db,
                         bool auto_range = false) const;

    void normalize_to_uint8(const ComplexMatrix& focused,
                            std::vector<UInt8>& output,
                            bool auto_range = false) const;

    Real get_min_db() const noexcept { return min_db_; }
    Real get_max_db() const noexcept { return max_db_; }

private:
    Real min_db_ = -40.0f;
    Real max_db_ = 10.0f;
    bool output_db_ = true;
    ProgressReporter* progress_ = nullptr;
};

class BinaryWriter {
public:
    BinaryWriter() = default;

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void write_real_matrix(const RealMatrix& mat, const std::string& filepath) const;

    void write_complex_matrix(const ComplexMatrix& mat, const std::string& filepath) const;

    void write_uint8(const std::vector<UInt8>& data,
                     UInt64 rows, UInt64 cols,
                     const std::string& filepath) const;

    static RealMatrix read_real_matrix(const std::string& filepath,
                                       UInt64 rows, UInt64 cols);

private:
    ProgressReporter* progress_ = nullptr;
};

class GeoTIFFWriter {
public:
    GeoTIFFWriter() = default;

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    static bool is_available();

    void write_float32(const RealMatrix& data,
                       const std::string& filepath,
                       const GeoReferenceConfig& georef) const;

    void write_uint8(const std::vector<UInt8>& data,
                     UInt64 rows, UInt64 cols,
                     const std::string& filepath,
                     const GeoReferenceConfig& georef) const;

    void write_rgb(const std::vector<UInt8>& r,
                   const std::vector<UInt8>& g,
                   const std::vector<UInt8>& b,
                   UInt64 rows, UInt64 cols,
                   const std::string& filepath,
                   const GeoReferenceConfig& georef) const;

private:
    ProgressReporter* progress_ = nullptr;
};

}
