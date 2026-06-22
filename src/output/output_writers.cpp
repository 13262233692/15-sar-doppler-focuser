#include <sar/output/output_writers.hpp>
#include <sar/common/math_utils.hpp>
#include <omp.h>
#include <algorithm>
#include <limits>
#include <iostream>

#ifndef SAR_NO_GDAL
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"
#endif

namespace sar::output {

using namespace sar::math;

BackscatterNormalizer::BackscatterNormalizer(const ProcessingConfig& config) {
    configure(config);
}

void BackscatterNormalizer::configure(const ProcessingConfig& config) {
    min_db_ = config.output_min_db;
    max_db_ = config.output_max_db;
    output_db_ = config.output_backscatter_db;
}

void BackscatterNormalizer::normalize_to_amplitude(
    const ComplexMatrix& focused,
    RealMatrix& amplitude) const {

    const UInt64 rows = focused.rows();
    const UInt64 cols = focused.cols();
    amplitude.resize(rows, cols);

    if (progress_) {
        progress_->start_stage("Normalizing amplitude", rows * cols);
    }

    #pragma omp parallel for schedule(dynamic)
    for (Int64 r = 0; r < static_cast<Int64>(rows); ++r) {
        for (UInt64 c = 0; c < cols; ++c) {
            amplitude.at(static_cast<UInt64>(r), c) = std::abs(focused.at(static_cast<UInt64>(r), c));
        }
        if (progress_) progress_->update(cols);
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

void BackscatterNormalizer::normalize_to_db(
    const ComplexMatrix& focused,
    RealMatrix& backscatter_db,
    bool auto_range) const {

    const UInt64 rows = focused.rows();
    const UInt64 cols = focused.cols();
    backscatter_db.resize(rows, cols);

    Real global_max = -std::numeric_limits<Real>::infinity();
    Real global_min = std::numeric_limits<Real>::infinity();

    #pragma omp parallel
    {
        Real thread_max = -std::numeric_limits<Real>::infinity();
        Real thread_min = std::numeric_limits<Real>::infinity();

        #pragma omp for schedule(dynamic) nowait
        for (Int64 r = 0; r < static_cast<Int64>(rows); ++r) {
            for (UInt64 c = 0; c < cols; ++c) {
                const Real amp = std::abs(focused.at(static_cast<UInt64>(r), c));
                const Real db = amp2db(amp);
                backscatter_db.at(static_cast<UInt64>(r), c) = db;
                if (db > thread_max) thread_max = db;
                if (db < thread_min) thread_min = db;
            }
        }

        #pragma omp critical
        {
            if (thread_max > global_max) global_max = thread_max;
            if (thread_min < global_min) global_min = thread_min;
        }
    }

    if (progress_) {
        progress_->start_stage("Normalizing backscatter to dB", rows * cols);
    }

    const Real db_min = auto_range ? global_min : min_db_;
    const Real db_max = auto_range ? global_max : max_db_;
    const Real db_range = db_max - db_min;

    if (db_range > 1e-10f) {
        #pragma omp parallel for schedule(dynamic)
        for (Int64 r = 0; r < static_cast<Int64>(rows); ++r) {
            for (UInt64 c = 0; c < cols; ++c) {
                Real& v = backscatter_db.at(static_cast<UInt64>(r), c);
                v = clamp((v - db_min) / db_range, 0.0f, 1.0f);
            }
            if (progress_) progress_->update(cols);
        }
    } else {
        backscatter_db.fill(0.0f);
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

void BackscatterNormalizer::normalize_to_uint8(
    const ComplexMatrix& focused,
    std::vector<UInt8>& output,
    bool auto_range) const {

    const UInt64 rows = focused.rows();
    const UInt64 cols = focused.cols();
    output.resize(rows * cols);

    Real global_max = -std::numeric_limits<Real>::infinity();
    Real global_min = std::numeric_limits<Real>::infinity();

    std::vector<Real> db_vals(rows * cols);

    #pragma omp parallel
    {
        Real thread_max = -std::numeric_limits<Real>::infinity();
        Real thread_min = std::numeric_limits<Real>::infinity();

        #pragma omp for schedule(dynamic) nowait
        for (Int64 i = 0; i < static_cast<Int64>(rows * cols); ++i) {
            const Real amp = std::abs(focused.data()[i]);
            const Real db = amp2db(amp);
            db_vals[static_cast<size_t>(i)] = db;
            if (db > thread_max) thread_max = db;
            if (db < thread_min) thread_min = db;
        }

        #pragma omp critical
        {
            if (thread_max > global_max) global_max = thread_max;
            if (thread_min < global_min) global_min = thread_min;
        }
    }

    const Real db_min = auto_range ? global_min : min_db_;
    const Real db_max = auto_range ? global_max : max_db_;
    const Real db_range = db_max - db_min;

    if (progress_) {
        progress_->start_stage("Quantizing to UInt8", rows * cols);
    }

    if (db_range > 1e-10f) {
        const Real scale = 255.0f / db_range;
        #pragma omp parallel for schedule(dynamic)
        for (Int64 i = 0; i < static_cast<Int64>(rows * cols); ++i) {
            Real norm = clamp((db_vals[static_cast<size_t>(i)] - db_min) * scale, 0.0f, 255.0f);
            output[static_cast<size_t>(i)] = static_cast<UInt8>(norm + 0.5f);
        }
    } else {
        std::fill(output.begin(), output.end(), UInt8(0));
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

void BinaryWriter::write_real_matrix(
    const RealMatrix& mat,
    const std::string& filepath) const {

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw IOException("Failed to open file for writing: " + filepath);
    }

    const UInt64 rows = mat.rows();
    const UInt64 cols = mat.cols();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(mat.data()), sizeof(Real) * rows * cols);

    if (!ofs) {
        throw IOException("Failed to write data to: " + filepath);
    }
}

void BinaryWriter::write_complex_matrix(
    const ComplexMatrix& mat,
    const std::string& filepath) const {

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw IOException("Failed to open file for writing: " + filepath);
    }

    const UInt64 rows = mat.rows();
    const UInt64 cols = mat.cols();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(mat.data()), sizeof(Complex) * rows * cols);

    if (!ofs) {
        throw IOException("Failed to write data to: " + filepath);
    }
}

void BinaryWriter::write_uint8(
    const std::vector<UInt8>& data,
    UInt64 rows, UInt64 cols,
    const std::string& filepath) const {

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw IOException("Failed to open file for writing: " + filepath);
    }

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(UInt64));
    ofs.write(reinterpret_cast<const char*>(data.data()), sizeof(UInt8) * rows * cols);

    if (!ofs) {
        throw IOException("Failed to write data to: " + filepath);
    }
}

RealMatrix BinaryWriter::read_real_matrix(
    const std::string& filepath,
    UInt64 rows, UInt64 cols) {

    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw IOException("Failed to open file for reading: " + filepath);
    }

    UInt64 r, c;
    ifs.read(reinterpret_cast<char*>(&r), sizeof(UInt64));
    ifs.read(reinterpret_cast<char*>(&c), sizeof(UInt64));

    if (rows == 0) rows = r;
    if (cols == 0) cols = c;

    RealMatrix mat(rows, cols);
    const UInt64 n = std::min(rows * cols, r * c);
    ifs.read(reinterpret_cast<char*>(mat.data()), sizeof(Real) * n);

    return mat;
}

bool GeoTIFFWriter::is_available() {
#ifndef SAR_NO_GDAL
    return true;
#else
    return false;
#endif
}

void GeoTIFFWriter::write_float32(
    const RealMatrix& data,
    const std::string& filepath,
    const GeoReferenceConfig& georef) const {

#ifndef SAR_NO_GDAL
    GDALAllRegister();

    const UInt64 rows = data.rows();
    const UInt64 cols = data.cols();

    char** papszOptions = nullptr;
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == nullptr) {
        throw IOException("GDAL GTiff driver not available");
    }

    GDALDataset* poDS = poDriver->Create(
        filepath.c_str(),
        static_cast<int>(cols),
        static_cast<int>(rows),
        1,
        GDT_Float32,
        papszOptions
    );
    if (poDS == nullptr) {
        throw IOException("Failed to create GeoTIFF: " + filepath);
    }

    if (georef.enable_georef) {
        double adfGeoTransform[6] = {
            georef.origin_lon,
            georef.pixel_spacing_lon,
            0.0,
            georef.origin_lat,
            0.0,
            -georef.pixel_spacing_lat
        };
        poDS->SetGeoTransform(adfGeoTransform);

        OGRSpatialReference oSRS;
        if (oSRS.importFromEPSG(4326) == OGRERR_NONE) {
            char* pszSRS_WKT = nullptr;
            oSRS.exportToWkt(&pszSRS_WKT);
            poDS->SetProjection(pszSRS_WKT);
            CPLFree(pszSRS_WKT);
        }
    }

    GDALRasterBand* poBand = poDS->GetRasterBand(1);
    poBand->SetNoDataValue(-9999.0);

    if (progress_) {
        progress_->start_stage("Writing GeoTIFF (Float32)", rows);
    }

    for (UInt64 r = 0; r < rows; ++r) {
        const CPLErr err = poBand->RasterIO(
            GF_Write,
            0, static_cast<int>(r),
            static_cast<int>(cols), 1,
            const_cast<Real*>(data.row(r).data()),
            static_cast<int>(cols), 1,
            GDT_Float32,
            0, 0,
            nullptr
        );
        if (err != CE_None) {
            GDALClose(poDS);
            throw IOException("Failed to write GeoTIFF row " + std::to_string(r));
        }
        if (progress_) progress_->update();
    }

    GDALClose(poDS);

    if (progress_) {
        progress_->finish_stage();
    }
#else
    std::cerr << "[SAR] WARNING: GeoTIFF output not available - GDAL was not found during build" << std::endl;
    return;
#endif
}

void GeoTIFFWriter::write_uint8(
    const std::vector<UInt8>& data,
    UInt64 rows, UInt64 cols,
    const std::string& filepath,
    const GeoReferenceConfig& georef) const {

#ifndef SAR_NO_GDAL
    GDALAllRegister();

    char** papszOptions = nullptr;
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == nullptr) {
        throw IOException("GDAL GTiff driver not available");
    }

    GDALDataset* poDS = poDriver->Create(
        filepath.c_str(),
        static_cast<int>(cols),
        static_cast<int>(rows),
        1,
        GDT_Byte,
        papszOptions
    );
    if (poDS == nullptr) {
        throw IOException("Failed to create GeoTIFF: " + filepath);
    }

    if (georef.enable_georef) {
        double adfGeoTransform[6] = {
            georef.origin_lon,
            georef.pixel_spacing_lon,
            0.0,
            georef.origin_lat,
            0.0,
            -georef.pixel_spacing_lat
        };
        poDS->SetGeoTransform(adfGeoTransform);

        OGRSpatialReference oSRS;
        if (oSRS.importFromEPSG(4326) == OGRERR_NONE) {
            char* pszSRS_WKT = nullptr;
            oSRS.exportToWkt(&pszSRS_WKT);
            poDS->SetProjection(pszSRS_WKT);
            CPLFree(pszSRS_WKT);
        }
    }

    GDALRasterBand* poBand = poDS->GetRasterBand(1);

    if (progress_) {
        progress_->start_stage("Writing GeoTIFF (UInt8)", rows);
    }

    for (UInt64 r = 0; r < rows; ++r) {
        const CPLErr err = poBand->RasterIO(
            GF_Write,
            0, static_cast<int>(r),
            static_cast<int>(cols), 1,
            const_cast<UInt8*>(data.data() + r * cols),
            static_cast<int>(cols), 1,
            GDT_Byte,
            0, 0,
            nullptr
        );
        if (err != CE_None) {
            GDALClose(poDS);
            throw IOException("Failed to write GeoTIFF row " + std::to_string(r));
        }
        if (progress_) progress_->update();
    }

    GDALClose(poDS);

    if (progress_) {
        progress_->finish_stage();
    }
#else
    std::cerr << "[SAR] WARNING: GeoTIFF output not available - GDAL was not found during build" << std::endl;
    return;
#endif
}

void GeoTIFFWriter::write_rgb(
    const std::vector<UInt8>& r,
    const std::vector<UInt8>& g,
    const std::vector<UInt8>& b,
    UInt64 rows, UInt64 cols,
    const std::string& filepath,
    const GeoReferenceConfig& georef) const {

#ifndef SAR_NO_GDAL
    GDALAllRegister();

    char** papszOptions = nullptr;
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == nullptr) {
        throw IOException("GDAL GTiff driver not available");
    }

    GDALDataset* poDS = poDriver->Create(
        filepath.c_str(),
        static_cast<int>(cols),
        static_cast<int>(rows),
        3,
        GDT_Byte,
        papszOptions
    );
    if (poDS == nullptr) {
        throw IOException("Failed to create GeoTIFF: " + filepath);
    }

    if (georef.enable_georef) {
        double adfGeoTransform[6] = {
            georef.origin_lon,
            georef.pixel_spacing_lon,
            0.0,
            georef.origin_lat,
            0.0,
            -georef.pixel_spacing_lat
        };
        poDS->SetGeoTransform(adfGeoTransform);

        OGRSpatialReference oSRS;
        if (oSRS.importFromEPSG(4326) == OGRERR_NONE) {
            char* pszSRS_WKT = nullptr;
            oSRS.exportToWkt(&pszSRS_WKT);
            poDS->SetProjection(pszSRS_WKT);
            CPLFree(pszSRS_WKT);
        }
    }

    const std::vector<UInt8>* bands[3] = {&r, &g, &b};
    for (int b = 0; b < 3; ++b) {
        GDALRasterBand* poBand = poDS->GetRasterBand(b + 1);
        for (UInt64 row = 0; row < rows; ++row) {
            poBand->RasterIO(
                GF_Write,
                0, static_cast<int>(row),
                static_cast<int>(cols), 1,
                const_cast<UInt8*>(bands[b]->data() + row * cols),
                static_cast<int>(cols), 1,
                GDT_Byte,
                0, 0,
                nullptr
            );
        }
    }

    GDALClose(poDS);
#else
    std::cerr << "[SAR] WARNING: GeoTIFF output not available - GDAL was not found during build" << std::endl;
    return;
#endif
}

}
