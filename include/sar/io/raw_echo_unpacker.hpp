#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <sar/common/progress.hpp>
#include <sar/io/mapped_file_reader.hpp>
#include <span>
#include <vector>
#include <memory>

namespace sar::io {

class RawEchoUnpacker {
public:
    RawEchoUnpacker() = default;

    RawEchoUnpacker(const SARMetadata& meta,
                    const std::string& filepath,
                    UInt64 header_bytes = 0);

    void open(const SARMetadata& meta,
              const std::string& filepath,
              UInt64 header_bytes = 0);

    void close() noexcept;

    bool is_open() const noexcept { return reader_.is_open(); }

    UInt64 azimuth_lines() const noexcept { return meta_.azimuth_lines; }
    UInt64 range_samples() const noexcept { return meta_.range_samples; }

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

    void unpack_all(ComplexMatrix& output) const;

    void unpack_range_line(UInt64 azimuth_idx,
                           std::span<Complex> out_line) const;

    void unpack_range_block(UInt64 azimuth_start,
                            UInt64 azimuth_count,
                            ComplexMatrix& out_block) const;

    std::span<Complex> unpack_range_line_scratch(UInt64 azimuth_idx,
                                                 std::span<Complex> scratch) const;

private:
    UInt64 bytes_per_range_line() const noexcept;
    UInt64 samples_per_range_line() const noexcept { return meta_.range_samples; }

    void convert_line_int8(const UInt8* raw, std::span<Complex> out) const noexcept;
    void convert_line_int16(const UInt8* raw, std::span<Complex> out) const noexcept;
    void convert_line_int32(const UInt8* raw, std::span<Complex> out) const noexcept;
    void convert_line_float32(const UInt8* raw, std::span<Complex> out) const noexcept;
    void convert_line_float64(const UInt8* raw, std::span<Complex> out) const noexcept;
    void convert_line_packed_bits(const UInt8* raw, std::span<Complex> out) const noexcept;

    SARMetadata      meta_;
    MappedFileReader reader_;
    UInt64           header_offset_ = 0;
    ProgressReporter* progress_     = nullptr;
};

class StreamingUnpacker {
public:
    StreamingUnpacker(const SARMetadata& meta,
                      const std::string& filepath,
                      UInt64 chunk_lines = 1024,
                      UInt64 header_bytes = 0);

    UInt64 total_lines() const noexcept { return meta_.azimuth_lines; }
    UInt64 chunk_size() const noexcept { return chunk_lines_; }
    UInt64 range_samples() const noexcept { return meta_.range_samples; }

    void set_progress_reporter(ProgressReporter* reporter) { unpacker_.set_progress_reporter(reporter); }

    UInt64 chunks_total() const noexcept;

    bool has_next() const noexcept { return current_line_ < meta_.azimuth_lines; }

    const ComplexMatrix& next_chunk();

    UInt64 current_start_line() const noexcept { return current_line_; }

    void reset() { current_line_ = 0; }

private:
    SARMetadata     meta_;
    RawEchoUnpacker unpacker_;
    UInt64          chunk_lines_;
    UInt64          current_line_ = 0;
    ComplexMatrix   chunk_buffer_;
};

}
