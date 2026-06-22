#include <sar/io/raw_echo_unpacker.hpp>
#include <sar/common/math_utils.hpp>
#include <omp.h>
#include <bit>
#include <algorithm>

namespace sar::io {

static inline Real normalize_int8(Int8 v) noexcept {
    return static_cast<Real>(v) * (1.0f / 127.0f);
}

static inline Real normalize_int16(Int16 v) noexcept {
    return static_cast<Real>(v) * (1.0f / 32767.0f);
}

static inline Real normalize_int32(Int32 v) noexcept {
    return static_cast<Real>(v) * (1.0f / 2147483647.0f);
}

RawEchoUnpacker::RawEchoUnpacker(const SARMetadata& meta,
                                 const std::string& filepath,
                                 UInt64 header_bytes) {
    open(meta, filepath, header_bytes);
}

void RawEchoUnpacker::open(const SARMetadata& meta,
                           const std::string& filepath,
                           UInt64 header_bytes) {
    meta_ = meta;
    header_offset_ = header_bytes;
    reader_.open(filepath);

    const UInt64 expected = header_offset_ + bytes_per_range_line() * meta_.azimuth_lines;
    if (reader_.size() < expected) {
        throw IOException(
            "File size (" + std::to_string(reader_.size()) +
            ") is smaller than expected (" + std::to_string(expected) + ")"
        );
    }
}

void RawEchoUnpacker::close() noexcept {
    reader_.close();
}

UInt64 RawEchoUnpacker::bytes_per_range_line() const noexcept {
    const UInt64 samples = meta_.range_samples;
    switch (meta_.sample_format) {
        case SampleFormat::COMPLEX_INT8:     return samples * 2;
        case SampleFormat::COMPLEX_INT16:    return samples * 4;
        case SampleFormat::COMPLEX_INT32:    return samples * 8;
        case SampleFormat::COMPLEX_FLOAT32:  return samples * 8;
        case SampleFormat::COMPLEX_FLOAT64:  return samples * 16;
        case SampleFormat::RAW_PACKED_BITS:  return (samples * 2 * meta_.bits_per_sample + 7) / 8;
    }
    return 0;
}

void RawEchoUnpacker::convert_line_int8(const UInt8* raw, std::span<Complex> out) const noexcept {
    const auto* src = reinterpret_cast<const Int8*>(raw);
    for (UInt64 i = 0; i < out.size(); ++i) {
        out[i] = Complex(normalize_int8(src[2*i]), normalize_int8(src[2*i+1]));
    }
}

void RawEchoUnpacker::convert_line_int16(const UInt8* raw, std::span<Complex> out) const noexcept {
    const auto* src = reinterpret_cast<const Int16*>(raw);
    for (UInt64 i = 0; i < out.size(); ++i) {
        out[i] = Complex(normalize_int16(src[2*i]), normalize_int16(src[2*i+1]));
    }
}

void RawEchoUnpacker::convert_line_int32(const UInt8* raw, std::span<Complex> out) const noexcept {
    const auto* src = reinterpret_cast<const Int32*>(raw);
    for (UInt64 i = 0; i < out.size(); ++i) {
        out[i] = Complex(normalize_int32(src[2*i]), normalize_int32(src[2*i+1]));
    }
}

void RawEchoUnpacker::convert_line_float32(const UInt8* raw, std::span<Complex> out) const noexcept {
    const auto* src = reinterpret_cast<const float*>(raw);
    for (UInt64 i = 0; i < out.size(); ++i) {
        out[i] = Complex(static_cast<Real>(src[2*i]), static_cast<Real>(src[2*i+1]));
    }
}

void RawEchoUnpacker::convert_line_float64(const UInt8* raw, std::span<Complex> out) const noexcept {
    const auto* src = reinterpret_cast<const double*>(raw);
    for (UInt64 i = 0; i < out.size(); ++i) {
        out[i] = Complex(static_cast<Real>(src[2*i]), static_cast<Real>(src[2*i+1]));
    }
}

void RawEchoUnpacker::convert_line_packed_bits(const UInt8* raw, std::span<Complex> out) const noexcept {
    const UInt32 bits_per_val = meta_.bits_per_sample;
    const UInt32 max_val = (1u << bits_per_val) - 1u;
    const Real inv_range = 2.0f / static_cast<Real>(max_val);
    const UInt32 mask = max_val;

    UInt32 bit_buf = 0;
    Int32  bit_cnt = 0;
    UInt64 byte_idx = 0;
    const UInt64 total_vals = out.size() * 2;

    for (UInt64 vi = 0; vi < total_vals; ++vi) {
        while (bit_cnt < static_cast<Int32>(bits_per_val)) {
            bit_buf = (bit_buf << 8) | raw[byte_idx++];
            bit_cnt += 8;
        }
        bit_cnt -= bits_per_val;
        const UInt32 raw_val = (bit_buf >> bit_cnt) & mask;
        const Real fval = static_cast<Real>(raw_val) * inv_range - 1.0f;
        if (vi % 2 == 0) {
            out[vi / 2].real(fval);
        } else {
            out[vi / 2].imag(fval);
        }
    }
}

void RawEchoUnpacker::unpack_range_line(UInt64 azimuth_idx,
                                        std::span<Complex> out_line) const {
    if (azimuth_idx >= meta_.azimuth_lines) {
        throw InvalidParameterException("azimuth_idx out of range");
    }
    if (out_line.size() < meta_.range_samples) {
        throw InvalidParameterException("output span too small");
    }

    const UInt64 offset = header_offset_ + azimuth_idx * bytes_per_range_line();
    const UInt64 required = bytes_per_range_line();
    const UInt8* raw = reader_.safe_ptr(offset, required);

    switch (meta_.sample_format) {
        case SampleFormat::COMPLEX_INT8:
            convert_line_int8(raw, out_line);
            break;
        case SampleFormat::COMPLEX_INT16:
            convert_line_int16(raw, out_line);
            break;
        case SampleFormat::COMPLEX_INT32:
            convert_line_int32(raw, out_line);
            break;
        case SampleFormat::COMPLEX_FLOAT32:
            convert_line_float32(raw, out_line);
            break;
        case SampleFormat::COMPLEX_FLOAT64:
            convert_line_float64(raw, out_line);
            break;
        case SampleFormat::RAW_PACKED_BITS:
            convert_line_packed_bits(raw, out_line);
            break;
    }
}

std::span<Complex> RawEchoUnpacker::unpack_range_line_scratch(
    UInt64 azimuth_idx,
    std::span<Complex> scratch) const {
    unpack_range_line(azimuth_idx, scratch);
    return scratch.first(meta_.range_samples);
}

void RawEchoUnpacker::unpack_range_block(UInt64 azimuth_start,
                                         UInt64 azimuth_count,
                                         ComplexMatrix& out_block) const {
    if (azimuth_start + azimuth_count > meta_.azimuth_lines) {
        throw InvalidParameterException("block out of range");
    }
    out_block.resize(azimuth_count, meta_.range_samples);
    out_block.reset_watermark();

    if (progress_) {
        progress_->start_stage("Unpacking echo data", azimuth_count);
    }

    #pragma omp parallel for schedule(dynamic)
    for (Int64 i = 0; i < static_cast<Int64>(azimuth_count); ++i) {
        const UInt64 row_idx = static_cast<UInt64>(i);
        auto line = out_block.row(row_idx);
        unpack_range_line(azimuth_start + row_idx, line);

        out_block.ensure_row_watermark(row_idx);

        if (progress_) progress_->update();
    }

    out_block.ensure_row_watermark(azimuth_count - 1);

    if (progress_) {
        progress_->finish_stage();
    }
}

void RawEchoUnpacker::unpack_all(ComplexMatrix& output) const {
    unpack_range_block(0, meta_.azimuth_lines, output);
}

StreamingUnpacker::StreamingUnpacker(const SARMetadata& meta,
                                     const std::string& filepath,
                                     UInt64 chunk_lines,
                                     UInt64 header_bytes)
    : meta_(meta)
    , chunk_lines_(chunk_lines) {
    unpacker_.open(meta, filepath, header_bytes);
}

UInt64 StreamingUnpacker::chunks_total() const noexcept {
    if (chunk_lines_ == 0) return 0;
    return (meta_.azimuth_lines + chunk_lines_ - 1) / chunk_lines_;
}

const ComplexMatrix& StreamingUnpacker::next_chunk() {
    const UInt64 remaining = meta_.azimuth_lines - current_line_;
    const UInt64 n = (std::min)(chunk_lines_, remaining);
    unpacker_.unpack_range_block(current_line_, n, chunk_buffer_);
    current_line_ += n;
    return chunk_buffer_;
}

}
