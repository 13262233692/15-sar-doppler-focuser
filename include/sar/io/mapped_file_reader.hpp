#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <string>
#include <span>
#include <cstdint>
#include <memory>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace sar::io {

class MappedFileReader {
public:
    MappedFileReader() = default;
    explicit MappedFileReader(const std::string& filepath);
    ~MappedFileReader();

    MappedFileReader(const MappedFileReader&) = delete;
    MappedFileReader& operator=(const MappedFileReader&) = delete;

    MappedFileReader(MappedFileReader&& other) noexcept;
    MappedFileReader& operator=(MappedFileReader&& other) noexcept;

    void open(const std::string& filepath);
    void close() noexcept;

    bool is_open() const noexcept { return data_ != nullptr; }

    UInt64 size() const noexcept { return file_size_; }

    const UInt8* data() const noexcept { return data_; }

    std::span<const UInt8> all() const noexcept {
        return {data_, static_cast<size_t>(file_size_)};
    }

    std::span<const UInt8> region(UInt64 offset, UInt64 length) const {
        if (offset + length > file_size_) {
            throw IOException("MappedFileReader::region - out of bounds");
        }
        return {data_ + offset, static_cast<size_t>(length)};
    }

    template<typename T>
    const T* at(UInt64 byte_offset) const {
        if (byte_offset + sizeof(T) > file_size_) {
            throw IOException("MappedFileReader::at - out of bounds");
        }
        return reinterpret_cast<const T*>(data_ + byte_offset);
    }

private:
#if defined(_WIN32) || defined(_WIN64)
    HANDLE    file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE    map_handle_  = NULL;
#else
    int       fd_          = -1;
#endif
    UInt8*    data_        = nullptr;
    UInt64    file_size_   = 0;
};

}
