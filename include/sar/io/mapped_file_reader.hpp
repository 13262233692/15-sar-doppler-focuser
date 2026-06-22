#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <string>
#include <span>
#include <cstdint>
#include <memory>
#include <atomic>
#include <mutex>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace sar::io {

class MappedFileGuard {
public:
    MappedFileGuard() = default;

#if defined(_WIN32) || defined(_WIN64)
    MappedFileGuard(HANDLE file_handle, HANDLE map_handle, void* data, UInt64 size)
        : file_handle_(file_handle), map_handle_(map_handle), data_(static_cast<UInt8*>(data)), size_(size) {}
#else
    MappedFileGuard(int fd, void* data, UInt64 size)
        : fd_(fd), data_(static_cast<UInt8*>(data)), size_(size) {}
#endif

    ~MappedFileGuard() { release(); }

    MappedFileGuard(const MappedFileGuard&) = delete;
    MappedFileGuard& operator=(const MappedFileGuard&) = delete;

    MappedFileGuard(MappedFileGuard&& other) noexcept
#if defined(_WIN32) || defined(_WIN64)
        : file_handle_(other.file_handle_), map_handle_(other.map_handle_)
#else
        : fd_(other.fd_)
#endif
        , data_(other.data_), size_(other.size_), released_(other.released_.load(std::memory_order_relaxed))
    {
#if defined(_WIN32) || defined(_WIN64)
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.map_handle_ = NULL;
#else
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
        other.released_.store(true, std::memory_order_relaxed);
    }

    MappedFileGuard& operator=(MappedFileGuard&& other) noexcept {
        if (this != &other) {
            release();
#if defined(_WIN32) || defined(_WIN64)
            file_handle_ = other.file_handle_;
            map_handle_ = other.map_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.map_handle_ = NULL;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            data_ = other.data_;
            size_ = other.size_;
            released_.store(other.released_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.data_ = nullptr;
            other.size_ = 0;
            other.released_.store(true, std::memory_order_relaxed);
        }
        return *this;
    }

    void release() noexcept {
        bool expected = false;
        if (!released_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

#if defined(_WIN32) || defined(_WIN64)
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (map_handle_ != NULL) {
            CloseHandle(map_handle_);
            map_handle_ = NULL;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_ != nullptr && size_ > 0) {
            ::munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
    }

    const UInt8* data() const noexcept { return data_; }
    UInt64 size() const noexcept { return size_; }
    bool is_released() const noexcept { return released_.load(std::memory_order_acquire); }

private:
#if defined(_WIN32) || defined(_WIN64)
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE map_handle_ = NULL;
#else
    int fd_ = -1;
#endif
    UInt8* data_ = nullptr;
    UInt64 size_ = 0;
    std::atomic<bool> released_{false};
};

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

    bool is_open() const noexcept {
        auto* p = data_.load(std::memory_order_acquire);
        return p != nullptr;
    }

    UInt64 size() const noexcept { return file_size_.load(std::memory_order_acquire); }

    const UInt8* data() const noexcept {
        return data_.load(std::memory_order_acquire);
    }

    std::span<const UInt8> all() const noexcept {
        auto* p = data_.load(std::memory_order_acquire);
        auto sz = file_size_.load(std::memory_order_acquire);
        return {p, static_cast<size_t>(sz)};
    }

    std::span<const UInt8> region(UInt64 offset, UInt64 length) const {
        auto* p = data_.load(std::memory_order_acquire);
        auto sz = file_size_.load(std::memory_order_acquire);
        if (offset + length > sz) {
            throw IOException("MappedFileReader::region - out of bounds");
        }
        return {p + offset, static_cast<size_t>(length)};
    }

    template<typename T>
    const T* at(UInt64 byte_offset) const {
        auto* p = data_.load(std::memory_order_acquire);
        auto sz = file_size_.load(std::memory_order_acquire);
        if (byte_offset + sizeof(T) > sz) {
            throw IOException("MappedFileReader::at - out of bounds");
        }
        return reinterpret_cast<const T*>(p + byte_offset);
    }

    const UInt8* safe_ptr(UInt64 byte_offset, UInt64 required_bytes) const {
        auto* p = data_.load(std::memory_order_acquire);
        auto sz = file_size_.load(std::memory_order_acquire);
        if (byte_offset + required_bytes > sz) {
            throw IOException("MappedFileReader::safe_ptr - out of bounds: offset=" +
                std::to_string(byte_offset) + " need=" + std::to_string(required_bytes) +
                " size=" + std::to_string(sz));
        }
        return p + byte_offset;
    }

private:
    std::mutex open_mutex_;
    std::atomic<UInt8*> data_{nullptr};
    std::atomic<UInt64> file_size_{0};
    Ptr<MappedFileGuard> guard_;
};

}
