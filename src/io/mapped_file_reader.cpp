#include <sar/io/mapped_file_reader.hpp>
#include <iostream>

namespace sar::io {

MappedFileReader::MappedFileReader(const std::string& filepath) {
    open(filepath);
}

MappedFileReader::~MappedFileReader() {
    close();
}

MappedFileReader::MappedFileReader(MappedFileReader&& other) noexcept
    : guard_(std::move(other.guard_))
{
    data_.store(other.data_.load(std::memory_order_acquire), std::memory_order_release);
    file_size_.store(other.file_size_.load(std::memory_order_acquire), std::memory_order_release);

    other.data_.store(nullptr, std::memory_order_release);
    other.file_size_.store(0, std::memory_order_release);
}

MappedFileReader& MappedFileReader::operator=(MappedFileReader&& other) noexcept {
    if (this != &other) {
        close();
        std::lock_guard<std::mutex> lk1(open_mutex_);
        std::lock_guard<std::mutex> lk2(other.open_mutex_);
        guard_ = std::move(other.guard_);
        data_.store(other.data_.load(std::memory_order_acquire), std::memory_order_release);
        file_size_.store(other.file_size_.load(std::memory_order_acquire), std::memory_order_release);
        other.data_.store(nullptr, std::memory_order_release);
        other.file_size_.store(0, std::memory_order_release);
    }
    return *this;
}

void MappedFileReader::open(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(open_mutex_);

    close();

#if defined(_WIN32) || defined(_WIN64)
    HANDLE file_handle = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL
    );
    if (file_handle == INVALID_HANDLE_VALUE) {
        throw IOException("Failed to open file: " + filepath);
    }

    LARGE_INTEGER li_size;
    if (!GetFileSizeEx(file_handle, &li_size)) {
        CloseHandle(file_handle);
        throw IOException("Failed to get file size: " + filepath);
    }
    const UInt64 fsize = static_cast<UInt64>(li_size.QuadPart);

    if (fsize == 0) {
        CloseHandle(file_handle);
        throw IOException("File is empty: " + filepath);
    }

    HANDLE map_handle = CreateFileMappingA(
        file_handle,
        NULL,
        PAGE_READONLY,
        0, 0,
        NULL
    );
    if (map_handle == NULL) {
        CloseHandle(file_handle);
        throw IOException("Failed to create file mapping: " + filepath);
    }

    void* mapped = MapViewOfFile(
        map_handle,
        FILE_MAP_READ,
        0, 0, 0
    );
    if (mapped == nullptr) {
        CloseHandle(map_handle);
        CloseHandle(file_handle);
        throw IOException("Failed to map view of file: " + filepath);
    }

    auto* byte_ptr = static_cast<UInt8*>(mapped);
    guard_ = Ptr<MappedFileGuard>(new MappedFileGuard(file_handle, map_handle, mapped, fsize));

    file_size_.store(fsize, std::memory_order_release);
    data_.store(byte_ptr, std::memory_order_release);
#else
    int fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw IOException("Failed to open file: " + filepath);
    }

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw IOException("Failed to stat file: " + filepath);
    }
    const UInt64 fsize = static_cast<UInt64>(st.st_size);

    if (fsize == 0) {
        ::close(fd);
        throw IOException("File is empty: " + filepath);
    }

    void* ptr = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd);
        throw IOException("Failed to mmap file: " + filepath);
    }

    ::madvise(ptr, fsize, MADV_SEQUENTIAL);

    auto* byte_ptr = static_cast<UInt8*>(ptr);
    guard_ = Ptr<MappedFileGuard>(new MappedFileGuard(fd, ptr, fsize));

    file_size_.store(fsize, std::memory_order_release);
    data_.store(byte_ptr, std::memory_order_release);
#endif
}

void MappedFileReader::close() noexcept {
    std::lock_guard<std::mutex> lock(open_mutex_);

    data_.store(nullptr, std::memory_order_release);
    file_size_.store(0, std::memory_order_release);

    if (guard_) {
        guard_->release();
        guard_.reset();
    }
}

}
