#include <sar/io/mapped_file_reader.hpp>

namespace sar::io {

MappedFileReader::MappedFileReader(const std::string& filepath) {
    open(filepath);
}

MappedFileReader::~MappedFileReader() {
    close();
}

MappedFileReader::MappedFileReader(MappedFileReader&& other) noexcept
#if defined(_WIN32) || defined(_WIN64)
    : file_handle_(other.file_handle_)
    , map_handle_(other.map_handle_)
#else
    : fd_(other.fd_)
#endif
    , data_(other.data_)
    , file_size_(other.file_size_)
{
#if defined(_WIN32) || defined(_WIN64)
    other.file_handle_ = INVALID_HANDLE_VALUE;
    other.map_handle_  = NULL;
#else
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.file_size_ = 0;
}

MappedFileReader& MappedFileReader::operator=(MappedFileReader&& other) noexcept {
    if (this != &other) {
        close();
#if defined(_WIN32) || defined(_WIN64)
        file_handle_ = other.file_handle_;
        map_handle_  = other.map_handle_;
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.map_handle_  = NULL;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        data_ = other.data_;
        file_size_ = other.file_size_;
        other.data_ = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

void MappedFileReader::open(const std::string& filepath) {
    close();

#if defined(_WIN32) || defined(_WIN64)
    file_handle_ = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw IOException("Failed to open file: " + filepath);
    }

    LARGE_INTEGER li_size;
    if (!GetFileSizeEx(file_handle_, &li_size)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw IOException("Failed to get file size: " + filepath);
    }
    file_size_ = static_cast<UInt64>(li_size.QuadPart);

    if (file_size_ == 0) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw IOException("File is empty: " + filepath);
    }

    map_handle_ = CreateFileMappingA(
        file_handle_,
        NULL,
        PAGE_READONLY,
        0, 0,
        NULL
    );
    if (map_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw IOException("Failed to create file mapping: " + filepath);
    }

    data_ = static_cast<UInt8*>(MapViewOfFile(
        map_handle_,
        FILE_MAP_READ,
        0, 0, 0
    ));
    if (data_ == nullptr) {
        CloseHandle(map_handle_);
        map_handle_ = NULL;
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw IOException("Failed to map view of file: " + filepath);
    }
#else
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw IOException("Failed to open file: " + filepath);
    }

    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw IOException("Failed to stat file: " + filepath);
    }
    file_size_ = static_cast<UInt64>(st.st_size);

    if (file_size_ == 0) {
        ::close(fd_);
        fd_ = -1;
        throw IOException("File is empty: " + filepath);
    }

    void* ptr = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw IOException("Failed to mmap file: " + filepath);
    }

    data_ = static_cast<UInt8*>(ptr);

    ::madvise(data_, file_size_, MADV_SEQUENTIAL);
#endif
}

void MappedFileReader::close() noexcept {
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
    if (data_ != nullptr && fd_ >= 0) {
        ::munmap(data_, file_size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    file_size_ = 0;
}

}
