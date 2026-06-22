#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <sar/common/progress.hpp>
#include <vector>
#include <span>
#include <memory>
#include <complex>
#include <cmath>

namespace sar::fft {

enum class FFTDirection {
    FORWARD = -1,
    INVERSE = 1
};

class FFTPlan {
public:
    FFTPlan() = default;
    FFTPlan(Int64 size, FFTDirection dir);

    Int64 size() const noexcept { return size_; }
    FFTDirection direction() const noexcept { return dir_; }

    void execute(std::span<Complex> data) const;
    void execute(std::span<Complex> in, std::span<Complex> out) const;

private:
    void build_twiddles();

    Int64       size_ = 0;
    FFTDirection dir_  = FFTDirection::FORWARD;
    std::vector<Complex> twiddles_;
    std::vector<Int64>   bit_reverse_;
};

class FFTEngine {
public:
    FFTEngine();
    explicit FFTEngine(UInt32 num_threads);
    ~FFTEngine() = default;

    void set_num_threads(UInt32 n);
    UInt32 num_threads() const noexcept { return num_threads_; }

    void fft_1d(std::span<Complex> data, FFTDirection dir) const;
    void fft_1d(std::span<Complex> in, std::span<Complex> out, FFTDirection dir) const;

    void fft_rows(ComplexMatrix& mat, FFTDirection dir) const;
    void fft_cols(ComplexMatrix& mat, FFTDirection dir) const;

    void set_progress_reporter(ProgressReporter* reporter) { progress_ = reporter; }

private:
    const FFTPlan& get_plan(Int64 size, FFTDirection dir) const;

    UInt32            num_threads_ = 1;
    mutable std::vector<FFTPlan> plan_cache_fwd_;
    mutable std::vector<FFTPlan> plan_cache_inv_;
    mutable std::mutex  cache_mutex_;
    ProgressReporter*   progress_ = nullptr;
};

class MemoryPool {
public:
    MemoryPool() = default;
    explicit MemoryPool(UInt64 initial_size);

    Complex* acquire(UInt64 num_elements);
    void release(Complex* ptr) noexcept;

    std::span<Complex> acquire_span(UInt64 num_elements) {
        return {acquire(num_elements), num_elements};
    }

    void reserve(UInt64 total_elements);
    void clear();

    UInt64 pool_size() const noexcept { return total_allocated_; }
    UInt64 in_use() const noexcept { return in_use_; }

private:
    struct Block {
        std::vector<Complex> data;
        bool in_use = false;
    };
    std::vector<Block> blocks_;
    UInt64 total_allocated_ = 0;
    UInt64 in_use_          = 0;
    std::mutex mutex_;
};

class FFTWorkspace {
public:
    FFTWorkspace() = default;

    void reserve(UInt64 rows, UInt64 cols);

    ComplexMatrix& buffer1() { return buf1_; }
    ComplexMatrix& buffer2() { return buf2_; }
    std::vector<Complex>& scratch() { return scratch_; }

    void clear() {
        buf1_.clear();
        buf2_.clear();
        scratch_.clear();
    }

private:
    ComplexMatrix buf1_;
    ComplexMatrix buf2_;
    std::vector<Complex> scratch_;
};

}
