#include <sar/fft/fft_engine.hpp>
#include <sar/common/math_utils.hpp>
#include <omp.h>
#include <algorithm>
#include <bit>

namespace sar::fft {

static Int64 bit_reverse_index(Int64 i, Int64 n) noexcept {
    Int64 result = 0;
    const Int64 bits = std::bit_width(static_cast<UInt64>(n)) - 1;
    for (Int64 b = 0; b < bits; ++b) {
        if (i & (1LL << b)) {
            result |= 1LL << (bits - 1 - b);
        }
    }
    return result;
}

FFTPlan::FFTPlan(Int64 size, FFTDirection dir)
    : size_(size), dir_(dir) {
    if (size <= 0 || (size & (size - 1)) != 0) {
        throw InvalidParameterException("FFT size must be a power of 2");
    }
    build_twiddles();
}

void FFTPlan::build_twiddles() {
    using namespace sar::math;
    const Real sign = (dir_ == FFTDirection::FORWARD) ? -1.0f : 1.0f;
    bit_reverse_.resize(static_cast<size_t>(size_));
    for (Int64 i = 0; i < size_; ++i) {
        bit_reverse_[static_cast<size_t>(i)] = bit_reverse_index(i, size_);
    }
    twiddles_.resize(static_cast<size_t>(size_ / 2));
    for (Int64 k = 0; k < size_ / 2; ++k) {
        const Real phase = sign * TWO_PI * static_cast<Real>(k) / static_cast<Real>(size_);
        twiddles_[static_cast<size_t>(k)] = Complex(std::cos(phase), std::sin(phase));
    }
}

void FFTPlan::execute(std::span<Complex> data) const {
    if (static_cast<Int64>(data.size()) != size_) {
        throw InvalidParameterException("FFT data size mismatch");
    }

    Complex* const x = data.data();
    const Int64 N = size_;

    for (Int64 i = 0; i < N; ++i) {
        const Int64 j = bit_reverse_[static_cast<size_t>(i)];
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    for (Int64 s = 1; s <= std::bit_width(static_cast<UInt64>(N)) - 1; ++s) {
        const Int64 m = 1LL << s;
        const Int64 m2 = m >> 1;
        const Int64 step = N / m;

        for (Int64 k = 0; k < N; k += m) {
            for (Int64 j = 0; j < m2; ++j) {
                const Complex w = twiddles_[static_cast<size_t>(j * step)];
                const Complex t = w * x[k + j + m2];
                const Complex u = x[k + j];
                x[k + j]       = u + t;
                x[k + j + m2]  = u - t;
            }
        }
    }

    if (dir_ == FFTDirection::INVERSE) {
        const Real inv_n = 1.0f / static_cast<Real>(N);
        for (Int64 i = 0; i < N; ++i) {
            x[i] *= inv_n;
        }
    }
}

void FFTPlan::execute(std::span<Complex> in, std::span<Complex> out) const {
    if (static_cast<Int64>(in.size()) != size_ || static_cast<Int64>(out.size()) != size_) {
        throw InvalidParameterException("FFT data size mismatch");
    }
    std::copy(in.begin(), in.end(), out.begin());
    execute(out);
}

FFTEngine::FFTEngine() {
    num_threads_ = static_cast<UInt32>(std::max(1, omp_get_max_threads()));
}

FFTEngine::FFTEngine(UInt32 num_threads) {
    set_num_threads(num_threads);
}

void FFTEngine::set_num_threads(UInt32 n) {
    num_threads_ = std::max(1u, n);
}

const FFTPlan& FFTEngine::get_plan(Int64 size, FFTDirection dir) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto& cache = (dir == FFTDirection::FORWARD) ? plan_cache_fwd_ : plan_cache_inv_;
    for (const auto& p : cache) {
        if (p.size() == size) {
            return p;
        }
    }
    cache.emplace_back(size, dir);
    return cache.back();
}

void FFTEngine::fft_1d(std::span<Complex> data, FFTDirection dir) const {
    const Int64 N = static_cast<Int64>(data.size());
    const Int64 Npow2 = math::next_pow2(N);

    if (Npow2 == N) {
        const auto& plan = get_plan(N, dir);
        plan.execute(data);
    } else {
        std::vector<Complex> padded(static_cast<size_t>(Npow2), Complex(0, 0));
        std::copy(data.begin(), data.end(), padded.begin());
        const auto& plan = get_plan(Npow2, dir);
        plan.execute(padded);
        if (dir == FFTDirection::INVERSE) {
            std::copy_n(padded.begin(), N, data.begin());
        } else {
            std::copy_n(padded.begin(), N, data.begin());
        }
    }
}

void FFTEngine::fft_1d(std::span<Complex> in, std::span<Complex> out, FFTDirection dir) const {
    std::copy(in.begin(), in.end(), out.begin());
    fft_1d(out, dir);
}

void FFTEngine::fft_rows(ComplexMatrix& mat, FFTDirection dir) const {
    const Int64 rows = static_cast<Int64>(mat.rows());
    const Int64 cols = static_cast<Int64>(mat.cols());
    const Int64 cols_pow2 = math::next_pow2(cols);

    if (progress_) {
        progress_->start_stage(
            dir == FFTDirection::FORWARD ? "FFT rows (forward)" : "FFT rows (inverse)",
            static_cast<UInt64>(rows)
        );
    }

    #pragma omp parallel for schedule(dynamic) num_threads(static_cast<Int32>(num_threads_))
    for (Int64 r = 0; r < rows; ++r) {
        auto row = mat.row(static_cast<UInt64>(r));
        if (cols_pow2 == cols) {
            const auto& plan = get_plan(cols, dir);
            plan.execute(row);
        } else {
            std::vector<Complex> padded(static_cast<size_t>(cols_pow2), Complex(0, 0));
            std::copy(row.begin(), row.end(), padded.begin());
            const auto& plan = get_plan(cols_pow2, dir);
            plan.execute(padded);
            std::copy_n(padded.begin(), cols, row.begin());
        }
        if (progress_) progress_->update();
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

void FFTEngine::fft_cols(ComplexMatrix& mat, FFTDirection dir) const {
    const Int64 rows = static_cast<Int64>(mat.rows());
    const Int64 cols = static_cast<Int64>(mat.cols());
    const Int64 rows_pow2 = math::next_pow2(rows);

    if (progress_) {
        progress_->start_stage(
            dir == FFTDirection::FORWARD ? "FFT cols (forward)" : "FFT cols (inverse)",
            static_cast<UInt64>(cols)
        );
    }

    #pragma omp parallel for schedule(dynamic) num_threads(static_cast<Int32>(num_threads_))
    for (Int64 c = 0; c < cols; ++c) {
        std::vector<Complex> col_buf(static_cast<size_t>(rows_pow2), Complex(0, 0));
        for (Int64 r = 0; r < rows; ++r) {
            col_buf[static_cast<size_t>(r)] = mat.at(static_cast<UInt64>(r), static_cast<UInt64>(c));
        }
        const auto& plan = get_plan(rows_pow2, dir);
        plan.execute(col_buf);
        for (Int64 r = 0; r < rows; ++r) {
            mat.at(static_cast<UInt64>(r), static_cast<UInt64>(c)) = col_buf[static_cast<size_t>(r)];
        }
        if (progress_) progress_->update();
    }

    if (progress_) {
        progress_->finish_stage();
    }
}

MemoryPool::MemoryPool(UInt64 initial_size) {
    reserve(initial_size);
}

Complex* MemoryPool::acquire(UInt64 num_elements) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
        if (!block.in_use && block.data.size() >= num_elements) {
            block.in_use = true;
            in_use_ += num_elements;
            return block.data.data();
        }
    }
    blocks_.push_back(Block{std::vector<Complex>(num_elements), true});
    total_allocated_ += num_elements;
    in_use_ += num_elements;
    return blocks_.back().data.data();
}

void MemoryPool::release(Complex* ptr) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
        if (block.data.data() == ptr) {
            block.in_use = false;
            in_use_ -= block.data.size();
            return;
        }
    }
}

void MemoryPool::reserve(UInt64 total_elements) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_allocated_ < total_elements) {
        blocks_.push_back(Block{std::vector<Complex>(total_elements - total_allocated_), false});
        total_allocated_ = total_elements;
    }
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.clear();
    total_allocated_ = 0;
    in_use_ = 0;
}

void FFTWorkspace::reserve(UInt64 rows, UInt64 cols) {
    buf1_.resize(rows, cols);
    buf2_.resize(rows, cols);
    scratch_.resize(std::max(rows, cols) * 2);
}

}
