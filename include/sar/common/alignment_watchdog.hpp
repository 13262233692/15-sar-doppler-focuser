#pragma once

#include <sar/common/types.hpp>
#include <sar/common/exceptions.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <span>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace sar::io {

constexpr UInt64 CACHE_LINE_SIZE = 64;

inline UInt64 align_up(UInt64 value, UInt64 alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(const void* ptr, UInt64 alignment = CACHE_LINE_SIZE) noexcept {
    return (reinterpret_cast<UInt64>(ptr) & (alignment - 1)) == 0;
}

class AlignmentWatchdog {
public:
    static void verify_aligned(const void* ptr, const char* label = "pointer") {
        if (!is_aligned(ptr, CACHE_LINE_SIZE)) {
            throw MemoryException(
                std::string("AlignmentWatchdog: ") + label +
                " at address " + std::to_string(reinterpret_cast<UInt64>(ptr)) +
                " is not aligned to " + std::to_string(CACHE_LINE_SIZE) + " bytes");
        }
    }

    static UInt64 cache_lines_spanned(UInt64 byte_offset, UInt64 byte_length) noexcept {
        if (byte_length == 0) return 0;
        const UInt64 first_line = byte_offset / CACHE_LINE_SIZE;
        const UInt64 last_byte = byte_offset + byte_length - 1;
        const UInt64 last_line = last_byte / CACHE_LINE_SIZE;
        return last_line - first_line + 1;
    }
};

class ThreadBarrier {
public:
    explicit ThreadBarrier(UInt32 count) : threshold_(count), count_(count), generation_(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const UInt32 gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    UInt32 threshold_;
    UInt32 count_;
    UInt32 generation_;
};

class WatermarkTracker {
public:
    explicit WatermarkTracker(UInt64 total_lines)
        : total_lines_(total_lines)
        , completed_lines_(0)
        , watermark_(0)
        , per_thread_state_(0)
    {}

    void initialize(UInt32 num_threads) {
        per_thread_state_.resize(num_threads, {0, 0});
    }

    void mark_line_written(UInt32 thread_id, UInt64 azimuth_idx) {
        if (thread_id < per_thread_state_.size()) {
            per_thread_state_[thread_id].last_written = azimuth_idx;
            per_thread_state_[thread_id].count++;
        }
        UInt64 prev = completed_lines_.fetch_add(1, std::memory_order_release);
        if (prev + 1 == total_lines_) {
            UInt64 wm = watermark_.load(std::memory_order_acquire);
            UInt64 new_wm = compute_contiguous_watermark();
            while (wm < new_wm && !watermark_.compare_exchange_weak(wm, new_wm, std::memory_order_release, std::memory_order_relaxed)) {}
        }
    }

    UInt64 watermark() const noexcept {
        return watermark_.load(std::memory_order_acquire);
    }

    UInt64 completed() const noexcept {
        return completed_lines_.load(std::memory_order_acquire);
    }

    bool is_complete() const noexcept {
        return completed_lines_.load(std::memory_order_acquire) >= total_lines_;
    }

    void reset() {
        completed_lines_.store(0, std::memory_order_relaxed);
        watermark_.store(0, std::memory_order_relaxed);
        for (auto& s : per_thread_state_) {
            s.last_written = 0;
            s.count = 0;
        }
    }

    void verify_safe_read(UInt64 azimuth_idx) const {
        const UInt64 wm = watermark_.load(std::memory_order_acquire);
        if (azimuth_idx > wm) {
            throw MemoryException(
                "WatermarkTracker: unsafe read at azimuth " +
                std::to_string(azimuth_idx) +
                " exceeds write watermark " + std::to_string(wm));
        }
    }

    void wait_until_watermark(UInt64 target, UInt64 spin_count = 1000) const {
        for (UInt64 i = 0; i < spin_count; ++i) {
            if (watermark_.load(std::memory_order_acquire) >= target) return;
        }
        while (watermark_.load(std::memory_order_acquire) < target) {
            std::this_thread::yield();
        }
    }

private:
    UInt64 compute_contiguous_watermark() const {
        std::vector<UInt64> written;
        written.reserve(per_thread_state_.size());
        for (const auto& s : per_thread_state_) {
            if (s.count > 0) written.push_back(s.last_written);
        }
        if (written.empty()) return 0;
        std::sort(written.begin(), written.end());

        UInt64 contiguous = 0;
        for (UInt64 i = 0; i <= written[0]; ++i) {
            bool found = false;
            for (auto w : written) {
                if (w == i) { found = true; break; }
            }
            if (!found && i > 0) break;
            contiguous = i + 1;
        }
        return contiguous > 0 ? contiguous - 1 : 0;
    }

    struct ThreadState {
        UInt64 last_written = 0;
        UInt64 count = 0;
    };

    UInt64 total_lines_;
    std::atomic<UInt64> completed_lines_;
    std::atomic<UInt64> watermark_;
    std::vector<ThreadState> per_thread_state_;
};

}
