#pragma once

#include <sar/common/types.hpp>
#include <atomic>
#include <functional>
#include <string>
#include <chrono>

namespace sar {

class ProgressReporter {
public:
    using Callback = std::function<void(const std::string& stage, Real percent, UInt64 processed, UInt64 total)>;

    ProgressReporter() = default;
    explicit ProgressReporter(Callback cb) : callback_(std::move(cb)) {}

    void set_callback(Callback cb) { callback_ = std::move(cb); }

    void start_stage(const std::string& name, UInt64 total_work) {
        stage_name_ = name;
        total_      = total_work;
        processed_.store(0, std::memory_order_relaxed);
        start_time_ = std::chrono::high_resolution_clock::now();
        report(0);
    }

    void update(UInt64 delta = 1) {
        const UInt64 p = processed_.fetch_add(delta, std::memory_order_relaxed) + delta;
        const Real pct = total_ > 0 ? (static_cast<Real>(p) / static_cast<Real>(total_)) : 1.0;
        if (callback_ && (p % report_interval_ == 0 || p == total_)) {
            callback_(stage_name_, pct, p, total_);
        }
    }

    void finish_stage() {
        processed_.store(total_, std::memory_order_relaxed);
        report(1.0);
    }

    void set_report_interval(UInt64 interval) { report_interval_ = interval; }

    Real elapsed_seconds() const {
        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<Real>(now - start_time_).count();
    }

private:
    void report(Real pct) const {
        if (callback_) {
            callback_(stage_name_, pct, processed_.load(std::memory_order_relaxed), total_);
        }
    }

    Callback  callback_;
    std::string stage_name_ = "Idle";
    UInt64    total_        = 0;
    std::atomic<UInt64> processed_{0};
    UInt64    report_interval_ = 128;
    std::chrono::high_resolution_clock::time_point start_time_;
};

}
