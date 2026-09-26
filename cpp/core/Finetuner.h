#pragma once
// Stage-2 NSGA-II finetuner - mirror of python/invsynth/finetune.py.
// Runs on a background thread; evaluations are spread over worker threads.
#include "Features.h"
#include "Params.h"
#include <atomic>
#include <functional>
#include <vector>

namespace invsynth {

enum class WindowMode { Selected, Full, Progressive };

struct FinetuneConfig {
    int population = 32;
    int generations = 45;
    double tolerance = 0.02;
    int patience = 12;
    WindowMode mode = WindowMode::Progressive;
    double maxCompareSec = 30.0;   // bounds cost for very long files
    int threads = 0;               // 0 = hardware_concurrency - 1
    uint64_t seed = 1;
};

struct FinetuneResult {
    ParamVector params{};
    double loss = 1e9;
    int generationsRun = 0;
    bool cancelled = false;
};

using ProgressFn = std::function<void(int gen, int total, double bestLoss, double windowSec)>;

// x: full target at kSR, peak-normalised. p0: stage-1 guess. w: analysis window.
FinetuneResult finetune(const std::vector<float>& x, const ParamVector& p0,
                        const AnalysisWindow& w, const FinetuneConfig& cfg,
                        const ProgressFn& progress = {},
                        const std::atomic<bool>* cancel = nullptr);

// Exposed for tests / the plugin's "fit envelope only" button.
ParamVector matchDuration(const ParamVector& p, const float* note, int n);

} // namespace invsynth
