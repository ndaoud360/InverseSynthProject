#include "Pipeline.h"
#include "Effects.h"
#include <algorithm>
#include <cmath>

namespace invsynth {

void conditionInput(std::vector<float>& x) {
    float pk = 0.f;
    for (float v : x) pk = std::max(pk, std::abs(v));
    if (pk > 0.f) {
        const float thr = pk * 1e-3f;
        size_t first = 0;
        while (first < x.size() && std::abs(x[first]) <= thr) ++first;
        if (first < x.size()) x.erase(x.begin(), x.begin() + (long)(first > 32 ? first - 32 : 0));
    }
    const size_t minLen = (size_t)(0.25 * kSR);
    if (x.size() < minLen) x.resize(minLen, 0.f);
}

AnalysisResult analyse(std::vector<float> x, Predictor& predictor, const AnalysisOptions& opt,
                       const ProgressFn& progress, const std::atomic<bool>* cancel) {
    AnalysisResult r;
    if (x.empty()) { r.message = "empty audio"; return r; }
    conditionInput(x);
    r.durationSec = (double)x.size() / kSR;

    auto in = prepareInput(x);          // x is now peak-normalised
    r.window = in.window;

    std::string err;
    if (!predictor.predict(in.mel, in.ctx, r.stage1, err)) { r.message = "predictor: " + err; return r; }
    r.params = r.stage1;

    if (opt.runFinetune && !(cancel && cancel->load())) {
        auto ft = finetune(x, r.stage1, in.window, opt.finetune, progress, cancel);
        r.params = ft.params;
        r.loss = ft.loss;
        if (ft.cancelled) r.message = "cancelled - kept best result so far";
    }

    // loudness: render 1 s at the root and aim for -6 dBFS peak in the plugin
    auto y = renderNote(r.params, kSR, gateSamples(r.params, kSR), kSR, false);
    float pk = 0.f;
    for (float v : y) pk = std::max(pk, std::abs(v));
    r.suggestedGain = pk > 1e-6f ? std::clamp(0.5f / pk, 0.01f, 20.f) : 1.f;
    r.ok = true;
    if (r.message.empty()) r.message = "done";
    return r;
}

} // namespace invsynth
