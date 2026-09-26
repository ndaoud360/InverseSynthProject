#pragma once
// Orchestrates the whole offline analysis (runs on a background thread):
//
//   mono @22.05k ─> trim/pad ─> peak-normalise ─> select 5 s window
//      ─> log-mel (1,128,T) + ctx ─> ONNX predictor (stage 1)
//      ─> NSGA-II finetuner over selected / growing / full region (stage 2)
//      ─> parameter vector + suggested output gain
#include "Finetuner.h"
#include "Predictor.h"
#include <atomic>
#include <string>

namespace invsynth {

struct AnalysisResult {
    bool ok = false;
    std::string message;
    ParamVector stage1{}, params{};
    double loss = 0.0;
    AnalysisWindow window;
    double durationSec = 0.0;
    float suggestedGain = 1.f;       // makes a C4 (root) note peak around -6 dBFS
};

struct AnalysisOptions {
    bool runFinetune = true;
    FinetuneConfig finetune;
};

void conditionInput(std::vector<float>& x);   // trim leading silence, min length

AnalysisResult analyse(std::vector<float> mono22k, Predictor& predictor,
                       const AnalysisOptions& opt, const ProgressFn& progress = {},
                       const std::atomic<bool>* cancel = nullptr);

} // namespace invsynth
