#pragma once
// Stage-1 predictor on ONNX Runtime (CPU EP).
//
// The ORT shared library is loaded at RUNTIME from an explicit path (the one
// bundled inside the plugin) instead of being linked:
//   * Windows ships its own onnxruntime.dll in System32; a normal link can
//     silently bind to that older copy and crash. Loading by full path avoids it.
//   * The VST3/AU validators (and JUCE's moduleinfo step) can load the plugin
//     even when the library is missing: the plugin still loads, analysis is
//     simply disabled with a clear message.
#include "Features.h"
#include "Params.h"
#include <memory>
#include <string>

namespace invsynth {

class Predictor {
public:
    Predictor();
    ~Predictor();
    // Load libonnxruntime.{dylib,so} / onnxruntime.dll. Idempotent, process-wide.
    static bool loadRuntime(const std::string& libraryPath, std::string& error);
    static bool runtimeLoaded();

    bool loadModel(const std::string& onnxPath, std::string& error, int threads = 2);
    bool loadModelFromMemory(const void* data, size_t size, std::string& error, int threads = 2);
    bool isReady() const;

    // mel: network features (1,128,T) with any T >= 1. Thread-safe per instance
    // as long as calls are not concurrent.
    bool predict(const Spectrogram& mel, const std::array<float, 3>& ctx,
                 ParamVector& out, std::string& error);

private:
    bool finishLoad(std::string& error);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace invsynth
