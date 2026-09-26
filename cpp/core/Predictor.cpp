#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include "Predictor.h"
#include <algorithm>
#include <mutex>

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace invsynth {

namespace {
std::mutex gRuntimeMutex;
bool gRuntimeOk = false;

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w((size_t)std::max(n - 1, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}
#endif
} // namespace

bool Predictor::loadRuntime(const std::string& path, std::string& error) {
    std::lock_guard<std::mutex> lk(gRuntimeMutex);
    if (gRuntimeOk) return true;
    using GetApiBaseFn = const OrtApiBase* (*)();
    GetApiBaseFn fn = nullptr;
#ifdef _WIN32
    HMODULE h = LoadLibraryExW(widen(path).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) { error = "could not load " + path; return false; }
    fn = reinterpret_cast<GetApiBaseFn>(GetProcAddress(h, "OrtGetApiBase"));
#else
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { error = std::string("dlopen failed: ") + dlerror(); return false; }
    fn = reinterpret_cast<GetApiBaseFn>(dlsym(h, "OrtGetApiBase"));
#endif
    if (!fn) { error = "OrtGetApiBase not found in " + path; return false; }
    const OrtApi* api = fn()->GetApi(ORT_API_VERSION);
    if (!api) {
        error = "ONNX Runtime library is older than the headers (API " +
                std::to_string(ORT_API_VERSION) + ")";
        return false;
    }
    Ort::InitApi(api);          // hands the function table to the C++ wrapper
    gRuntimeOk = true;
    return true;
}

bool Predictor::runtimeLoaded() {
    std::lock_guard<std::mutex> lk(gRuntimeMutex);
    return gRuntimeOk;
}

struct Predictor::Impl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    // Every Ort object is heap-held: an Ort::* value, even a null one, calls
    // into the API on destruction, which crashes if the runtime never loaded.
    std::unique_ptr<Ort::MemoryInfo> mem;
    std::string inMel, inCtx, out;
};

Predictor::Predictor() : impl_(std::make_unique<Impl>()) {}
Predictor::~Predictor() = default;
bool Predictor::isReady() const { return impl_->session != nullptr; }

static Ort::SessionOptions makeOptions(int threads) {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(threads);          // leave cores for the DAW
    o.SetInterOpNumThreads(1);
    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    o.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    return o;
}

bool Predictor::loadModel(const std::string& path, std::string& error, int threads) {
    if (!runtimeLoaded()) { error = "ONNX Runtime not loaded"; return false; }
    try {
        if (!impl_->env) impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "invsynth");
#ifdef _WIN32
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, widen(path).c_str(), makeOptions(threads));
#else
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, path.c_str(), makeOptions(threads));
#endif
    } catch (const Ort::Exception& e) {
        error = e.what();
        impl_->session.reset();
        return false;
    }
    return finishLoad(error);
}

bool Predictor::loadModelFromMemory(const void* data, size_t size, std::string& error, int threads) {
    if (!runtimeLoaded()) { error = "ONNX Runtime not loaded"; return false; }
    try {
        if (!impl_->env) impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "invsynth");
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, data, size, makeOptions(threads));
    } catch (const Ort::Exception& e) {
        error = e.what();
        impl_->session.reset();
        return false;
    }
    return finishLoad(error);
}

bool Predictor::finishLoad(std::string& error) {
    try {
        Ort::AllocatorWithDefaultOptions alloc;
        auto& s = *impl_->session;
        if (s.GetInputCount() != 2 || s.GetOutputCount() != 1) {
            error = "unexpected model signature";
            impl_->session.reset();
            return false;
        }
        impl_->inMel = s.GetInputNameAllocated(0, alloc).get();
        impl_->inCtx = s.GetInputNameAllocated(1, alloc).get();
        impl_->out = s.GetOutputNameAllocated(0, alloc).get();
        // front-end contract check (written by export_onnx.py)
        auto meta = s.GetModelMetadata();
        auto cfg = meta.LookupCustomMetadataMapAllocated("invsynth_config", alloc);
        if (cfg) {
            std::string c = cfg.get();
            auto has = [&](const std::string& k) { return c.find(k) != std::string::npos; };
            if (!has("\"n_mels\": " + std::to_string(kMels)) || !has("\"hop\": " + std::to_string(kHop)) ||
                !has("\"sr\": " + std::to_string(kSR)) || !has("\"n_params\": " + std::to_string(kNumParams))) {
                error = "model feature config does not match this build: " + c;
                impl_->session.reset();
                return false;
            }
        }
        impl_->mem = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const Ort::Exception& e) {
        error = e.what();
        impl_->session.reset();
        return false;
    }
    return true;
}

bool Predictor::predict(const Spectrogram& mel, const std::array<float, 3>& ctx,
                        ParamVector& out, std::string& error) {
    if (!isReady()) { error = "model not loaded"; return false; }
    try {
        std::vector<float> melData = mel.v;             // ORT wants non-const
        std::array<float, 3> ctxData = ctx;
        const int64_t melShape[3] = {1, mel.nMels, mel.T};
        const int64_t ctxShape[2] = {1, 3};
        Ort::Value inputs[2] = {
            Ort::Value::CreateTensor<float>(*impl_->mem, melData.data(), melData.size(), melShape, 3),
            Ort::Value::CreateTensor<float>(*impl_->mem, ctxData.data(), 3, ctxShape, 2)};
        const char* inNames[2] = {impl_->inMel.c_str(), impl_->inCtx.c_str()};
        const char* outNames[1] = {impl_->out.c_str()};
        auto res = impl_->session->Run(Ort::RunOptions{nullptr}, inNames, inputs, 2, outNames, 1);
        const float* o = res[0].GetTensorData<float>();
        if (res[0].GetTensorTypeAndShapeInfo().GetElementCount() != (size_t)kNumParams) {
            error = "model output size mismatch";
            return false;
        }
        for (int i = 0; i < kNumParams; ++i) out[(size_t)i] = std::clamp(o[i], 0.f, 1.f);
    } catch (const Ort::Exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

} // namespace invsynth
