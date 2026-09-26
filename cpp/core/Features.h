#pragma once
// Mirror of python/invsynth/features.py (log-mel, onset flux, density-based
// 5-second window selection). Input audio must already be mono @ kSR.
#include <array>
#include <vector>

namespace invsynth {

constexpr int kSR = 22050;
constexpr int kNfft = 1024;
constexpr int kHop = 256;
constexpr int kMels = 128;
constexpr double kFmin = 20.0;
constexpr double kFmax = kSR / 2.0;
constexpr double kWindowSec = 5.0;
constexpr double kDbFloor = -100.0;
constexpr double kFramesPerSec = (double)kSR / kHop;

struct Spectrogram {
    int nMels = kMels, T = 0;
    std::vector<float> v;            // row-major [mel * T + t]  == ONNX (1,128,T)
    float& at(int m, int t) { return v[(size_t)m * T + t]; }
    float at(int m, int t) const { return v[(size_t)m * T + t]; }
};

struct AnalysisWindow {
    int start = 0, length = 0, total = 0;          // samples
    std::array<float, 3> ctx() const;
};

const std::vector<float>& melBasis();              // kMels x (kNfft/2+1)

Spectrogram melDb(const float* x, int n);          // dB values
void dbToFeature(Spectrogram& s);                   // in place -> network range
std::vector<float> onsetEnvelope(const Spectrogram& db);
std::vector<int> pickOnsets(const std::vector<float>& env, double k = 1.5, int minGap = 4);
std::vector<float> densityCurve(const Spectrogram& db);
AnalysisWindow selectWindow(const std::vector<float>& x, double windowSec = kWindowSec,
                            double snapSec = 2.0);
void peakNormalizeInPlace(float* x, int n, float target = 0.9f);  // no-op on silence

struct PreparedInput {
    Spectrogram mel;                 // network features (1,128,T)
    std::array<float, 3> ctx{};
    AnalysisWindow window;
};
// x is modified (peak-normalised) in place.
PreparedInput prepareInput(std::vector<float>& x, double windowSec = kWindowSec);

} // namespace invsynth
