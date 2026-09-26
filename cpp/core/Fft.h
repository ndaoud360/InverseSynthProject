#pragma once
// Small iterative radix-2 FFT (power-of-two sizes). Dependency-free so the
// core compiles without JUCE; swap for juce::dsp::FFT / pffft / Accelerate for
// extra speed - the finetuner's loss is the hot path.
#include <cmath>
#include <complex>
#include <vector>

namespace invsynth {

class RealFft {
public:
    explicit RealFft(int n) : n_(n), tw_((size_t)n / 2), rev_((size_t)n), buf_((size_t)n) {
        int bits = 0;
        while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
            rev_[(size_t)i] = r;
        }
        for (int i = 0; i < n / 2; ++i)
            tw_[(size_t)i] = std::polar(1.0, -2.0 * 3.14159265358979323846 * i / n);
        win_.resize((size_t)n);
        for (int i = 0; i < n; ++i)   // periodic Hann (== np.hanning(n+1)[:-1])
            win_[(size_t)i] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / n);
    }
    int size() const { return n_; }

    // frame: n samples (unwindowed). Writes n/2+1 values of |X|^2 or |X|.
    void spectrum(const float* frame, float* out, bool power) {
        for (int i = 0; i < n_; ++i)
            buf_[(size_t)rev_[(size_t)i]] = std::complex<double>(frame[i] * win_[(size_t)i], 0.0);
        for (int len = 2; len <= n_; len <<= 1) {
            const int half = len >> 1, step = n_ / len;
            for (int i = 0; i < n_; i += len)
                for (int j = 0; j < half; ++j) {
                    auto t = tw_[(size_t)(j * step)] * buf_[(size_t)(i + j + half)];
                    buf_[(size_t)(i + j + half)] = buf_[(size_t)(i + j)] - t;
                    buf_[(size_t)(i + j)] += t;
                }
        }
        for (int k = 0; k <= n_ / 2; ++k) {
            const double p = std::norm(buf_[(size_t)k]);
            out[k] = (float)(power ? p : std::sqrt(p));
        }
    }

private:
    int n_;
    std::vector<std::complex<double>> tw_;
    std::vector<int> rev_;
    std::vector<std::complex<double>> buf_;
    std::vector<double> win_;
};

} // namespace invsynth
