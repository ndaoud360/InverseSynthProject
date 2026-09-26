#include "Features.h"
#include "Fft.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace invsynth {

// ------------------------------------------------------------ Slaney mel
static double hzToMel(double f) {
    const double fsp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fsp;
    const double logstep = std::log(6.4) / 27.0;
    return f >= minLogHz ? minLogMel + std::log(f / minLogHz) / logstep : f / fsp;
}
static double melToHz(double m) {
    const double fsp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fsp;
    const double logstep = std::log(6.4) / 27.0;
    return m >= minLogMel ? minLogHz * std::exp(logstep * (m - minLogMel)) : fsp * m;
}

const std::vector<float>& melBasis() {
    static const std::vector<float> basis = [] {
        const int nb = kNfft / 2 + 1;
        std::vector<double> fftf((size_t)nb), melf((size_t)kMels + 2);
        for (int i = 0; i < nb; ++i) fftf[(size_t)i] = (kSR / 2.0) * i / (nb - 1);
        const double m0 = hzToMel(kFmin), m1 = hzToMel(kFmax);
        for (int i = 0; i < kMels + 2; ++i) melf[(size_t)i] = melToHz(m0 + (m1 - m0) * i / (kMels + 1));
        std::vector<float> w((size_t)kMels * nb, 0.f);
        for (int m = 0; m < kMels; ++m) {
            const double enorm = 2.0 / (melf[(size_t)m + 2] - melf[(size_t)m]);
            const double d0 = melf[(size_t)m + 1] - melf[(size_t)m];
            const double d1 = melf[(size_t)m + 2] - melf[(size_t)m + 1];
            for (int k = 0; k < nb; ++k) {
                const double lower = -(melf[(size_t)m] - fftf[(size_t)k]) / d0;
                const double upper = (melf[(size_t)m + 2] - fftf[(size_t)k]) / d1;
                w[(size_t)m * nb + k] = (float)(std::max(0.0, std::min(lower, upper)) * enorm);
            }
        }
        return w;
    }();
    return basis;
}

// ------------------------------------------------------------ spectrogram
Spectrogram melDb(const float* x, int n) {
    const int pad = kNfft / 2, nb = kNfft / 2 + 1;
    const int T = 1 + n / kHop;   // == 1 + (n + 2*pad - nfft) / hop
    Spectrogram s;
    s.T = T;
    s.v.assign((size_t)kMels * T, 0.f);
    RealFft fft(kNfft);
    std::vector<float> frame((size_t)kNfft), pw((size_t)nb);
    const auto& B = melBasis();
    for (int t = 0; t < T; ++t) {
        const int start = t * kHop - pad;
        for (int i = 0; i < kNfft; ++i) {
            const int idx = start + i;
            frame[(size_t)i] = (idx >= 0 && idx < n) ? x[idx] : 0.f;
        }
        fft.spectrum(frame.data(), pw.data(), true);
        for (int m = 0; m < kMels; ++m) {
            const float* row = &B[(size_t)m * nb];
            // mel filters are sparse; a production build should store [lo,hi) per row
            double acc = 0.0;
            for (int k = 0; k < nb; ++k) acc += (double)row[k] * pw[(size_t)k];
            s.at(m, t) = (float)(10.0 * std::log10(std::max(acc, 1e-10)));
        }
    }
    return s;
}

void dbToFeature(Spectrogram& s) {
    for (float& v : s.v) v = (float)((std::max((double)v, kDbFloor) + 50.0) / 50.0);
}

std::vector<float> onsetEnvelope(const Spectrogram& db) {
    std::vector<float> env((size_t)db.T, 0.f);
    for (int t = 0; t < db.T; ++t) {
        double acc = 0.0;
        for (int m = 0; m < db.nMels; ++m) {
            const double cur = std::max((double)db.at(m, t), kDbFloor);
            const double prev = t > 0 ? std::max((double)db.at(m, t - 1), kDbFloor) : kDbFloor;
            acc += std::max(cur - prev, 0.0);
        }
        env[(size_t)t] = (float)(acc / db.nMels);
    }
    return env;
}

static void zscore(std::vector<double>& v) {
    const double mu = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double a : v) var += (a - mu) * (a - mu);
    const double sd = std::sqrt(var / v.size());
    for (double& a : v) a = std::clamp((a - mu) / (sd + 1e-8), -3.0, 3.0);   // robust
}

std::vector<int> pickOnsets(const std::vector<float>& env, double k, int minGap) {
    const size_t T = env.size();
    double mu = 0, var = 0;
    for (float e : env) mu += e;
    mu /= T;
    for (float e : env) var += (e - mu) * (e - mu);
    const double thr = mu + k * std::sqrt(var / T);
    std::vector<int> peaks;
    long last = -1000000000L;
    for (size_t t = 0; t < T; ++t) {
        const double l = t > 0 ? env[t - 1] : -INFINITY;
        const double r = t + 1 < T ? env[t + 1] : -INFINITY;
        if (env[t] >= thr && env[t] >= l && env[t] > r && (long)t - last >= minGap) {
            peaks.push_back((int)t);
            last = (long)t;
        }
    }
    return peaks;
}

std::vector<float> densityCurve(const Spectrogram& db) {
    const int T = db.T, M = db.nMels;
    auto on = onsetEnvelope(db);
    std::vector<double> flux(on.begin(), on.end()), energy((size_t)T), ent((size_t)T);
    std::vector<double> p((size_t)M);
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        for (int m = 0; m < M; ++m) {
            const double d = std::max((double)db.at(m, t), kDbFloor);
            p[(size_t)m] = std::pow(10.0, d / 10.0);
            sum += p[(size_t)m];
        }
        energy[(size_t)t] = 10.0 * std::log10(sum + 1e-10);   // frame loudness (dB)
        double h = 0.0;
        for (int m = 0; m < M; ++m) {
            const double q = p[(size_t)m] / (sum + 1e-12);
            h -= q * std::log(q + 1e-12);
        }
        ent[(size_t)t] = h / std::log((double)M);
    }
    zscore(flux); zscore(energy); zscore(ent);
    std::vector<float> d((size_t)T);
    for (int t = 0; t < T; ++t) d[(size_t)t] = (float)(flux[(size_t)t] + energy[(size_t)t] + 0.5 * ent[(size_t)t]);
    return d;
}

std::array<float, 3> AnalysisWindow::ctx() const {
    const double totalS = (double)total / kSR;
    return {(float)((double)start / std::max(total, 1)),
            (float)(((double)length / kSR) / kWindowSec),
            (float)(std::min(totalS, 60.0) / 60.0)};
}

AnalysisWindow selectWindow(const std::vector<float>& x, double windowSec, double snapSec) {
    const int n = (int)x.size();
    const int wlen = (int)(windowSec * kSR);
    if (n <= wlen) return {0, n, n};
    const auto db = melDb(x.data(), n);
    const auto dens = densityCurve(db);
    const int wf = (int)std::lround(windowSec * kFramesPerSec);
    std::vector<double> cs((size_t)db.T + 1, 0.0);
    for (int t = 0; t < db.T; ++t) cs[(size_t)t + 1] = cs[(size_t)t] + dens[(size_t)t];
    int best = 0;
    double bestScore = -INFINITY;
    for (int s = 0; s + wf <= db.T; ++s) {
        const double sc = cs[(size_t)(s + wf)] - cs[(size_t)s];
        if (sc > bestScore) { bestScore = sc; best = s; }
    }
    const auto onsets = pickOnsets(onsetEnvelope(db));
    const int snap = (int)(snapSec * kFramesPerSec);
    for (int o : onsets)
        if (o <= best && o >= best - snap) { best = o; break; }
    return {std::min(best * kHop, n - wlen), wlen, n};
}

void peakNormalizeInPlace(float* x, int n, float target) {
    float pk = 0.f;
    for (int i = 0; i < n; ++i) pk = std::max(pk, std::abs(x[i]));
    if (pk < 1e-6f) return;
    const float g = target / pk;
    for (int i = 0; i < n; ++i) x[i] *= g;
}

PreparedInput prepareInput(std::vector<float>& x, double windowSec) {
    peakNormalizeInPlace(x.data(), (int)x.size());
    PreparedInput r;
    r.window = selectWindow(x, windowSec);
    std::vector<float> seg(x.begin() + r.window.start, x.begin() + r.window.start + r.window.length);
    peakNormalizeInPlace(seg.data(), (int)seg.size());
    r.mel = melDb(seg.data(), (int)seg.size());
    dbToFeature(r.mel);
    r.ctx = r.window.ctx();
    return r;
}

} // namespace invsynth
