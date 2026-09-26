#include "SynthVoice.h"
#include <algorithm>
#include <cmath>

namespace invsynth {

static constexpr double kPi = 3.14159265358979323846;

const std::vector<float>& wavetables() {
    static const std::vector<float> wt = [] {
        std::vector<double> t((size_t)kWtFrames * kWtSize, 0.0);
        for (int k = 0; k < kWtFrames; ++k) {
            double* f = &t[(size_t)k * kWtSize];
            for (int h = 1; h <= 64; ++h) {
                double a = std::pow((double)h, -(0.6 + 0.2 * k));
                if (h % 2 == 0) a *= 1.0 - (double)k / (kWtFrames - 1);
                a *= 1.0 + 0.5 * std::sin(h * k * 0.7);
                for (int n = 0; n < kWtSize; ++n)
                    f[n] += a * std::sin(2.0 * kPi * h * ((double)n / kWtSize));
            }
            double pk = 0;
            for (int n = 0; n < kWtSize; ++n) pk = std::max(pk, std::abs(f[n]));
            for (int n = 0; n < kWtSize; ++n) f[n] /= pk;
        }
        return std::vector<float>(t.begin(), t.end());
    }();
    return wt;
}

void VoiceSettings::fromParams(const ParamVector& p) {
    w1 = toClass(p[kOsc1Wave], 6);  wtp1 = p[kOsc1WtPos]; lvl1 = p[kOsc1Level];
    ratio1 = std::pow(2.0, (semis(p[kOsc1Coarse]) + cents(p[kOsc1Fine]) / 100.0) / 12.0);
    w2 = toClass(p[kOsc2Wave], 6);  wtp2 = p[kOsc2WtPos]; lvl2 = p[kOsc2Level];
    ratio2 = std::pow(2.0, (semis(p[kOsc2Coarse]) + cents(p[kOsc2Fine]) / 100.0) / 12.0);
    fm = (double)p[kFmAmount] * p[kFmAmount] * 2.0;
    ring = p[kRingMix];
    sync = toClass(p[kSync], 2);
    rootHz = invsynth::rootHz(p[kPitch]);
    ftype = toClass(p[kFiltType], 3);
    cutoff = cutoffHz(p[kFiltCutoff]);
    k = 2.0 - 1.96 * p[kFiltRes];
    envOct = (2.0 * p[kFiltEnvAmt] - 1.0) * 6.0;
    fA = envTime(p[kFEnvA]); fD = envTime(p[kFEnvD]); fS = p[kFEnvS]; fR = envTime(p[kFEnvR]);
    aA = envTime(p[kAEnvA]); aD = envTime(p[kAEnvD]); aS = p[kAEnvS]; aR = envTime(p[kAEnvR]);
    lfoHz = invsynth::lfoHz(p[kLfoRate]);
    lshape = toClass(p[kLfoShape], 5);
    lPitch = std::pow((double)p[kLfoToPitch], 3.0) * 12.0;
    lCut = (double)p[kLfoToCutoff] * p[kLfoToCutoff] * 3.0;
    lAmp = p[kLfoToAmp];
    lWt = p[kLfoToWt];
}

void SynthVoice::prepare(double sampleRate) {
    sr_ = sampleRate;
    setSettings(s_);
    active_ = false;
}

void SynthVoice::setSettings(const VoiceSettings& s) {
    s_ = s;
    fa_ = 1.0 / (s.fA * sr_); fd_ = std::exp(-5.0 / (s.fD * sr_)); fr_ = std::exp(-5.0 / (s.fR * sr_));
    aa_ = 1.0 / (s.aA * sr_); ad_ = std::exp(-5.0 / (s.aD * sr_)); ar_ = std::exp(-5.0 / (s.aR * sr_));
}

void SynthVoice::noteOn(double pitchRatio, float velocity, int64_t gateAt) {
    pitchRatio_ = pitchRatio;
    velocity_ = velocity;
    gateAt_ = gateAt;
    idx_ = 0;
    fStage_ = aStage_ = 0;
    fLev_ = aLev_ = 0.0;
    ph1_ = ph2_ = lph_ = 0.0;
    ic1_ = ic2_ = 0.0;
    rngNoise_.s = 12345;
    rngLfo_.s = 777;
    sh_ = rngLfo_.next();
    active_ = true;
}

void SynthVoice::noteOff() {
    if (active_) { fStage_ = 3; aStage_ = 3; }
}

static inline double blep(double t, double dt) {
    if (t < dt) { t /= dt; return t + t - t * t - 1.0; }
    if (t > 1.0 - dt) { t = (t - 1.0) / dt; return t * t + t + t + 1.0; }
    return 0.0;
}

static inline double osc(int wave, double t, double dt, double wtpos, double noise) {
    switch (wave) {
        case 0: return std::sin(2.0 * kPi * t);
        case 1: return 2.0 * t - 1.0 - blep(t, dt);
        case 2: {
            double t2 = t + 0.5; t2 -= std::floor(t2);
            return (t < 0.5 ? 1.0 : -1.0) + blep(t, dt) - blep(t2, dt);
        }
        case 3: return 1.0 - 4.0 * std::abs(t - 0.5);
        case 4: return noise;
        default: {
            const float* wt = wavetables().data();
            double x = wtpos * (kWtFrames - 1);
            int f0 = std::min((int)x, kWtFrames - 2);
            double fr = x - f0;
            double p = t * kWtSize;
            int i0 = std::min((int)p, kWtSize - 1);
            double pf = p - i0;
            int i1 = (i0 + 1) & (kWtSize - 1);
            const float* A = wt + (size_t)f0 * kWtSize;
            const float* B = A + kWtSize;
            double a = A[i0] + pf * (A[i1] - A[i0]);
            double b = B[i0] + pf * (B[i1] - B[i0]);
            return a + fr * (b - a);
        }
    }
}

static inline double lfoWave(int shape, double t, double sh) {
    switch (shape) {
        case 0: return std::sin(2.0 * kPi * t);
        case 1: return 1.0 - 4.0 * std::abs(t - 0.5);
        case 2: return 2.0 * t - 1.0;
        case 3: return t < 0.5 ? 1.0 : -1.0;
        default: return sh;
    }
}

static inline void envStep(int& stage, double& lev, double a, double d, double s, double r) {
    if (stage == 0) { lev += a; if (lev >= 1.0) { lev = 1.0; stage = 1; } }
    else if (stage == 1) lev = s + (lev - s) * d;
    else if (stage == 3) lev *= r;
}

void SynthVoice::render(float* out, int n) {
    if (!active_) return;
    const auto& s = s_;
    const double f1b = s.rootHz * pitchRatio_ * s.ratio1;
    const double f2b = s.rootHz * pitchRatio_ * s.ratio2;
    const double nyq = 0.45 * sr_;
    for (int j = 0; j < n; ++j, ++idx_) {
        if (idx_ == gateAt_) { fStage_ = 3; aStage_ = 3; }
        envStep(fStage_, fLev_, fa_, fd_, s.fS, fr_);
        envStep(aStage_, aLev_, aa_, ad_, s.aS, ar_);

        lph_ += s.lfoHz / sr_;
        if (lph_ >= 1.0) { lph_ -= 1.0; if (s.lshape == 4) sh_ = rngLfo_.next(); }
        const double l = lfoWave(s.lshape, lph_, sh_);

        const double pm = std::pow(2.0, s.lPitch * l / 12.0);
        const double inc1 = std::min(f1b * pm / sr_, 0.5);
        const double inc2 = std::min(f2b * pm / sr_, 0.5);
        const double wp1 = std::clamp(s.wtp1 + s.lWt * 0.5 * l, 0.0, 1.0);
        const double wp2 = std::clamp(s.wtp2 + s.lWt * 0.5 * l, 0.0, 1.0);
        const double n2 = s.w2 == 4 ? rngNoise_.next() : 0.0;
        const double n1 = s.w1 == 4 ? rngNoise_.next() : 0.0;

        ph2_ += inc2;
        bool wrapped = false;
        if (ph2_ >= 1.0) { ph2_ -= std::floor(ph2_); wrapped = true; }
        const double o2 = osc(s.w2, ph2_, inc2, wp2, n2);

        ph1_ += inc1;
        if (ph1_ >= 1.0) ph1_ -= std::floor(ph1_);
        if (s.sync == 1 && wrapped) { ph1_ = ph2_ * inc1 / inc2; ph1_ -= std::floor(ph1_); }
        double pmod = ph1_ + s.fm * o2;
        pmod -= std::floor(pmod);
        const double o1 = osc(s.w1, pmod, inc1, wp1, n1);

        const double x = (1.0 - s.ring) * (s.lvl1 * o1 + s.lvl2 * o2) + s.ring * o1 * o2;

        if (idx_ % kCtrlRate == 0) {
            double fc = s.cutoff * std::pow(2.0, s.envOct * fLev_ + s.lCut * l);
            fc = std::clamp(fc, 20.0, nyq);
            const double g = std::tan(kPi * fc / sr_);
            a1_ = 1.0 / (1.0 + g * (g + s.k));
            a2_ = g * a1_;
            a3_ = g * a2_;
        }
        const double v3 = x - ic2_;
        const double v1 = a1_ * ic1_ + a2_ * v3;
        const double v2 = ic2_ + a2_ * ic1_ + a3_ * v3;
        ic1_ = 2.0 * v1 - ic1_;
        ic2_ = 2.0 * v2 - ic2_;
        const double y = s.ftype == 0 ? v2 : (s.ftype == 1 ? v1 : x - s.k * v1 - v2);

        const double trem = 1.0 - s.lAmp * (0.5 - 0.5 * l);
        out[j] += (float)(y * aLev_ * trem * velocity_);

        if (autoKill && aStage_ == 3 && aLev_ < 1e-5) { active_ = false; return; }
    }
}

} // namespace invsynth
