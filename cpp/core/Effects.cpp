#include "Effects.h"
#include <algorithm>
#include <cmath>

namespace invsynth {

static constexpr double kPi = 3.14159265358979323846;
static const int kCombLens[4] = {1116, 1188, 1277, 1356};
static const int kApLens[2] = {556, 441};

void FxChain::prepare(double sampleRate) {
    sr_ = sampleRate;
    cbuf_.assign((size_t)(int)(0.03 * sr_) + 2, 0.0);
    const double scale = sr_ / 44100.0;
    for (int j = 0; j < 4; ++j) comb_[j].assign((size_t)std::max((int)(kCombLens[j] * scale), 1), 0.0);
    for (int j = 0; j < 2; ++j) ap_[j].assign((size_t)std::max((int)(kApLens[j] * scale), 1), 0.0);
    reset();
}

void FxChain::reset() {
    std::fill(cbuf_.begin(), cbuf_.end(), 0.0);
    for (auto& c : comb_) std::fill(c.begin(), c.end(), 0.0);
    for (auto& a : ap_) std::fill(a.begin(), a.end(), 0.0);
    cw_ = 0; cph_ = 0;
    std::fill(std::begin(cidx_), std::end(cidx_), 0);
    std::fill(std::begin(aidx_), std::end(aidx_), 0);
    std::fill(std::begin(cfilt_), std::end(cfilt_), 0.0);
}

void FxChain::setParams(const ParamVector& p) {
    drive_ = 1.0 + 29.0 * (double)p[kDistDrive] * p[kDistDrive];
    dmix_ = p[kDistMix];
    crate_ = chorusHz(p[kChorusRate]);
    cdepth_ = p[kChorusDepth];
    cmix_ = p[kChorusMix];
    fb_ = 0.7 + 0.28 * p[kRevSize];
    damp_ = p[kRevDamp] * 0.4;
    rmix_ = p[kRevMix];
}

void FxChain::process(float* x, int n) {
    if (dmix_ > 0.0)
        for (int i = 0; i < n; ++i)
            x[i] = (float)((1.0 - dmix_) * x[i] + dmix_ * std::tanh(drive_ * x[i]));

    if (cmix_ > 0.0) {
        const int blen = (int)cbuf_.size();
        for (int i = 0; i < n; ++i) {
            cbuf_[(size_t)cw_] = x[i];
            const double d = (0.007 + cdepth_ * 0.005 * (0.5 + 0.5 * std::sin(2.0 * kPi * cph_))) * sr_;
            cph_ += crate_ / sr_;
            if (cph_ >= 1.0) cph_ -= 1.0;
            double rp = cw_ - d;
            if (rp < 0.0) rp += blen;
            const int i0 = (int)rp;
            const double fr = rp - i0;
            const int i1 = i0 + 1 >= blen ? 0 : i0 + 1;
            const double dl = cbuf_[(size_t)i0] + fr * (cbuf_[(size_t)i1] - cbuf_[(size_t)i0]);
            x[i] = (float)((1.0 - cmix_) * x[i] + cmix_ * 0.5 * (x[i] + dl));
            if (++cw_ >= blen) cw_ = 0;
        }
    }

    if (rmix_ > 0.0) {
        for (int i = 0; i < n; ++i) {
            const double inp = x[i] * 0.03;
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) {
                auto& b = comb_[j];
                const double o = b[(size_t)cidx_[j]];
                cfilt_[j] = o * (1.0 - damp_) + cfilt_[j] * damp_;
                b[(size_t)cidx_[j]] = inp + cfilt_[j] * fb_;
                if (++cidx_[j] >= (int)b.size()) cidx_[j] = 0;
                acc += o;
            }
            for (int j = 0; j < 2; ++j) {
                auto& b = ap_[j];
                const double bo = b[(size_t)aidx_[j]];
                const double o = -acc + bo;
                b[(size_t)aidx_[j]] = acc + bo * 0.5;
                if (++aidx_[j] >= (int)b.size()) aidx_[j] = 0;
                acc = o;
            }
            x[i] = (float)((1.0 - rmix_) * x[i] + rmix_ * acc * 3.0);
        }
    }
}

int gateSamples(const ParamVector& p, int nFullNote) {
    return (int)(nFullNote * gateFraction(p[kGate]));
}

void peakNormalize(std::vector<float>& x, float target) {
    float pk = 0.f;
    for (float v : x) pk = std::max(pk, std::abs(v));
    if (pk < 1e-6f || !std::isfinite(pk)) { std::fill(x.begin(), x.end(), 0.f); return; }
    const float g = target / pk;
    for (float& v : x) v *= g;
}

std::vector<float> renderNote(const ParamVector& p, int n, int gate, double sr, bool normalize) {
    std::vector<float> out((size_t)std::max(n, 1), 0.f);
    VoiceSettings vs;
    vs.fromParams(p);
    SynthVoice v;
    v.autoKill = false;
    v.prepare(sr);
    v.setSettings(vs);
    v.noteOn(1.0, 1.f, gate);
    v.render(out.data(), (int)out.size());
    FxChain fx;
    fx.prepare(sr);
    fx.setParams(p);
    fx.process(out.data(), (int)out.size());
    if (normalize) peakNormalize(out);
    return out;
}

} // namespace invsynth
