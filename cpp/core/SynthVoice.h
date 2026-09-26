#pragma once
// Real-time safe synth voice. Line-by-line mirror of engine.render_voice().
// No allocation, no locks: safe to call from the audio thread.
#include "Params.h"
#include <cstdint>
#include <vector>

namespace invsynth {

constexpr int kWtFrames = 8;
constexpr int kWtSize = 2048;
constexpr int kCtrlRate = 16;

// Shared, immutable after first call (thread-safe static init).
const std::vector<float>& wavetables();   // kWtFrames * kWtSize

struct Lcg {
    uint64_t s = 0;
    double next() {                          // [-1, 1)
        s = (s * 1664525ull + 1013904223ull) & 0xFFFFFFFFull;
        return (double)s / 2147483648.0 - 1.0;
    }
};

// Values derived from the normalised vector once per block (not per sample).
struct VoiceSettings {
    int w1 = 1, w2 = 0, sync = 0, ftype = 0, lshape = 0;
    double wtp1 = 0, wtp2 = 0, lvl1 = 1, lvl2 = 0;
    double ratio1 = 1, ratio2 = 1;           // osc tuning ratio vs root
    double fm = 0, ring = 0;
    double rootHz = 261.6;
    double cutoff = 1000, k = 2, envOct = 0;
    double fA = 1, fD = 0.5, fS = 0.5, fR = 0.5;   // seconds / level
    double aA = 1, aD = 0.5, aS = 0.8, aR = 0.5;
    double lfoHz = 1, lPitch = 0, lCut = 0, lAmp = 0, lWt = 0;
    void fromParams(const ParamVector& p);
};

class SynthVoice {
public:
    void prepare(double sampleRate);
    void setSettings(const VoiceSettings& s);     // cheap; call per block
    // pitchRatio = 1 plays the root inferred from the sample.
    // gateAt >= 0 schedules note-off after that many samples (offline renders).
    void noteOn(double pitchRatio, float velocity = 1.f, int64_t gateAt = -1);
    void noteOff();
    void kill() { active_ = false; }
    bool isActive() const { return active_; }
    bool isReleasing() const { return aStage_ == 3; }
    int note = -1;                                 // MIDI note owning the voice
    uint64_t age = 0;
    bool autoKill = true;                          // false for offline renders

    // Adds (mixes) n samples into out.
    void render(float* out, int n);

private:
    void updateCoefs();
    double sr_ = 44100.0;
    VoiceSettings s_;
    bool active_ = false;
    double velocity_ = 1.0, pitchRatio_ = 1.0;
    int64_t idx_ = 0, gateAt_ = -1;
    // envelope coefficients
    double fa_ = 0, fd_ = 0, fr_ = 0, aa_ = 0, ad_ = 0, ar_ = 0;
    int fStage_ = 4, aStage_ = 4;
    double fLev_ = 0, aLev_ = 0;
    // oscillators / lfo
    double ph1_ = 0, ph2_ = 0, lph_ = 0, sh_ = 0;
    Lcg rngNoise_, rngLfo_;
    // SVF
    double ic1_ = 0, ic2_ = 0, a1_ = 0, a2_ = 0, a3_ = 0;
};

} // namespace invsynth
