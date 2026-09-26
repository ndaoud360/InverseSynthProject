#pragma once
// Mono FX chain: Distortion -> Chorus -> Reverb. Mirror of engine.render_fx().
// prepare() allocates; process() is allocation-free (real-time safe).
#include "Params.h"
#include "SynthVoice.h"
#include <vector>

namespace invsynth {

class FxChain {
public:
    void prepare(double sampleRate);
    void reset();
    void setParams(const ParamVector& p);
    void process(float* x, int n);

private:
    double sr_ = 44100.0;
    double drive_ = 1, dmix_ = 0, crate_ = 1, cdepth_ = 0, cmix_ = 0, fb_ = 0.8, damp_ = 0, rmix_ = 0;
    std::vector<double> cbuf_;
    int cw_ = 0;
    double cph_ = 0;
    std::vector<double> comb_[4], ap_[2];
    int cidx_[4] = {}, aidx_[2] = {};
    double cfilt_[4] = {};
};

// Offline render used by the finetuner and for auditioning (not real-time).
// Returns exactly n samples; note-off after `gate` samples.
std::vector<float> renderNote(const ParamVector& p, int n, int gate, double sr,
                              bool normalize = true);
int gateSamples(const ParamVector& p, int nFullNote);
void peakNormalize(std::vector<float>& x, float target = 0.9f);

} // namespace invsynth
