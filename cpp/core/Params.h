#pragma once
// Parameter table - MUST mirror python/invsynth/params.py and the mapping
// functions in python/invsynth/engine.py (tests/test_parity.py checks this).
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace invsynth {

constexpr int kNumParams = 41;
using ParamVector = std::array<float, kNumParams>;

enum ParamId {
    kOsc1Wave, kOsc1WtPos, kOsc1Level, kOsc1Coarse, kOsc1Fine,
    kOsc2Wave, kOsc2WtPos, kOsc2Level, kOsc2Coarse, kOsc2Fine,
    kFmAmount, kRingMix, kSync,
    kPitch,
    kFiltType, kFiltCutoff, kFiltRes, kFiltEnvAmt,
    kFEnvA, kFEnvD, kFEnvS, kFEnvR,
    kAEnvA, kAEnvD, kAEnvS, kAEnvR,
    kGate,
    kLfoRate, kLfoShape, kLfoToPitch, kLfoToCutoff, kLfoToAmp, kLfoToWt,
    kDistDrive, kDistMix, kChorusRate, kChorusDepth, kChorusMix,
    kRevSize, kRevDamp, kRevMix
};

struct ParamInfo {
    const char* id;
    const char* label;
    int numClasses;      // 0 = continuous
    float defaultValue;
    const char* section;
};

inline const std::array<ParamInfo, kNumParams>& paramTable() {
    static const std::array<ParamInfo, kNumParams> t = {{
        {"osc1_wave", "Wave", 6, 1.f / 12 + 1.f / 6, "Osc 1"},
        {"osc1_wtpos", "WT Pos", 0, 0.f, "Osc 1"},
        {"osc1_level", "Level", 0, 0.8f, "Osc 1"},
        {"osc1_coarse", "Coarse", 0, 0.5f, "Osc 1"},
        {"osc1_fine", "Fine", 0, 0.5f, "Osc 1"},
        {"osc2_wave", "Wave", 6, 1.f / 12, "Osc 2"},
        {"osc2_wtpos", "WT Pos", 0, 0.f, "Osc 2"},
        {"osc2_level", "Level", 0, 0.f, "Osc 2"},
        {"osc2_coarse", "Coarse", 0, 0.5f, "Osc 2"},
        {"osc2_fine", "Fine", 0, 0.5f, "Osc 2"},
        {"fm_amount", "FM", 0, 0.f, "Mod"},
        {"ring_mix", "Ring", 0, 0.f, "Mod"},
        {"sync", "Sync", 2, 0.25f, "Mod"},
        {"pitch", "Root", 0, 0.5f, "Mod"},
        {"filt_type", "Type", 3, 1.f / 6, "Filter"},
        {"filt_cutoff", "Cutoff", 0, 0.8f, "Filter"},
        {"filt_res", "Reso", 0, 0.1f, "Filter"},
        {"filt_env_amt", "Env Amt", 0, 0.5f, "Filter"},
        {"fenv_attack", "A", 0, 0.1f, "Filter Env"},
        {"fenv_decay", "D", 0, 0.5f, "Filter Env"},
        {"fenv_sustain", "S", 0, 0.5f, "Filter Env"},
        {"fenv_release", "R", 0, 0.5f, "Filter Env"},
        {"aenv_attack", "A", 0, 0.1f, "Amp Env"},
        {"aenv_decay", "D", 0, 0.5f, "Amp Env"},
        {"aenv_sustain", "S", 0, 0.8f, "Amp Env"},
        {"aenv_release", "R", 0, 0.4f, "Amp Env"},
        {"gate", "Gate", 0, 0.8f, "Amp Env"},
        {"lfo_rate", "Rate", 0, 0.5f, "LFO"},
        {"lfo_shape", "Shape", 5, 0.1f, "LFO"},
        {"lfo_to_pitch", "> Pitch", 0, 0.f, "LFO"},
        {"lfo_to_cutoff", "> Cutoff", 0, 0.f, "LFO"},
        {"lfo_to_amp", "> Amp", 0, 0.f, "LFO"},
        {"lfo_to_wtpos", "> WT", 0, 0.f, "LFO"},
        {"dist_drive", "Drive", 0, 0.3f, "FX"},
        {"dist_mix", "Dist Mix", 0, 0.f, "FX"},
        {"chorus_rate", "Ch Rate", 0, 0.3f, "FX"},
        {"chorus_depth", "Ch Depth", 0, 0.5f, "FX"},
        {"chorus_mix", "Ch Mix", 0, 0.f, "FX"},
        {"rev_size", "Rv Size", 0, 0.5f, "FX"},
        {"rev_damp", "Rv Damp", 0, 0.5f, "FX"},
        {"rev_mix", "Rv Mix", 0, 0.f, "FX"},
    }};
    return t;
}

inline ParamVector defaultParams() {
    ParamVector p{};
    for (int i = 0; i < kNumParams; ++i) p[i] = paramTable()[i].defaultValue;
    return p;
}

// ---------------------------------------------------------------- mappings
inline int toClass(double v, int n) {
    int k = (int)(v * n);
    return k > n - 1 ? n - 1 : (k < 0 ? 0 : k);
}
inline double envTime(double v) { return 0.001 * std::pow(10000.0, v); }       // 1ms..10s
inline double semis(double v) { return std::floor(-24.0 + 48.0 * v + 0.5); }
inline double cents(double v) { return -50.0 + 100.0 * v; }
inline double rootMidi(double v) { return 24.0 + 72.0 * v; }
inline double rootHz(double v) { return 440.0 * std::pow(2.0, (rootMidi(v) - 69.0) / 12.0); }
inline double cutoffHz(double v) { return 20.0 * std::pow(1000.0, v); }
inline double lfoHz(double v) { return 0.05 * std::pow(400.0, v); }
inline double chorusHz(double v) { return 0.1 * std::pow(50.0, v); }
inline double gateFraction(double v) { return 0.05 + 0.95 * v; }

// ----------------------------------------------------------- display text
inline std::string paramToText(int idx, float v) {
    static const char* waves[] = {"Sine", "Saw", "Square", "Triangle", "Noise", "Wavetable"};
    static const char* ftypes[] = {"Lowpass", "Bandpass", "Highpass"};
    static const char* lfos[] = {"Sine", "Triangle", "Saw", "Square", "S&H"};
    static const char* notes[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    char b[64];
    switch (idx) {
        case kOsc1Wave: case kOsc2Wave: return waves[toClass(v, 6)];
        case kFiltType: return ftypes[toClass(v, 3)];
        case kLfoShape: return lfos[toClass(v, 5)];
        case kSync: return toClass(v, 2) ? "On" : "Off";
        case kOsc1Coarse: case kOsc2Coarse: std::snprintf(b, 64, "%+d st", (int)semis(v)); return b;
        case kOsc1Fine: case kOsc2Fine: std::snprintf(b, 64, "%+.0f ct", cents(v)); return b;
        case kPitch: {
            double m = rootMidi(v); int n = (int)std::floor(m + 0.5);
            std::snprintf(b, 64, "%s%d %+.0fct", notes[((n % 12) + 12) % 12], n / 12 - 1, (m - n) * 100);
            return b;
        }
        case kFiltCutoff: std::snprintf(b, 64, "%.0f Hz", cutoffHz(v)); return b;
        case kFiltEnvAmt: std::snprintf(b, 64, "%+.1f oct", (2.0 * v - 1.0) * 6.0); return b;
        case kFEnvA: case kFEnvD: case kFEnvR: case kAEnvA: case kAEnvD: case kAEnvR: {
            double t = envTime(v);
            if (t < 1.0) std::snprintf(b, 64, "%.0f ms", t * 1000); else std::snprintf(b, 64, "%.2f s", t);
            return b;
        }
        case kLfoRate: std::snprintf(b, 64, "%.2f Hz", lfoHz(v)); return b;
        case kChorusRate: std::snprintf(b, 64, "%.2f Hz", chorusHz(v)); return b;
        default: std::snprintf(b, 64, "%.0f%%", v * 100.0); return b;
    }
}

} // namespace invsynth
