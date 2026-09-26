#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>

using namespace invsynth;

namespace {
#if JUCE_WINDOWS
const char* kOrtLibName = "onnxruntime.dll";
#elif JUCE_MAC
const char* kOrtLibName = "libonnxruntime.dylib";
#else
const char* kOrtLibName = "libonnxruntime.so";
#endif
const char* kModelName = "inverse_synth.onnx";

juce::File appDataDir() {
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("InverseSynth");
}
} // namespace

// =========================================================================
InverseSynthProcessor::InverseSynthProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createLayout()) {
    for (int i = 0; i < kNumParams; ++i)
        raw_[(size_t)i] = apvts.getRawParameterValue(paramTable()[(size_t)i].id);
    gainParam_ = apvts.getRawParameterValue("output_gain");
    formats_.registerBasicFormats();
    current_ = defaultParams();
    tryAutoLoadModel();
}

InverseSynthProcessor::~InverseSynthProcessor() {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
}

juce::AudioProcessorValueTreeState::ParameterLayout InverseSynthProcessor::createLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    for (int i = 0; i < kNumParams; ++i) {
        const auto& info = paramTable()[(size_t)i];
        auto attrs = juce::AudioParameterFloatAttributes().withStringFromValueFunction(
            [i](float v, int) { return juce::String(paramToText(i, v)); });
        layout.add(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{info.id, 1}, juce::String(info.section) + " " + info.label,
            juce::NormalisableRange<float>(0.f, 1.f), info.defaultValue, attrs));
    }
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"output_gain", 1}, "Output Gain",
        juce::NormalisableRange<float>(-30.f, 30.f, 0.1f), 0.f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    return layout;
}

bool InverseSynthProcessor::isBusesLayoutSupported(const BusesLayout& l) const {
    const auto out = l.getMainOutputChannelSet();
    return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo())
           && l.getMainInputChannelSet().isDisabled();
}

// =========================================================================
// Audio thread
// =========================================================================
void InverseSynthProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    for (auto& v : voices_) v.prepare(sampleRate);
    fx_.prepare(sampleRate);
    mono_.assign((size_t)std::max(samplesPerBlock, 4096), 0.f);
    paramsDirty_ = true;
}

void InverseSynthProcessor::handleNoteOn(int note, float vel) {
    SynthVoice* pick = nullptr;
    for (auto& v : voices_) if (!v.isActive()) { pick = &v; break; }
    if (!pick)       // steal: oldest releasing voice, else oldest voice
        for (int pass = 0; pass < 2 && !pick; ++pass)
            for (auto& v : voices_)
                if ((pass == 1 || v.isReleasing()) && (!pick || v.age < pick->age)) pick = &v;
    pick->note = note;
    pick->age = ++noteCounter_;
    pick->noteOn(std::pow(2.0, (note - 60) / 12.0), vel);   // MIDI 60 = detected root
}

void InverseSynthProcessor::handleNoteOff(int note) {
    for (auto& v : voices_)
        if (v.isActive() && v.note == note && !v.isReleasing()) v.noteOff();
}

void InverseSynthProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    keyboardState.processNextMidiBuffer(midi, 0, n, true);

    // 41 atomic loads per block; settings are only rebuilt when something moved
    bool changed = paramsDirty_;
    for (int i = 0; i < kNumParams; ++i) {
        const float v = raw_[(size_t)i]->load(std::memory_order_relaxed);
        if (v != current_[(size_t)i]) { current_[(size_t)i] = v; changed = true; }
    }
    if (changed) {
        VoiceSettings s;
        s.fromParams(current_);
        for (auto& v : voices_) v.setSettings(s);
        fx_.setParams(current_);
        paramsDirty_ = false;
    }

    if ((int)mono_.size() < n) mono_.resize((size_t)n);   // only if host exceeds its promise
    std::fill(mono_.begin(), mono_.begin() + n, 0.f);

    auto renderRange = [&](int from, int to) {
        if (to <= from) return;
        for (auto& v : voices_) v.render(mono_.data() + from, to - from);
    };
    int pos = 0;
    for (const auto meta : midi) {
        const int t = juce::jlimit(0, n, meta.samplePosition);
        renderRange(pos, t);
        pos = t;
        const auto m = meta.getMessage();
        if (m.isNoteOn()) handleNoteOn(m.getNoteNumber(), m.getFloatVelocity());
        else if (m.isNoteOff()) handleNoteOff(m.getNoteNumber());
        else if (m.isAllNotesOff() || m.isAllSoundOff())
            for (auto& v : voices_) v.noteOff();
    }
    renderRange(pos, n);

    fx_.process(mono_.data(), n);
    const float gain = juce::Decibels::decibelsToGain(gainParam_->load());
    for (int i = 0; i < n; ++i) mono_[(size_t)i] *= gain;

    if (previewPlaying_.load()) {          // audition the original sample
        juce::SpinLock::ScopedTryLockType lk(previewLock_);
        if (lk.isLocked() && preview_.getNumSamples() > 0) {
            const float* src = preview_.getReadPointer(0);
            const int avail = std::min(n, preview_.getNumSamples() - previewPos_);
            for (int i = 0; i < avail; ++i) mono_[(size_t)i] += src[previewPos_ + i];
            previewPos_ += avail;
            if (previewPos_ >= preview_.getNumSamples()) previewPlaying_ = false;
        }
    }

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        buffer.copyFrom(ch, 0, mono_.data(), n);
}

// =========================================================================
// Worker thread jobs
// =========================================================================
void InverseSynthProcessor::runJob(std::function<void()> job) {
    if (busy_.exchange(true)) return;
    if (worker_.joinable()) worker_.join();
    cancel_ = false;
    progress_ = 0.f;
    worker_ = std::thread([this, job = std::move(job)] {
        job();
        busy_ = false;
    });
}

void InverseSynthProcessor::setStatus(const juce::String& s) {
    const juce::ScopedLock lk(uiLock_);
    ui_.status = s;
}

InverseSynthProcessor::UiState InverseSynthProcessor::getUiState() const {
    const juce::ScopedLock lk(uiLock_);
    return ui_;
}

// Streams any-length file through a band-limited resampler: memory is bounded
// by the OUTPUT size (88 KB per second at 22.05 kHz), never the file size.
std::vector<float> InverseSynthProcessor::decode(const juce::File& f, double targetRate,
                                                 double maxSeconds, double& fileSeconds) {
    std::unique_ptr<juce::AudioFormatReader> reader(formats_.createReaderFor(f));
    if (!reader || reader->sampleRate <= 0) return {};
    fileSeconds = (double)reader->lengthInSamples / reader->sampleRate;
    const int ch = (int)std::max(1u, reader->numChannels);
    juce::AudioFormatReaderSource src(reader.get(), false);
    juce::ResamplingAudioSource rs(&src, false, ch);
    rs.setResamplingRatio(reader->sampleRate / targetRate);
    constexpr int kBlock = 8192;
    rs.prepareToPlay(kBlock, targetRate);
    const auto total = (size_t)(std::min(fileSeconds, maxSeconds) * targetRate);
    juce::AudioBuffer<float> buf(ch, kBlock);
    std::vector<float> out;
    out.reserve(total);
    while (out.size() < total && !cancel_.load()) {
        const int n = (int)std::min<size_t>(kBlock, total - out.size());
        juce::AudioSourceChannelInfo info(&buf, 0, n);
        rs.getNextAudioBlock(info);
        for (int i = 0; i < n; ++i) {
            float s = 0.f;
            for (int c = 0; c < ch; ++c) s += buf.getSample(c, i);
            out.push_back(s / (float)ch);
        }
    }
    rs.releaseResources();
    return out;
}

void InverseSynthProcessor::loadAudioFile(const juce::File& f) {
    const double hostRate = sampleRate_;
    runJob([this, f, hostRate] {
        setStatus("Loading " + f.getFileName() + " ...");
        double secs = 0, secs2 = 0;
        auto x = decode(f, kSR, 1e12, secs);                // full length for analysis
        if (x.empty()) { setStatus("Could not read " + f.getFileName()); return; }
        auto prev = decode(f, hostRate, 120.0, secs2);     // audition copy (capped)

        // window preview for the UI (the analysis job recomputes identically)
        auto y = x;
        conditionInput(y);
        peakNormalizeInPlace(y.data(), (int)y.size());
        const auto w = selectWindow(y);
        std::vector<float> ov(1024, 0.f);
        for (size_t i = 0; i < y.size(); ++i) {
            auto& b = ov[i * ov.size() / y.size()];
            b = std::max(b, std::abs(y[i]));
        }
        {
            juce::SpinLock::ScopedLockType lk(previewLock_);
            preview_.setSize(1, (int)prev.size());
            if (!prev.empty()) preview_.copyFrom(0, 0, prev.data(), (int)prev.size());
            peakNormalizeInPlace(preview_.getWritePointer(0), preview_.getNumSamples(), 0.5f);
            previewRate_ = hostRate;
            previewPos_ = 0;
        }
        const juce::ScopedLock lk(uiLock_);
        target22k_ = std::move(x);
        targetFile_ = f;
        ui_.fileName = f.getFileName();
        ui_.durationSec = secs;
        ui_.windowStartSec = (double)w.start / kSR;
        ui_.windowLenSec = (double)w.length / kSR;
        ui_.overview = std::move(ov);
        ui_.hasTarget = true;
        ui_.status = juce::String::formatted("Loaded %.2f s. Analysis window %.2f - %.2f s",
                                             secs, ui_.windowStartSec, ui_.windowStartSec + ui_.windowLenSec);
    });
}

void InverseSynthProcessor::startAnalysis(const AnalysisOptions& opt) {
    juce::WeakReference<InverseSynthProcessor> weak(this);   // created on message thread
    runJob([this, opt, weak] {
        std::vector<float> x;
        {
            const juce::ScopedLock lk(uiLock_);
            x = target22k_;
        }
        if (x.empty()) { setStatus("Load an audio file first"); return; }
        if (!predictor_.isReady()) { setStatus("No model loaded - use 'Load model'"); return; }
        setStatus("Stage 1: neural prediction ...");
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        auto r = analyse(std::move(x), predictor_, opt,
            [this](int g, int total, double loss, double winSec) {
                progress_ = (float)(g + 1) / (float)total;
                lastLoss_ = loss;
                setStatus(juce::String::formatted("Stage 2: generation %d/%d  loss %.4f  (comparing %.1f s)",
                                                  g + 1, total, loss, winSec));
            }, &cancel_);
        const double secs = (juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0;
        if (!r.ok) { setStatus("Analysis failed: " + juce::String(r.message)); return; }
        progress_ = 1.f;
        setStatus(juce::String::formatted("Match %s in %.1f s (loss %.4f). Play C4 to hear the root.",
                                          r.message.c_str(), secs, r.loss));
        juce::MessageManager::callAsync([weak, p = r.params, g = r.suggestedGain] {
            if (auto* self = weak.get()) self->applyParams(p, g);
        });
    });
}

void InverseSynthProcessor::applyParams(const ParamVector& p, float gain) {
    for (int i = 0; i < kNumParams; ++i) {
        auto* prm = apvts.getParameter(paramTable()[(size_t)i].id);
        prm->beginChangeGesture();
        prm->setValueNotifyingHost(p[(size_t)i]);
        prm->endChangeGesture();
    }
    auto* g = apvts.getParameter("output_gain");
    g->beginChangeGesture();
    g->setValueNotifyingHost(g->convertTo0to1(juce::jlimit(-30.f, 30.f, juce::Decibels::gainToDecibels(gain))));
    g->endChangeGesture();
}

void InverseSynthProcessor::setPreviewPlaying(bool on) {
    if (on) {
        juce::SpinLock::ScopedLockType lk(previewLock_);
        previewPos_ = 0;
    }
    previewPlaying_ = on;
}

// =========================================================================
// Model / runtime discovery
// =========================================================================
juce::File InverseSynthProcessor::pluginBinary() {
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile);
}

juce::File InverseSynthProcessor::findRuntimeLibrary() {
    const auto bin = pluginBinary();
    const juce::File candidates[] = {
        bin.getSiblingFile(kOrtLibName),                                        // Win / Linux bundle
        bin.getChildFile("Contents/Frameworks").getChildFile(kOrtLibName),      // mac bundle path
        bin.getParentDirectory().getSiblingFile("Frameworks").getChildFile(kOrtLibName), // mac binary path
        appDataDir().getChildFile(kOrtLibName),
    };
    for (const auto& c : candidates) if (c.existsAsFile()) return c;
    return {};
}

bool InverseSynthProcessor::loadModelFile(const juce::File& f) {
    std::string err;
    if (!Predictor::runtimeLoaded()) {
        const auto lib = findRuntimeLibrary();
        if (lib == juce::File() || !Predictor::loadRuntime(lib.getFullPathName().toStdString(), err)) {
            setStatus("ONNX Runtime missing (" + juce::String(kOrtLibName) + "): " + juce::String(err));
            return false;
        }
    }
    if (!predictor_.loadModel(f.getFullPathName().toStdString(), err)) {
        setStatus("Model load failed: " + juce::String(err));
        return false;
    }
    modelFile_ = f;
    setStatus("Model loaded: " + f.getFileName());
    return true;
}

void InverseSynthProcessor::tryAutoLoadModel() {
    const auto bin = pluginBinary();
    const juce::File candidates[] = {
        appDataDir().getChildFile(kModelName),
        bin.getSiblingFile(kModelName),
        bin.getChildFile("Contents/Resources").getChildFile(kModelName),
        bin.getParentDirectory().getSiblingFile("Resources").getChildFile(kModelName),
    };
    for (const auto& c : candidates)
        if (c.existsAsFile() && loadModelFile(c)) return;
    setStatus("No model found. Use 'Load model' or copy " + juce::String(kModelName) + " to " +
              appDataDir().getFullPathName());
}

// =========================================================================
// State
// =========================================================================
void InverseSynthProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts.copyState();
    state.setProperty("modelPath", modelFile_.getFullPathName(), nullptr);
    state.setProperty("targetPath", targetFile_.getFullPathName(), nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void InverseSynthProcessor::setStateInformation(const void* data, int size) {
    auto xml = getXmlFromBinary(data, size);
    if (!xml || !xml->hasTagName(apvts.state.getType())) return;
    auto state = juce::ValueTree::fromXml(*xml);
    apvts.replaceState(state);
    const juce::File model(state.getProperty("modelPath").toString());
    if (model.existsAsFile() && model != modelFile_) loadModelFile(model);
    const juce::File target(state.getProperty("targetPath").toString());
    if (target.existsAsFile()) loadAudioFile(target);
}

juce::AudioProcessorEditor* InverseSynthProcessor::createEditor() {
    return new InverseSynthEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new InverseSynthProcessor();
}
