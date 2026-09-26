#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "../core/Effects.h"
#include "../core/Pipeline.h"

#include <array>
#include <atomic>
#include <thread>

// Threading model
//   audio thread   : voices + FX only. Reads parameters through atomics; never
//                    locks (preview buffer uses a try-lock), never allocates.
//   worker thread  : file decoding/resampling, window selection, ONNX
//                    inference and the NSGA-II search. One job at a time.
//   message thread : UI; applies analysis results to the parameters (so the
//                    host sees normal automation-style parameter changes).
class InverseSynthProcessor : public juce::AudioProcessor {
public:
    static constexpr int kVoices = 8;
    InverseSynthProcessor();
    ~InverseSynthProcessor() override;

    // ---- AudioProcessor
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 10.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // ---- inverse synthesis API (message thread)
    void loadAudioFile(const juce::File& f);
    void startAnalysis(const invsynth::AnalysisOptions& opt);
    void cancelJob() { cancel_ = true; }
    bool isBusy() const { return busy_.load(); }
    bool loadModelFile(const juce::File& f);   // also remembered in the state
    bool isModelReady() const { return predictor_.isReady(); }
    void setPreviewPlaying(bool on);
    bool isPreviewPlaying() const { return previewPlaying_.load(); }
    void applyParams(const invsynth::ParamVector& p, float gain);

    struct UiState {
        juce::String status = "Drop an audio file to begin";
        juce::String fileName;
        double durationSec = 0, windowStartSec = 0, windowLenSec = 0;
        std::vector<float> overview;            // peak envelope for drawing
        bool hasTarget = false;
    };
    UiState getUiState() const;
    float getProgress() const { return progress_.load(); }
    double lastLoss() const { return lastLoss_.load(); }

    juce::AudioProcessorValueTreeState apvts;
    juce::MidiKeyboardState keyboardState;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void runJob(std::function<void()> job);
    void setStatus(const juce::String& s);
    void tryAutoLoadModel();
    static juce::File pluginBinary();
    static juce::File findRuntimeLibrary();
    std::vector<float> decode(const juce::File& f, double targetRate, double maxSeconds,
                              double& fileSeconds);
    void handleNoteOn(int note, float vel);
    void handleNoteOff(int note);

    // audio
    std::array<std::atomic<float>*, invsynth::kNumParams> raw_{};
    std::atomic<float>* gainParam_ = nullptr;
    std::array<invsynth::SynthVoice, kVoices> voices_;
    invsynth::FxChain fx_;
    invsynth::ParamVector current_{};
    bool paramsDirty_ = true;
    std::vector<float> mono_;
    uint64_t noteCounter_ = 0;
    double sampleRate_ = 44100.0;

    juce::SpinLock previewLock_;
    juce::AudioBuffer<float> preview_;           // original sample at host rate
    double previewRate_ = 0;
    std::atomic<bool> previewPlaying_{false};
    int previewPos_ = 0;

    // analysis
    juce::AudioFormatManager formats_;
    invsynth::Predictor predictor_;
    std::vector<float> target22k_;               // mono @ 22.05 kHz, full length
    juce::File targetFile_, modelFile_;
    std::thread worker_;
    std::atomic<bool> busy_{false}, cancel_{false};
    std::atomic<float> progress_{0.f};
    std::atomic<double> lastLoss_{0.0};
    mutable juce::CriticalSection uiLock_;
    UiState ui_;

    JUCE_DECLARE_WEAK_REFERENCEABLE(InverseSynthProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InverseSynthProcessor)
};
