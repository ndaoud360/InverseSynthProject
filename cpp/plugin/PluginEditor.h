#pragma once
#include "PluginProcessor.h"

// Waveform + analysis-window display that doubles as the drop target.
class DropZone : public juce::Component, public juce::FileDragAndDropTarget {
public:
    std::function<void(const juce::File&)> onFile;
    std::function<void()> onClick;
    void setState(const InverseSynthProcessor::UiState& s) { st_ = s; repaint(); }
    void paint(juce::Graphics&) override;
    bool isInterestedInFileDrag(const juce::StringArray&) override { return true; }
    void fileDragEnter(const juce::StringArray&, int, int) override { hover_ = true; repaint(); }
    void fileDragExit(const juce::StringArray&) override { hover_ = false; repaint(); }
    void filesDropped(const juce::StringArray& files, int, int) override;
    void mouseUp(const juce::MouseEvent&) override { if (onClick) onClick(); }

private:
    InverseSynthProcessor::UiState st_;
    bool hover_ = false;
};

class Knob : public juce::Component {
public:
    Knob(juce::AudioProcessorValueTreeState& s, const juce::String& id, const juce::String& label);
    void resized() override;

private:
    juce::Slider slider_;
    juce::Label label_;
    juce::AudioProcessorValueTreeState::SliderAttachment att_;
};

class ParamPanel : public juce::Component {
public:
    explicit ParamPanel(juce::AudioProcessorValueTreeState& s);
    void paint(juce::Graphics&) override;
    void resized() override;
    int preferredHeight(int width) const;

private:
    struct Section { juce::String name; std::vector<std::unique_ptr<Knob>> knobs; juce::Rectangle<int> area; };
    std::vector<Section> sections_;
};

class InverseSynthEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit InverseSynthEditor(InverseSynthProcessor&);
    ~InverseSynthEditor() override;
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void chooseAudio();
    void chooseModel();
    void auditionMatch();

    InverseSynthProcessor& proc_;
    juce::Label title_, status_, modelState_;
    DropZone drop_;
    juce::TextButton loadModel_{"Load model"}, analyse_{"Analyse"}, cancel_{"Cancel"},
                     playTarget_{"Play original"}, playMatch_{"Play match"};
    juce::ToggleButton finetune_{"Fine-tune (stage 2)"};
    juce::ComboBox mode_;
    juce::Slider generations_, maxSeconds_, gain_;
    juce::Label genLabel_, secLabel_, gainLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment gainAtt_;
    double progress_ = 0.0;
    juce::ProgressBar progressBar_{progress_};
    ParamPanel panel_;
    juce::Viewport viewport_;
    juce::MidiKeyboardComponent keyboard_;
    std::unique_ptr<juce::FileChooser> chooser_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InverseSynthEditor)
};
