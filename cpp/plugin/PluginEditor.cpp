#include "PluginEditor.h"

using namespace invsynth;

namespace {
const juce::Colour kBg(0xff16181d), kPanel(0xff1f232b), kAccent(0xff4fc3f7),
                   kWindow(0x334fc3f7), kText(0xffdfe3ea), kDim(0xff8a93a3);
constexpr int kKnobW = 68, kKnobH = 88, kSecPad = 8, kSecTitle = 18;
}

// ------------------------------------------------------------------ DropZone
void DropZone::paint(juce::Graphics& g) {
    auto r = getLocalBounds().toFloat().reduced(1.f);
    g.setColour(kPanel);
    g.fillRoundedRectangle(r, 6.f);
    g.setColour(hover_ ? kAccent : kDim.withAlpha(0.5f));
    g.drawRoundedRectangle(r, 6.f, hover_ ? 2.f : 1.f);
    if (!st_.hasTarget || st_.overview.empty()) {
        g.setColour(kDim);
        g.drawText("Drop an audio file here (any length) or click to browse",
                   getLocalBounds(), juce::Justification::centred);
        return;
    }
    const auto area = getLocalBounds().reduced(8, 18);
    const double dur = std::max(st_.durationSec, 1e-6);
    const float x0 = area.getX() + (float)(st_.windowStartSec / dur) * area.getWidth();
    const float x1 = area.getX() + (float)((st_.windowStartSec + st_.windowLenSec) / dur) * area.getWidth();
    g.setColour(kWindow);
    g.fillRect(juce::Rectangle<float>(x0, (float)area.getY(), std::max(2.f, x1 - x0), (float)area.getHeight()));
    g.setColour(kAccent);
    const float mid = (float)area.getCentreY(), h = area.getHeight() * 0.5f;
    const size_t n = st_.overview.size();
    for (int px = 0; px < area.getWidth(); ++px) {
        const size_t a = (size_t)px * n / (size_t)area.getWidth();
        const size_t b = std::max(a + 1, (size_t)(px + 1) * n / (size_t)area.getWidth());
        float v = 0.f;
        for (size_t i = a; i < b && i < n; ++i) v = std::max(v, st_.overview[i]);
        g.drawVerticalLine(area.getX() + px, mid - v * h, mid + v * h);
    }
    g.setColour(kText);
    g.setFont(12.f);
    g.drawText(st_.fileName + juce::String::formatted("   %.2f s   |   window %.2f - %.2f s",
               st_.durationSec, st_.windowStartSec, st_.windowStartSec + st_.windowLenSec),
               getLocalBounds().reduced(8, 2).removeFromTop(16), juce::Justification::left);
}

void DropZone::filesDropped(const juce::StringArray& files, int, int) {
    hover_ = false;
    repaint();
    if (!files.isEmpty() && onFile) onFile(juce::File(files[0]));
}

// ---------------------------------------------------------------------- Knob
Knob::Knob(juce::AudioProcessorValueTreeState& s, const juce::String& id, const juce::String& label)
    : slider_(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow),
      att_(s, id, slider_) {
    slider_.setTextBoxStyle(juce::Slider::TextBoxBelow, false, kKnobW, 16);
    slider_.setColour(juce::Slider::rotarySliderFillColourId, kAccent);
    slider_.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider_.setColour(juce::Slider::textBoxTextColourId, kText);
    label_.setText(label, juce::dontSendNotification);
    label_.setJustificationType(juce::Justification::centred);
    label_.setColour(juce::Label::textColourId, kDim);
    label_.setFont(juce::FontOptions(11.f));
    addAndMakeVisible(slider_);
    addAndMakeVisible(label_);
}

void Knob::resized() {
    auto r = getLocalBounds();
    label_.setBounds(r.removeFromTop(14));
    slider_.setBounds(r);
}

// ----------------------------------------------------------------- ParamPanel
ParamPanel::ParamPanel(juce::AudioProcessorValueTreeState& s) {
    for (int i = 0; i < kNumParams; ++i) {
        const auto& info = paramTable()[(size_t)i];
        if (sections_.empty() || sections_.back().name != info.section)
            sections_.push_back({info.section, {}, {}});
        sections_.back().knobs.push_back(std::make_unique<Knob>(s, info.id, info.label));
        addAndMakeVisible(*sections_.back().knobs.back());
    }
}

int ParamPanel::preferredHeight(int width) const {
    int x = 0, y = 0;
    const int rowH = kKnobH + kSecTitle + 2 * kSecPad;
    for (auto& s : sections_) {
        const int w = (int)s.knobs.size() * kKnobW + 2 * kSecPad;
        if (x > 0 && x + w > width) { x = 0; y += rowH + 6; }
        x += w + 6;
    }
    return y + rowH + 4;
}

void ParamPanel::resized() {
    int x = 0, y = 0;
    const int rowH = kKnobH + kSecTitle + 2 * kSecPad;
    for (auto& s : sections_) {
        const int w = (int)s.knobs.size() * kKnobW + 2 * kSecPad;
        if (x > 0 && x + w > getWidth()) { x = 0; y += rowH + 6; }
        s.area = {x, y, w, rowH};
        for (size_t k = 0; k < s.knobs.size(); ++k)
            s.knobs[k]->setBounds(x + kSecPad + (int)k * kKnobW, y + kSecTitle + kSecPad, kKnobW, kKnobH);
        x += w + 6;
    }
}

void ParamPanel::paint(juce::Graphics& g) {
    for (auto& s : sections_) {
        g.setColour(kPanel);
        g.fillRoundedRectangle(s.area.toFloat(), 5.f);
        g.setColour(kAccent);
        g.setFont(juce::FontOptions(12.f, juce::Font::bold));
        g.drawText(s.name.toUpperCase(), s.area.withHeight(kSecTitle + 4).reduced(kSecPad, 0),
                   juce::Justification::left);
    }
}

// ---------------------------------------------------------------------- Editor
InverseSynthEditor::InverseSynthEditor(InverseSynthProcessor& p)
    : AudioProcessorEditor(p), proc_(p),
      gainAtt_(p.apvts, "output_gain", gain_),
      panel_(p.apvts),
      keyboard_(p.keyboardState, juce::MidiKeyboardComponent::horizontalKeyboard) {
    title_.setText("INVERSE SYNTH", juce::dontSendNotification);
    title_.setFont(juce::FontOptions(20.f, juce::Font::bold));
    title_.setColour(juce::Label::textColourId, kText);
    for (auto* l : {&status_, &modelState_, &genLabel_, &secLabel_, &gainLabel_})
        l->setColour(juce::Label::textColourId, kDim);
    genLabel_.setText("Generations", juce::dontSendNotification);
    secLabel_.setText("Max compare (s)", juce::dontSendNotification);
    gainLabel_.setText("Output dB", juce::dontSendNotification);

    mode_.addItem("Selected 5 s window", 1);
    mode_.addItem("Whole file", 2);
    mode_.addItem("Progressive (5 s > 2x > whole)", 3);
    mode_.setSelectedId(3);
    finetune_.setToggleState(true, juce::dontSendNotification);

    for (auto* s : {&generations_, &maxSeconds_, &gain_}) {
        s->setSliderStyle(juce::Slider::LinearHorizontal);
        s->setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
    }
    generations_.setRange(6, 300, 1);
    generations_.setValue(45);
    maxSeconds_.setRange(5, 300, 1);
    maxSeconds_.setValue(30);

    drop_.onFile = [this](const juce::File& f) { proc_.loadAudioFile(f); };
    drop_.onClick = [this] { chooseAudio(); };
    loadModel_.onClick = [this] { chooseModel(); };
    cancel_.onClick = [this] { proc_.cancelJob(); };
    playTarget_.onClick = [this] { proc_.setPreviewPlaying(!proc_.isPreviewPlaying()); };
    playMatch_.onClick = [this] { auditionMatch(); };
    analyse_.onClick = [this] {
        AnalysisOptions opt;
        opt.runFinetune = finetune_.getToggleState();
        opt.finetune.generations = (int)generations_.getValue();
        opt.finetune.maxCompareSec = maxSeconds_.getValue();
        opt.finetune.mode = mode_.getSelectedId() == 1 ? WindowMode::Selected
                          : mode_.getSelectedId() == 2 ? WindowMode::Full : WindowMode::Progressive;
        proc_.startAnalysis(opt);
    };

    for (juce::Component* c : std::initializer_list<juce::Component*>{
             &title_, &status_, &modelState_, &drop_, &loadModel_, &analyse_, &cancel_, &playTarget_,
             &playMatch_, &finetune_, &mode_, &generations_, &maxSeconds_, &gain_, &genLabel_,
             &secLabel_, &gainLabel_, &progressBar_, &viewport_, &keyboard_})
        addAndMakeVisible(c);
    viewport_.setViewedComponent(&panel_, false);
    viewport_.setScrollBarsShown(true, false);
    keyboard_.setAvailableRange(24, 96);
    keyboard_.setKeyWidth(14.f);

    setResizable(true, true);
    setResizeLimits(720, 560, 1800, 1400);
    setSize(980, 720);
    startTimerHz(15);
}

InverseSynthEditor::~InverseSynthEditor() { stopTimer(); }

void InverseSynthEditor::paint(juce::Graphics& g) { g.fillAll(kBg); }

void InverseSynthEditor::resized() {
    auto r = getLocalBounds().reduced(10);
    auto top = r.removeFromTop(28);
    title_.setBounds(top.removeFromLeft(200));
    loadModel_.setBounds(top.removeFromRight(110));
    top.removeFromRight(8);
    modelState_.setBounds(top);
    r.removeFromTop(6);
    drop_.setBounds(r.removeFromTop(110));
    r.removeFromTop(6);

    auto row = r.removeFromTop(26);
    analyse_.setBounds(row.removeFromLeft(90)); row.removeFromLeft(4);
    cancel_.setBounds(row.removeFromLeft(70)); row.removeFromLeft(8);
    finetune_.setBounds(row.removeFromLeft(150)); row.removeFromLeft(4);
    mode_.setBounds(row.removeFromLeft(230)); row.removeFromLeft(8);
    playTarget_.setBounds(row.removeFromLeft(100)); row.removeFromLeft(4);
    playMatch_.setBounds(row.removeFromLeft(90));
    r.removeFromTop(4);

    auto row2 = r.removeFromTop(24);
    const int third = row2.getWidth() / 3;
    auto a = row2.removeFromLeft(third), b = row2.removeFromLeft(third), c = row2;
    genLabel_.setBounds(a.removeFromLeft(80)); generations_.setBounds(a.reduced(4, 0));
    secLabel_.setBounds(b.removeFromLeft(105)); maxSeconds_.setBounds(b.reduced(4, 0));
    gainLabel_.setBounds(c.removeFromLeft(70)); gain_.setBounds(c.reduced(4, 0));
    r.removeFromTop(4);
    progressBar_.setBounds(r.removeFromTop(18));
    status_.setBounds(r.removeFromTop(22));

    keyboard_.setBounds(r.removeFromBottom(70));
    r.removeFromBottom(6);
    viewport_.setBounds(r);
    const int w = r.getWidth() - viewport_.getScrollBarThickness();
    panel_.setSize(w, panel_.preferredHeight(w));
}

void InverseSynthEditor::timerCallback() {
    const auto st = proc_.getUiState();
    status_.setText(st.status, juce::dontSendNotification);
    modelState_.setText(proc_.isModelReady() ? "Model: ready (offline, CPU)" : "Model: not loaded",
                        juce::dontSendNotification);
    drop_.setState(st);
    const bool busy = proc_.isBusy();
    progress_ = busy ? (double)proc_.getProgress() : (proc_.getProgress() >= 1.f ? 1.0 : 0.0);
    analyse_.setEnabled(!busy && st.hasTarget);
    cancel_.setEnabled(busy);
    playTarget_.setButtonText(proc_.isPreviewPlaying() ? "Stop original" : "Play original");
}

void InverseSynthEditor::chooseAudio() {
    chooser_ = std::make_unique<juce::FileChooser>("Choose a sound to replicate", juce::File(),
                                                   "*.wav;*.aif;*.aiff;*.flac;*.ogg;*.mp3");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              if (fc.getResult().existsAsFile()) proc_.loadAudioFile(fc.getResult());
                          });
}

void InverseSynthEditor::chooseModel() {
    chooser_ = std::make_unique<juce::FileChooser>("Choose inverse_synth.onnx", juce::File(), "*.onnx");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              if (fc.getResult().existsAsFile()) proc_.loadModelFile(fc.getResult());
                          });
}

// Plays the root note for as long as the analysed note was held.
void InverseSynthEditor::auditionMatch() {
    const auto st = proc_.getUiState();
    const float gate = proc_.apvts.getRawParameterValue("gate")->load();
    const double held = std::clamp(st.durationSec * gateFraction(gate), 0.2, 10.0);
    proc_.keyboardState.noteOn(1, 60, 0.9f);
    juce::WeakReference<InverseSynthProcessor> weak(&proc_);   // survives editor closing
    juce::Timer::callAfterDelay((int)(held * 1000.0), [weak] {
        if (auto* p = weak.get()) p->keyboardState.noteOff(1, 60, 0.f);
    });
}
