#include "PluginEditor.h"

PitchShiftBlindSpotsAudioProcessorEditor::PitchShiftBlindSpotsAudioProcessorEditor(
    PitchShiftBlindSpotsAudioProcessor& processor)
    : AudioProcessorEditor(&processor),
      processor_(processor)
{
    auto& state = processor_.getValueTreeState();

    shiftSlider_.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    shiftSlider_.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 86, 22);
    shiftSlider_.setNumDecimalPlacesToDisplay(2);
    addAndMakeVisible(shiftSlider_);

    dryWetSlider_.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    dryWetSlider_.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 86, 22);
    dryWetSlider_.setNumDecimalPlacesToDisplay(2);
    addAndMakeVisible(dryWetSlider_);

    backendModeBox_.addItemList(juce::StringArray { "Adaptive", "Phase Vocoder", "WSOLA-lite", "PSOLA-lite", "Rubber Band", "Bypass" }, 1);
    addAndMakeVisible(backendModeBox_);

    safeModeButton_.setButtonText("Safe mode");
    addAndMakeVisible(safeModeButton_);

    shiftLabel_.setText("Shift", juce::dontSendNotification);
    dryWetLabel_.setText("Dry/Wet", juce::dontSendNotification);
    backendModeLabel_.setText("Backend", juce::dontSendNotification);
    for (auto* label : { &shiftLabel_, &dryWetLabel_, &backendModeLabel_, &stateLabel_, &strategyLabel_, &backendLabel_, &backendStatusLabel_, &disagreementLabel_ })
    {
        label->setJustificationType(juce::Justification::centred);
        label->setColour(juce::Label::textColourId, juce::Colours::whitesmoke);
        addAndMakeVisible(*label);
    }

    shiftAttachment_ = std::make_unique<SliderAttachment>(state, "shift", shiftSlider_);
    dryWetAttachment_ = std::make_unique<SliderAttachment>(state, "dryWet", dryWetSlider_);
    backendModeAttachment_ = std::make_unique<ComboBoxAttachment>(state, "backendMode", backendModeBox_);
    safeModeAttachment_ = std::make_unique<ButtonAttachment>(state, "safeMode", safeModeButton_);

    setSize(580, 420);
    startTimerHz(20);
    timerCallback();
}

void PitchShiftBlindSpotsAudioProcessorEditor::paint(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    juce::ColourGradient gradient(
        juce::Colour::fromRGB(16, 28, 30),
        bounds.getTopLeft(),
        juce::Colour::fromRGB(43, 66, 58),
        bounds.getBottomRight(),
        false);
    gradient.addColour(0.58, juce::Colour::fromRGB(93, 79, 64));
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(bounds.reduced(8.0f), 18.0f);

    g.setColour(juce::Colour::fromRGB(244, 232, 207));
    g.setFont(juce::FontOptions(24.0f, juce::Font::bold));
    g.drawText("Pitch Shift Blind Spots", 28, 24, getWidth() - 56, 34, juce::Justification::centredLeft);

    g.setColour(juce::Colour::fromRGB(196, 177, 145));
    g.setFont(juce::FontOptions(13.0f));
    g.drawText("Adaptive selector prototype v0", 30, 58, getWidth() - 60, 24, juce::Justification::centredLeft);

    g.setColour(juce::Colour::fromRGBA(255, 250, 235, 34));
    g.drawRoundedRectangle(getLocalBounds().reduced(18).toFloat(), 18.0f, 1.2f);
}

void PitchShiftBlindSpotsAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(30);
    bounds.removeFromTop(74);

    auto controls = bounds.removeFromTop(190);
    auto shiftArea = controls.removeFromLeft(160);
    auto dryWetArea = controls.removeFromLeft(160);
    auto backendArea = controls.removeFromLeft(190);

    shiftLabel_.setBounds(shiftArea.removeFromTop(24));
    shiftSlider_.setBounds(shiftArea.reduced(10));

    dryWetLabel_.setBounds(dryWetArea.removeFromTop(24));
    dryWetSlider_.setBounds(dryWetArea.reduced(10));

    backendModeLabel_.setBounds(backendArea.removeFromTop(24));
    backendModeBox_.setBounds(backendArea.removeFromTop(32).reduced(6, 0));
    backendArea.removeFromTop(12);
    safeModeButton_.setBounds(backendArea.removeFromTop(34).reduced(6, 0));

    bounds.removeFromTop(16);
    stateLabel_.setBounds(bounds.removeFromTop(30));
    strategyLabel_.setBounds(bounds.removeFromTop(30));
    backendLabel_.setBounds(bounds.removeFromTop(30));
    backendStatusLabel_.setBounds(bounds.removeFromTop(30));
    disagreementLabel_.setBounds(bounds.removeFromTop(30));
}

void PitchShiftBlindSpotsAudioProcessorEditor::timerCallback()
{
    stateLabel_.setText("State: " + processor_.getLastStateName(), juce::dontSendNotification);
    strategyLabel_.setText("Observer wants: " + processor_.getLastStrategyName(), juce::dontSendNotification);
    backendLabel_.setText("Using: " + processor_.getLastBackendName(), juce::dontSendNotification);
    backendStatusLabel_.setText("Status: " + processor_.getBackendStatus(), juce::dontSendNotification);
    disagreementLabel_.setText(
        "Observer disagreement: " + juce::String(processor_.getLastDisagreement(), 3)
            + (processor_.isSafeModeActive() ? "  | safe mode active" : ""),
        juce::dontSendNotification);
}
