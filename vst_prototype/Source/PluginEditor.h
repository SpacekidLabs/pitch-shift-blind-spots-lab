#pragma once

#include <JuceHeader.h>

#include "PluginProcessor.h"

class PitchShiftBlindSpotsAudioProcessorEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit PitchShiftBlindSpotsAudioProcessorEditor(PitchShiftBlindSpotsAudioProcessor& processor);
    ~PitchShiftBlindSpotsAudioProcessorEditor() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;

    PitchShiftBlindSpotsAudioProcessor& processor_;

    juce::Slider shiftSlider_;
    juce::Slider dryWetSlider_;
    juce::ToggleButton safeModeButton_;
    juce::Label shiftLabel_;
    juce::Label dryWetLabel_;
    juce::Label stateLabel_;
    juce::Label strategyLabel_;
    juce::Label backendLabel_;
    juce::Label disagreementLabel_;

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    std::unique_ptr<SliderAttachment> shiftAttachment_;
    std::unique_ptr<SliderAttachment> dryWetAttachment_;
    std::unique_ptr<ButtonAttachment> safeModeAttachment_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PitchShiftBlindSpotsAudioProcessorEditor)
};
