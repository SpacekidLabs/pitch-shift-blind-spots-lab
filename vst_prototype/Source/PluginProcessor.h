#pragma once

#include <JuceHeader.h>

#include <vector>

#include "AdaptivePitchEngine.h"
#include "PhaseVocoderPitchShifter.h"
#include "RubberBandPitchShifter.h"
#include "SimplePitchShifter.h"

class PitchShiftBlindSpotsAudioProcessor final : public juce::AudioProcessor
{
public:
    PitchShiftBlindSpotsAudioProcessor();
    ~PitchShiftBlindSpotsAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getValueTreeState() { return parameters_; }
    juce::String getLastStateName() const;
    juce::String getLastStrategyName() const;
    juce::String getLastBackendName() const;
    juce::String getBackendStatus() const;
    float getLastDisagreement() const { return lastDisagreement_.load(); }
    bool isSafeModeActive() const { return lastSafeModeActive_.load(); }

private:
    enum class ActiveBackend
    {
        bypass,
        phaseVocoder,
        wsolaLite,
        psolaLite,
        rubberBand,
        rubberBandUnavailableUsingPhaseVocoder,
        phaseVocoderRescuedByWsolaLite
    };

    enum class BackendMode
    {
        adaptive = 0,
        phaseVocoder,
        wsolaLite,
        psolaLite,
        rubberBand,
        bypass
    };

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    static const char* toString(ActiveBackend backend);
    static ActiveBackend chooseAdaptiveBackend(psbsl::PitchStrategy strategy, float shiftSemitones);
    static bool renderLooksCollapsed(const juce::AudioBuffer<float>& dry, const juce::AudioBuffer<float>& wet, int numChannels, int numSamples);

    juce::AudioProcessorValueTreeState parameters_;
    psbsl::AdaptivePitchEngine engine_;
    psbsl::PhaseVocoderPitchShifter phaseVocoder_;
    psbsl::RubberBandPitchShifter rubberBand_;
    psbsl::SimplePitchShifter pitchShifter_;

    std::atomic<int> lastState_ { static_cast<int>(psbsl::SignalState::untracked) };
    std::atomic<int> lastStrategy_ { static_cast<int>(psbsl::PitchStrategy::rubberBand) };
    std::atomic<int> lastBackend_ { static_cast<int>(ActiveBackend::bypass) };
    std::atomic<int> lastBackendMode_ { static_cast<int>(BackendMode::adaptive) };
    std::atomic<float> lastDisagreement_ { 0.0f };
    std::atomic<bool> lastSafeModeActive_ { false };

    juce::AudioBuffer<float> monoScratch_;
    juce::AudioBuffer<float> dryScratch_;
    juce::AudioBuffer<float> wetScratch_;
    std::vector<const float*> backendInputPointers_;
    std::vector<float*> backendOutputPointers_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PitchShiftBlindSpotsAudioProcessor)
};
