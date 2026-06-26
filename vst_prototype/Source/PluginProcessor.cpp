#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace
{
template <typename EnumType>
EnumType enumFromAtomic(const std::atomic<int>& value, EnumType fallback)
{
    const auto raw = value.load();
    return raw >= 0 ? static_cast<EnumType>(raw) : fallback;
}
} // namespace

PitchShiftBlindSpotsAudioProcessor::PitchShiftBlindSpotsAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "PARAMETERS", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout PitchShiftBlindSpotsAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "shift",
        "Shift",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.01f),
        0.0f,
        "semitones"));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "dryWet",
        "Dry/Wet",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f),
        1.0f));
    params.push_back(std::make_unique<juce::AudioParameterBool>("safeMode", "Safe Mode", true));
    return { params.begin(), params.end() };
}

void PitchShiftBlindSpotsAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engine_.prepare(sampleRate, samplesPerBlock);
    monoScratch_.setSize(1, samplesPerBlock, false, false, true);
}

void PitchShiftBlindSpotsAudioProcessor::releaseResources()
{
}

bool PitchShiftBlindSpotsAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& mainOut = layouts.getMainOutputChannelSet();
    const auto& mainIn = layouts.getMainInputChannelSet();
    return mainIn == mainOut && (mainOut == juce::AudioChannelSet::mono() || mainOut == juce::AudioChannelSet::stereo());
}

void PitchShiftBlindSpotsAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return;

    if (monoScratch_.getNumSamples() < numSamples)
        monoScratch_.setSize(1, numSamples, false, false, true);

    auto* mono = monoScratch_.getWritePointer(0);
    std::fill(mono, mono + numSamples, 0.0f);
    for (int channel = 0; channel < numChannels; ++channel)
    {
        const auto* input = buffer.getReadPointer(channel);
        for (int i = 0; i < numSamples; ++i)
            mono[i] += input[i] / static_cast<float>(numChannels);
    }

    psbsl::EngineParams params;
    params.shiftSemitones = *parameters_.getRawParameterValue("shift");
    params.dryWet = *parameters_.getRawParameterValue("dryWet");
    params.safeMode = parameters_.getRawParameterValue("safeMode")->load() > 0.5f;

    const auto features = engine_.analyzeBlock(mono, numSamples);
    const auto decision = engine_.decide(features, params);

    lastState_.store(static_cast<int>(decision.state));
    lastStrategy_.store(static_cast<int>(decision.strategy));
    lastDisagreement_.store(features.observerDisagreement);
    lastSafeModeActive_.store(decision.safeModeActive);

    // v0 is intentionally pass-through. The selector state is live; pitch-shift backends attach here next.
    for (int channel = 0; channel < numChannels; ++channel)
        buffer.applyGain(channel, 0, numSamples, juce::jlimit(0.0f, 1.0f, decision.effectiveDryWet));
}

juce::AudioProcessorEditor* PitchShiftBlindSpotsAudioProcessor::createEditor()
{
    return new PitchShiftBlindSpotsAudioProcessorEditor(*this);
}

void PitchShiftBlindSpotsAudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String PitchShiftBlindSpotsAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void PitchShiftBlindSpotsAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void PitchShiftBlindSpotsAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = parameters_.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void PitchShiftBlindSpotsAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
        parameters_.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::String PitchShiftBlindSpotsAudioProcessor::getLastStateName() const
{
    return psbsl::AdaptivePitchEngine::toString(enumFromAtomic(lastState_, psbsl::SignalState::untracked));
}

juce::String PitchShiftBlindSpotsAudioProcessor::getLastStrategyName() const
{
    return psbsl::AdaptivePitchEngine::toString(enumFromAtomic(lastStrategy_, psbsl::PitchStrategy::rubberBand));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PitchShiftBlindSpotsAudioProcessor();
}
