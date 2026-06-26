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
    phaseVocoder_.prepare(sampleRate, getTotalNumOutputChannels());
    pitchShifter_.prepare(sampleRate, getTotalNumOutputChannels());
    monoScratch_.setSize(1, samplesPerBlock, false, false, true);
    dryScratch_.setSize(getTotalNumOutputChannels(), samplesPerBlock, false, false, true);
    wetScratch_.setSize(getTotalNumOutputChannels(), samplesPerBlock, false, false, true);
}

void PitchShiftBlindSpotsAudioProcessor::releaseResources()
{
    phaseVocoder_.reset();
    pitchShifter_.reset();
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
    if (dryScratch_.getNumSamples() < numSamples || dryScratch_.getNumChannels() < numChannels)
        dryScratch_.setSize(numChannels, numSamples, false, false, true);
    if (wetScratch_.getNumSamples() < numSamples || wetScratch_.getNumChannels() < numChannels)
        wetScratch_.setSize(numChannels, numSamples, false, false, true);

    dryScratch_.makeCopyOf(buffer, true);
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

    auto activeBackend = ActiveBackend::phaseVocoder;
    if (std::abs(decision.effectiveShiftSemitones) < 0.01f)
        activeBackend = ActiveBackend::bypass;

    const auto wetAmount = juce::jlimit(0.0f, 1.0f, decision.effectiveDryWet);
    for (int channel = 0; channel < numChannels; ++channel)
    {
        if (activeBackend == ActiveBackend::bypass)
        {
            wetScratch_.copyFrom(channel, 0, dryScratch_, channel, 0, numSamples);
        }
        else if (activeBackend == ActiveBackend::phaseVocoder)
        {
            phaseVocoder_.processChannel(
                dryScratch_.getReadPointer(channel),
                wetScratch_.getWritePointer(channel),
                numSamples,
                channel,
                decision.effectiveShiftSemitones);
        }
        else
        {
            pitchShifter_.processChannel(
                dryScratch_.getReadPointer(channel),
                wetScratch_.getWritePointer(channel),
                numSamples,
                channel,
                decision.effectiveShiftSemitones);
        }
    }

    if (activeBackend == ActiveBackend::phaseVocoder && renderLooksCollapsed(dryScratch_, wetScratch_, numChannels, numSamples))
    {
        activeBackend = ActiveBackend::phaseVocoderRescuedByOverlap;
        for (int channel = 0; channel < numChannels; ++channel)
        {
            pitchShifter_.processChannel(
                dryScratch_.getReadPointer(channel),
                wetScratch_.getWritePointer(channel),
                numSamples,
                channel,
                decision.effectiveShiftSemitones);
        }
    }

    lastBackend_.store(static_cast<int>(activeBackend));

    for (int channel = 0; channel < numChannels; ++channel)
    {
        const auto localWetAmount = activeBackend == ActiveBackend::phaseVocoderRescuedByOverlap ? std::min(wetAmount, 0.75f) : wetAmount;
        const auto localDryAmount = 1.0f - localWetAmount;

        auto* output = buffer.getWritePointer(channel);
        const auto* dry = dryScratch_.getReadPointer(channel);
        const auto* wet = wetScratch_.getReadPointer(channel);
        for (int i = 0; i < numSamples; ++i)
            output[i] = dry[i] * localDryAmount + wet[i] * localWetAmount;
    }
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

juce::String PitchShiftBlindSpotsAudioProcessor::getLastBackendName() const
{
    return toString(enumFromAtomic(lastBackend_, ActiveBackend::bypass));
}

const char* PitchShiftBlindSpotsAudioProcessor::toString(ActiveBackend backend)
{
    switch (backend)
    {
        case ActiveBackend::bypass:
            return "bypass";
        case ActiveBackend::phaseVocoder:
            return "phase_vocoder_fallback";
        case ActiveBackend::simpleOverlap:
            return "simple_overlap_legacy";
        case ActiveBackend::phaseVocoderRescuedByOverlap:
            return "phase_vocoder_collapsed_rescued_by_overlap";
    }

    return "unknown";
}

bool PitchShiftBlindSpotsAudioProcessor::renderLooksCollapsed(
    const juce::AudioBuffer<float>& dry,
    const juce::AudioBuffer<float>& wet,
    int numChannels,
    int numSamples)
{
    double drySquares = 0.0;
    double wetSquares = 0.0;
    auto dryActive = 0;
    auto wetActive = 0;

    for (int channel = 0; channel < numChannels; ++channel)
    {
        const auto* dryData = dry.getReadPointer(channel);
        const auto* wetData = wet.getReadPointer(channel);
        for (int i = 0; i < numSamples; ++i)
        {
            drySquares += static_cast<double>(dryData[i]) * dryData[i];
            wetSquares += static_cast<double>(wetData[i]) * wetData[i];
            if (std::abs(dryData[i]) > 1.0e-4f)
                ++dryActive;
            if (std::abs(wetData[i]) > 1.0e-4f)
                ++wetActive;
        }
    }

    const auto count = std::max(1, numChannels * numSamples);
    const auto dryRms = std::sqrt(drySquares / static_cast<double>(count));
    const auto wetRms = std::sqrt(wetSquares / static_cast<double>(count));
    if (dryRms < 1.0e-5)
        return false;

    const auto rmsRatio = wetRms / dryRms;
    const auto activeRatio = dryActive > 0 ? static_cast<double>(wetActive) / static_cast<double>(dryActive) : 1.0;
    return rmsRatio < 0.18 || activeRatio < 0.25;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PitchShiftBlindSpotsAudioProcessor();
}
