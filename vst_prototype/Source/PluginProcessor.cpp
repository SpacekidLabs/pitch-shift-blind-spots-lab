#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace
{
constexpr int adaptiveRouterStableBlocks = 8;
constexpr int adaptiveRouterMinimumHoldBlocks = 14;
constexpr int adaptiveRouterTransitionBlocks = 4;
constexpr float adaptiveRouterTransitionWetCeiling = 0.70f;

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
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "backendMode",
        "Backend Mode",
        juce::StringArray { "Adaptive", "Phase Vocoder", "WSOLA-lite", "PSOLA-lite", "Rubber Band", "Bypass" },
        0));
    params.push_back(std::make_unique<juce::AudioParameterBool>("safeMode", "Safe Mode", true));
    return { params.begin(), params.end() };
}

void PitchShiftBlindSpotsAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engine_.prepare(sampleRate, samplesPerBlock);
    phaseVocoder_.prepare(sampleRate, getTotalNumOutputChannels());
    rubberBand_.prepare(sampleRate, getTotalNumOutputChannels());
    pitchShifter_.prepare(sampleRate, getTotalNumOutputChannels());
    monoScratch_.setSize(1, samplesPerBlock, false, false, true);
    dryScratch_.setSize(getTotalNumOutputChannels(), samplesPerBlock, false, false, true);
    wetScratch_.setSize(getTotalNumOutputChannels(), samplesPerBlock, false, false, true);
    resetAdaptiveRouter();
}

void PitchShiftBlindSpotsAudioProcessor::releaseResources()
{
    phaseVocoder_.reset();
    rubberBand_.reset();
    pitchShifter_.reset();
    resetAdaptiveRouter();
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
    backendInputPointers_.resize(static_cast<std::size_t>(numChannels));
    backendOutputPointers_.resize(static_cast<std::size_t>(numChannels));

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
    const auto backendMode = static_cast<BackendMode>(static_cast<int>(parameters_.getRawParameterValue("backendMode")->load()));

    const auto features = engine_.analyzeBlock(mono, numSamples);
    const auto decision = engine_.decide(features, params);

    lastState_.store(static_cast<int>(decision.state));
    lastStrategy_.store(static_cast<int>(decision.strategy));
    lastBackendMode_.store(static_cast<int>(backendMode));
    lastDisagreement_.store(features.observerDisagreement);
    lastSafeModeActive_.store(decision.safeModeActive);

    auto requestedBackend = ActiveBackend::bypass;
    if (std::abs(decision.effectiveShiftSemitones) >= 0.01f)
    {
        switch (backendMode)
        {
            case BackendMode::adaptive:
                requestedBackend = chooseAdaptiveBackend(decision.strategy, decision.effectiveShiftSemitones);
                break;
            case BackendMode::phaseVocoder:
                requestedBackend = ActiveBackend::phaseVocoder;
                break;
            case BackendMode::wsolaLite:
                requestedBackend = ActiveBackend::wsolaLite;
                break;
            case BackendMode::psolaLite:
                requestedBackend = ActiveBackend::psolaLite;
                break;
            case BackendMode::rubberBand:
                requestedBackend = rubberBand_.isAvailable() ? ActiveBackend::rubberBand : ActiveBackend::rubberBandUnavailableUsingPhaseVocoder;
                break;
            case BackendMode::bypass:
                requestedBackend = ActiveBackend::bypass;
                break;
        }
    }
    if (requestedBackend == ActiveBackend::rubberBand && !rubberBand_.isAvailable())
        requestedBackend = ActiveBackend::rubberBandUnavailableUsingPhaseVocoder;

    auto activeBackend = requestedBackend;
    if (backendMode == BackendMode::adaptive)
        activeBackend = stabilizeAdaptiveBackend(requestedBackend);
    else
        resetAdaptiveRouter();

    const auto wetAmount = juce::jlimit(0.0f, 1.0f, decision.effectiveDryWet);
    for (int channel = 0; channel < numChannels; ++channel)
    {
        if (activeBackend == ActiveBackend::bypass)
        {
            wetScratch_.copyFrom(channel, 0, dryScratch_, channel, 0, numSamples);
        }
        else if (activeBackend == ActiveBackend::phaseVocoder
                 || activeBackend == ActiveBackend::rubberBandUnavailableUsingPhaseVocoder)
        {
            phaseVocoder_.processChannel(
                dryScratch_.getReadPointer(channel),
                wetScratch_.getWritePointer(channel),
                numSamples,
                channel,
                decision.effectiveShiftSemitones);
        }
        else if (activeBackend == ActiveBackend::rubberBand)
        {
            backendInputPointers_[static_cast<std::size_t>(channel)] = dryScratch_.getReadPointer(channel);
            backendOutputPointers_[static_cast<std::size_t>(channel)] = wetScratch_.getWritePointer(channel);
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

    if (activeBackend == ActiveBackend::rubberBand)
    {
        rubberBand_.processBlock(
            backendInputPointers_.data(),
            backendOutputPointers_.data(),
            numChannels,
            numSamples,
            decision.effectiveShiftSemitones);
    }

    if (params.safeMode
        && (activeBackend == ActiveBackend::phaseVocoder
            || activeBackend == ActiveBackend::rubberBandUnavailableUsingPhaseVocoder)
        && renderLooksCollapsed(dryScratch_, wetScratch_, numChannels, numSamples))
    {
        activeBackend = ActiveBackend::phaseVocoderRescuedByWsolaLite;
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
        auto localWetAmount = activeBackend == ActiveBackend::phaseVocoderRescuedByWsolaLite ? std::min(wetAmount, 0.75f) : wetAmount;
        if (adaptiveBackendTransitionBlocksRemaining_ > 0)
            localWetAmount = std::min(localWetAmount, adaptiveRouterTransitionWetCeiling);
        const auto localDryAmount = 1.0f - localWetAmount;

        auto* output = buffer.getWritePointer(channel);
        const auto* dry = dryScratch_.getReadPointer(channel);
        const auto* wet = wetScratch_.getReadPointer(channel);
        for (int i = 0; i < numSamples; ++i)
            output[i] = dry[i] * localDryAmount + wet[i] * localWetAmount;
    }

    if (adaptiveBackendTransitionBlocksRemaining_ > 0)
        --adaptiveBackendTransitionBlocksRemaining_;
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
            return "phase_vocoder";
        case ActiveBackend::wsolaLite:
            return "wsola_lite";
        case ActiveBackend::psolaLite:
            return "psola_lite";
        case ActiveBackend::rubberBand:
            return "rubber_band";
        case ActiveBackend::rubberBandUnavailableUsingPhaseVocoder:
            return "rubber_band_unavailable_using_phase_vocoder";
        case ActiveBackend::phaseVocoderRescuedByWsolaLite:
            return "phase_vocoder_collapsed_rescued_by_wsola_lite";
    }

    return "unknown";
}

PitchShiftBlindSpotsAudioProcessor::ActiveBackend PitchShiftBlindSpotsAudioProcessor::chooseAdaptiveBackend(
    psbsl::PitchStrategy strategy,
    float shiftSemitones)
{
    if (std::abs(shiftSemitones) < 0.01f)
        return ActiveBackend::bypass;

    switch (strategy)
    {
        case psbsl::PitchStrategy::phaseVocoder:
        case psbsl::PitchStrategy::spectralFallback:
        case psbsl::PitchStrategy::dryWetSafety:
            return ActiveBackend::phaseVocoder;
        case psbsl::PitchStrategy::wsola:
            return ActiveBackend::wsolaLite;
        case psbsl::PitchStrategy::psola:
            return ActiveBackend::psolaLite;
        case psbsl::PitchStrategy::rubberBand:
            return ActiveBackend::rubberBand;
    }

    return ActiveBackend::phaseVocoder;
}

PitchShiftBlindSpotsAudioProcessor::ActiveBackend PitchShiftBlindSpotsAudioProcessor::stabilizeAdaptiveBackend(
    ActiveBackend requestedBackend)
{
    if (requestedBackend == ActiveBackend::bypass)
    {
        resetAdaptiveRouter();
        return ActiveBackend::bypass;
    }

    if (!adaptiveRouterInitialized_)
    {
        adaptiveRouterInitialized_ = true;
        heldAdaptiveBackend_ = requestedBackend;
        pendingAdaptiveBackend_ = requestedBackend;
        pendingAdaptiveBackendBlocks_ = 0;
        adaptiveBackendHoldBlocksRemaining_ = adaptiveRouterMinimumHoldBlocks;
        adaptiveBackendTransitionBlocksRemaining_ = 0;
        return heldAdaptiveBackend_;
    }

    if (adaptiveBackendHoldBlocksRemaining_ > 0)
        --adaptiveBackendHoldBlocksRemaining_;

    if (requestedBackend == heldAdaptiveBackend_)
    {
        pendingAdaptiveBackend_ = requestedBackend;
        pendingAdaptiveBackendBlocks_ = 0;
        return heldAdaptiveBackend_;
    }

    if (adaptiveBackendHoldBlocksRemaining_ > 0)
        return heldAdaptiveBackend_;

    if (requestedBackend != pendingAdaptiveBackend_)
    {
        pendingAdaptiveBackend_ = requestedBackend;
        pendingAdaptiveBackendBlocks_ = 1;
        return heldAdaptiveBackend_;
    }

    ++pendingAdaptiveBackendBlocks_;
    if (pendingAdaptiveBackendBlocks_ < adaptiveRouterStableBlocks)
        return heldAdaptiveBackend_;

    heldAdaptiveBackend_ = requestedBackend;
    pendingAdaptiveBackendBlocks_ = 0;
    adaptiveBackendHoldBlocksRemaining_ = adaptiveRouterMinimumHoldBlocks;
    adaptiveBackendTransitionBlocksRemaining_ = adaptiveRouterTransitionBlocks;
    return heldAdaptiveBackend_;
}

void PitchShiftBlindSpotsAudioProcessor::resetAdaptiveRouter()
{
    adaptiveRouterInitialized_ = false;
    heldAdaptiveBackend_ = ActiveBackend::bypass;
    pendingAdaptiveBackend_ = ActiveBackend::bypass;
    pendingAdaptiveBackendBlocks_ = 0;
    adaptiveBackendHoldBlocksRemaining_ = 0;
    adaptiveBackendTransitionBlocksRemaining_ = 0;
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
    return rmsRatio < 0.30 || activeRatio < 0.55;
}

juce::String PitchShiftBlindSpotsAudioProcessor::getBackendStatus() const
{
    const auto mode = enumFromAtomic(lastBackendMode_, BackendMode::adaptive);
    if (mode != BackendMode::adaptive)
        return "manual backend mode";

    const auto backend = enumFromAtomic(lastBackend_, ActiveBackend::bypass);
    switch (backend)
    {
        case ActiveBackend::rubberBandUnavailableUsingPhaseVocoder:
            return "rubber band SDK not embedded yet";
        case ActiveBackend::rubberBand:
            return "real rubber band backend with adaptive smoothing";
        case ActiveBackend::phaseVocoderRescuedByWsolaLite:
            return "fallback health rescue active";
        case ActiveBackend::wsolaLite:
            return "prototype WSOLA approximation";
        case ActiveBackend::psolaLite:
            return "prototype PSOLA approximation";
        case ActiveBackend::phaseVocoder:
            return "available spectral backend with adaptive smoothing";
        case ActiveBackend::bypass:
            return "no pitch shift";
    }

    return "unknown backend status";
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PitchShiftBlindSpotsAudioProcessor();
}
