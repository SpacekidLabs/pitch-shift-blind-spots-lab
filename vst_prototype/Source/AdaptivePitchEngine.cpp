#include "AdaptivePitchEngine.h"

namespace psbsl
{

namespace
{
float clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}
} // namespace

void AdaptivePitchEngine::prepare(double sampleRate, int maxBlockSize)
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 44100.0;
    maxBlockSize_ = std::max(1, maxBlockSize);
}

SignalFeatures AdaptivePitchEngine::analyzeBlock(const float* samples, int numSamples) const
{
    SignalFeatures features;
    if (samples == nullptr || numSamples <= 0)
        return features;

    double sumSquares = 0.0;
    float peak = 0.0f;
    int zeroCrossings = 0;
    float previous = samples[0];
    float maxStep = 0.0f;
    double averageStep = 0.0;

    for (int i = 0; i < numSamples; ++i)
    {
        const float sample = samples[i];
        sumSquares += static_cast<double>(sample) * sample;
        peak = std::max(peak, std::abs(sample));

        if (i > 0)
        {
            if ((sample >= 0.0f && previous < 0.0f) || (sample < 0.0f && previous >= 0.0f))
                ++zeroCrossings;

            const float step = std::abs(sample - previous);
            maxStep = std::max(maxStep, step);
            averageStep += step;
            previous = sample;
        }
    }

    features.rms = static_cast<float>(std::sqrt(sumSquares / static_cast<double>(numSamples)));
    features.peak = peak;
    features.zeroCrossingRate = numSamples > 1 ? static_cast<float>(zeroCrossings) / static_cast<float>(numSamples - 1) : 0.0f;

    const float meanStep = numSamples > 1 ? static_cast<float>(averageStep / static_cast<double>(numSamples - 1)) : 0.0f;
    features.transientScore = clamp01((maxStep - meanStep) * 4.0f);

    const float crest = features.rms > 1.0e-6f ? features.peak / features.rms : 0.0f;
    const float zcrNoise = clamp01((features.zeroCrossingRate - 0.08f) / 0.22f);
    features.periodicity = clamp01(1.0f - zcrNoise - 0.05f * std::max(0.0f, crest - 8.0f));
    features.modulationScore = clamp01(meanStep * 12.0f);

    // v0 proxy until the plugin has true multi-observer pitch probes.
    features.observerDisagreement = clamp01(
        0.55f * features.modulationScore + 0.25f * features.transientScore + 0.20f * (1.0f - features.periodicity));

    return features;
}

Decision AdaptivePitchEngine::decide(const SignalFeatures& features, const EngineParams& params) const
{
    Decision decision;
    decision.state = classify(features);
    decision.effectiveDryWet = clamp01(params.dryWet);
    decision.effectiveShiftSemitones = params.shiftSemitones;

    switch (decision.state)
    {
        case SignalState::stablePeriodic:
            decision.strategy = PitchStrategy::rubberBand;
            break;
        case SignalState::octaveAmbiguous:
            decision.strategy = PitchStrategy::phaseVocoder;
            break;
        case SignalState::subharmonicAmbiguous:
            decision.strategy = PitchStrategy::phaseVocoder;
            break;
        case SignalState::detunedCompeting:
            decision.strategy = PitchStrategy::rubberBand;
            decision.effectiveShiftSemitones *= 0.75f;
            break;
        case SignalState::noiseLike:
            decision.strategy = PitchStrategy::spectralFallback;
            decision.effectiveDryWet *= 0.65f;
            break;
        case SignalState::transientLike:
            decision.strategy = PitchStrategy::wsola;
            break;
        case SignalState::modulatedPitch:
            decision.strategy = PitchStrategy::rubberBand;
            break;
        case SignalState::untracked:
            decision.strategy = PitchStrategy::rubberBand;
            decision.effectiveDryWet *= 0.75f;
            break;
    }

    decision.safeModeActive = params.safeMode && features.observerDisagreement > 0.62f;
    if (decision.safeModeActive)
    {
        if (decision.strategy == PitchStrategy::psola || decision.strategy == PitchStrategy::wsola)
            decision.strategy = PitchStrategy::rubberBand;

        decision.effectiveDryWet *= 0.75f;
        decision.effectiveShiftSemitones *= 0.85f;
    }

    return decision;
}

SignalState AdaptivePitchEngine::classify(const SignalFeatures& features) const
{
    if (features.rms < 1.0e-5f)
        return SignalState::untracked;

    if (features.zeroCrossingRate > 0.20f && features.periodicity < 0.45f)
        return SignalState::noiseLike;

    if (features.transientScore > 0.72f && features.periodicity < 0.65f)
        return SignalState::transientLike;

    if (features.observerDisagreement > 0.74f)
        return SignalState::detunedCompeting;

    if (features.modulationScore > 0.46f)
        return SignalState::modulatedPitch;

    if (features.periodicity > 0.72f)
        return SignalState::stablePeriodic;

    return SignalState::untracked;
}

const char* AdaptivePitchEngine::toString(SignalState state)
{
    switch (state)
    {
        case SignalState::stablePeriodic:
            return "stable_periodic";
        case SignalState::octaveAmbiguous:
            return "octave_ambiguous";
        case SignalState::subharmonicAmbiguous:
            return "subharmonic_ambiguous";
        case SignalState::detunedCompeting:
            return "detuned_competing";
        case SignalState::noiseLike:
            return "noise_like";
        case SignalState::transientLike:
            return "transient_like";
        case SignalState::modulatedPitch:
            return "modulated_pitch";
        case SignalState::untracked:
            return "untracked";
    }

    return "unknown";
}

const char* AdaptivePitchEngine::toString(PitchStrategy strategy)
{
    switch (strategy)
    {
        case PitchStrategy::phaseVocoder:
            return "phase_vocoder";
        case PitchStrategy::psola:
            return "psola";
        case PitchStrategy::wsola:
            return "wsola";
        case PitchStrategy::rubberBand:
            return "rubber_band";
        case PitchStrategy::spectralFallback:
            return "spectral_fallback";
        case PitchStrategy::dryWetSafety:
            return "dry_wet_safety";
    }

    return "unknown";
}

} // namespace psbsl
