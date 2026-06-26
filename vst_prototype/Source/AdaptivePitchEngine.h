#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace psbsl
{

enum class SignalState
{
    stablePeriodic,
    octaveAmbiguous,
    subharmonicAmbiguous,
    detunedCompeting,
    noiseLike,
    transientLike,
    modulatedPitch,
    untracked
};

enum class PitchStrategy
{
    phaseVocoder,
    psola,
    wsola,
    rubberBand,
    spectralFallback,
    dryWetSafety
};

struct EngineParams
{
    float shiftSemitones = 0.0f;
    float dryWet = 1.0f;
    bool safeMode = true;
};

struct SignalFeatures
{
    float rms = 0.0f;
    float peak = 0.0f;
    float zeroCrossingRate = 0.0f;
    float transientScore = 0.0f;
    float periodicity = 0.0f;
    float modulationScore = 0.0f;
    float observerDisagreement = 0.0f;
};

struct Decision
{
    SignalState state = SignalState::untracked;
    PitchStrategy strategy = PitchStrategy::rubberBand;
    bool safeModeActive = false;
    float effectiveDryWet = 1.0f;
    float effectiveShiftSemitones = 0.0f;
};

class AdaptivePitchEngine
{
public:
    void prepare(double sampleRate, int maxBlockSize);
    SignalFeatures analyzeBlock(const float* samples, int numSamples) const;
    Decision decide(const SignalFeatures& features, const EngineParams& params) const;

    static const char* toString(SignalState state);
    static const char* toString(PitchStrategy strategy);

private:
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;

    SignalState classify(const SignalFeatures& features) const;
};

} // namespace psbsl
