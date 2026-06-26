#pragma once

#include <memory>
#include <vector>

namespace psbsl
{

class RubberBandPitchShifter
{
public:
    RubberBandPitchShifter();
    ~RubberBandPitchShifter();

    void prepare(double sampleRate, int channels);
    void reset();

    bool isAvailable() const;
    void processBlock(
        const float* const* input,
        float* const* output,
        int numChannels,
        int numSamples,
        float shiftSemitones);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace psbsl
