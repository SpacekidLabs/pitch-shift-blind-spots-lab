#pragma once

#include <vector>

namespace psbsl
{

class SimplePitchShifter
{
public:
    void prepare(double sampleRate, int maxChannels);
    void reset();

    void processChannel(const float* input, float* output, int numSamples, int channel, float shiftSemitones);

private:
    struct ChannelState
    {
        std::vector<float> delay;
        int writeIndex = 0;
        float phase = 0.0f;
    };

    float readDelayed(const ChannelState& state, float delaySamples) const;
    void writeSample(ChannelState& state, float sample);

    double sampleRate_ = 44100.0;
    int windowSamples_ = 2048;
    int delaySize_ = 8192;
    std::vector<ChannelState> channels_;
};

} // namespace psbsl
