#include "SimplePitchShifter.h"

#include <algorithm>
#include <cmath>

namespace psbsl
{

namespace
{
constexpr float pi = 3.14159265358979323846f;

float wrap01(float value)
{
    value -= std::floor(value);
    return value < 0.0f ? value + 1.0f : value;
}

float hann(float phase)
{
    return 0.5f - 0.5f * std::cos(2.0f * pi * wrap01(phase));
}
} // namespace

void SimplePitchShifter::prepare(double sampleRate, int maxChannels)
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 44100.0;
    windowSamples_ = std::max(512, static_cast<int>(sampleRate_ * 0.080));
    delaySize_ = windowSamples_ * 2 + 8;
    channels_.resize(static_cast<std::size_t>(std::max(1, maxChannels)));

    for (auto& channel : channels_)
    {
        channel.delay.assign(static_cast<std::size_t>(delaySize_), 0.0f);
        channel.writeIndex = 0;
        channel.phase = 0.0f;
    }
}

void SimplePitchShifter::reset()
{
    for (auto& channel : channels_)
    {
        std::fill(channel.delay.begin(), channel.delay.end(), 0.0f);
        channel.writeIndex = 0;
        channel.phase = 0.0f;
    }
}

void SimplePitchShifter::processChannel(const float* input, float* output, int numSamples, int channel, float shiftSemitones)
{
    if (input == nullptr || output == nullptr || numSamples <= 0)
        return;

    if (channel < 0 || channel >= static_cast<int>(channels_.size()))
    {
        std::copy(input, input + numSamples, output);
        return;
    }

    auto& state = channels_[static_cast<std::size_t>(channel)];
    const float ratio = std::pow(2.0f, shiftSemitones / 12.0f);

    if (std::abs(shiftSemitones) < 0.01f || ratio <= 0.0f)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            writeSample(state, input[i]);
            output[i] = input[i];
        }
        return;
    }

    const bool pitchUp = ratio >= 1.0f;
    const float phaseIncrement = std::abs(ratio - 1.0f) / static_cast<float>(windowSamples_);
    const float window = static_cast<float>(windowSamples_);

    for (int i = 0; i < numSamples; ++i)
    {
        writeSample(state, input[i]);

        const float phaseA = state.phase;
        const float phaseB = wrap01(state.phase + 0.5f);
        const float delayA = pitchUp ? (1.0f - phaseA) * window : phaseA * window;
        const float delayB = pitchUp ? (1.0f - phaseB) * window : phaseB * window;
        const float gainA = hann(phaseA);
        const float gainB = hann(phaseB);
        const float sum = std::max(1.0e-6f, gainA + gainB);

        output[i] = (readDelayed(state, delayA) * gainA + readDelayed(state, delayB) * gainB) / sum;
        state.phase = wrap01(state.phase + phaseIncrement);
    }
}

float SimplePitchShifter::readDelayed(const ChannelState& state, float delaySamples) const
{
    float readPosition = static_cast<float>(state.writeIndex - 1) - delaySamples;
    while (readPosition < 0.0f)
        readPosition += static_cast<float>(delaySize_);

    const int index0 = static_cast<int>(std::floor(readPosition)) % delaySize_;
    const int index1 = (index0 + 1) % delaySize_;
    const float frac = readPosition - std::floor(readPosition);
    const float a = state.delay[static_cast<std::size_t>(index0)];
    const float b = state.delay[static_cast<std::size_t>(index1)];
    return a + (b - a) * frac;
}

void SimplePitchShifter::writeSample(ChannelState& state, float sample)
{
    state.delay[static_cast<std::size_t>(state.writeIndex)] = sample;
    state.writeIndex = (state.writeIndex + 1) % delaySize_;
}

} // namespace psbsl
