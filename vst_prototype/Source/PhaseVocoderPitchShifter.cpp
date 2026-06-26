#include "PhaseVocoderPitchShifter.h"

#include <algorithm>
#include <cmath>

namespace psbsl
{

namespace
{
constexpr float pi = 3.14159265358979323846f;
constexpr float twoPi = 2.0f * pi;

float princarg(float phase)
{
    while (phase < -pi)
        phase += twoPi;
    while (phase >= pi)
        phase -= twoPi;
    return phase;
}
} // namespace

void PhaseVocoderPitchShifter::prepare(double sampleRate, int maxChannels)
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 44100.0;
    inputLatency_ = fftSize_ - hopSize_;
    channels_.resize(static_cast<std::size_t>(std::max(1, maxChannels)));

    for (auto& channel : channels_)
    {
        channel.inputFifo.assign(static_cast<std::size_t>(fftSize_), 0.0f);
        channel.outputFifo.assign(static_cast<std::size_t>(fftSize_), 0.0f);
        channel.outputAccum.assign(static_cast<std::size_t>(fftSize_ * 2), 0.0f);
        channel.fftData.assign(static_cast<std::size_t>(fftSize_ * 2), 0.0f);
        channel.window.assign(static_cast<std::size_t>(fftSize_), 0.0f);
        channel.lastPhase.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);
        channel.sumPhase.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);
        channel.analysisMag.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);
        channel.analysisFreq.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);
        channel.synthMag.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);
        channel.synthFreq.assign(static_cast<std::size_t>(fftSize_ / 2 + 1), 0.0f);

        for (int i = 0; i < fftSize_; ++i)
            channel.window[static_cast<std::size_t>(i)] = 0.5f - 0.5f * std::cos(twoPi * static_cast<float>(i) / static_cast<float>(fftSize_));

        clearState(channel);
    }
}

void PhaseVocoderPitchShifter::reset()
{
    for (auto& channel : channels_)
        clearState(channel);
}

void PhaseVocoderPitchShifter::processChannel(const float* input, float* output, int numSamples, int channel, float shiftSemitones)
{
    if (input == nullptr || output == nullptr || numSamples <= 0)
        return;

    if (channel < 0 || channel >= static_cast<int>(channels_.size()))
    {
        std::copy(input, input + numSamples, output);
        return;
    }

    auto& state = channels_[static_cast<std::size_t>(channel)];
    const float pitchRatio = std::pow(2.0f, shiftSemitones / 12.0f);

    if (std::abs(shiftSemitones) < 0.01f || pitchRatio <= 0.0f)
    {
        std::copy(input, input + numSamples, output);
        return;
    }

    for (int i = 0; i < numSamples; ++i)
    {
        state.inputFifo[static_cast<std::size_t>(state.rover)] = input[i];
        output[i] = state.outputFifo[static_cast<std::size_t>(state.rover - inputLatency_)];
        state.rover += 1;

        if (state.rover >= fftSize_)
        {
            state.rover = inputLatency_;
            processFrame(state, pitchRatio);
        }
    }
}

void PhaseVocoderPitchShifter::processFrame(ChannelState& state, float pitchRatio)
{
    const auto halfSize = fftSize_ / 2;
    const auto freqPerBin = static_cast<float>(sampleRate_) / static_cast<float>(fftSize_);
    const auto expectedPhaseAdvance = twoPi * static_cast<float>(hopSize_) / static_cast<float>(fftSize_);

    std::fill(state.fftData.begin(), state.fftData.end(), 0.0f);
    for (int i = 0; i < fftSize_; ++i)
        state.fftData[static_cast<std::size_t>(i)] = state.inputFifo[static_cast<std::size_t>(i)] * state.window[static_cast<std::size_t>(i)];

    fft_.performRealOnlyForwardTransform(state.fftData.data());

    for (int k = 0; k <= halfSize; ++k)
    {
        const auto real = state.fftData[static_cast<std::size_t>(2 * k)];
        const auto imag = state.fftData[static_cast<std::size_t>(2 * k + 1)];
        const auto magnitude = 2.0f * std::sqrt(real * real + imag * imag);
        const auto phase = std::atan2(imag, real);

        auto phaseDelta = phase - state.lastPhase[static_cast<std::size_t>(k)];
        state.lastPhase[static_cast<std::size_t>(k)] = phase;
        phaseDelta = princarg(phaseDelta - static_cast<float>(k) * expectedPhaseAdvance);

        const auto trueFrequency = (static_cast<float>(k) + phaseDelta * static_cast<float>(overlap_) / twoPi) * freqPerBin;
        state.analysisMag[static_cast<std::size_t>(k)] = magnitude;
        state.analysisFreq[static_cast<std::size_t>(k)] = trueFrequency;
    }

    std::fill(state.synthMag.begin(), state.synthMag.end(), 0.0f);
    std::fill(state.synthFreq.begin(), state.synthFreq.end(), 0.0f);

    for (int k = 0; k <= halfSize; ++k)
    {
        const auto target = static_cast<int>(std::round(static_cast<float>(k) * pitchRatio));
        if (target <= halfSize)
        {
            state.synthMag[static_cast<std::size_t>(target)] += state.analysisMag[static_cast<std::size_t>(k)];
            state.synthFreq[static_cast<std::size_t>(target)] = state.analysisFreq[static_cast<std::size_t>(k)] * pitchRatio;
        }
    }

    std::fill(state.fftData.begin(), state.fftData.end(), 0.0f);
    for (int k = 0; k <= halfSize; ++k)
    {
        const auto magnitude = state.synthMag[static_cast<std::size_t>(k)];
        auto phaseDelta = state.synthFreq[static_cast<std::size_t>(k)] / freqPerBin - static_cast<float>(k);
        phaseDelta = phaseDelta * twoPi / static_cast<float>(overlap_);
        phaseDelta += static_cast<float>(k) * expectedPhaseAdvance;

        state.sumPhase[static_cast<std::size_t>(k)] += phaseDelta;
        const auto phase = state.sumPhase[static_cast<std::size_t>(k)];
        state.fftData[static_cast<std::size_t>(2 * k)] = magnitude * std::cos(phase);
        state.fftData[static_cast<std::size_t>(2 * k + 1)] = magnitude * std::sin(phase);
    }

    fft_.performRealOnlyInverseTransform(state.fftData.data());

    const auto scale = 1.0f / (static_cast<float>(fftSize_) * static_cast<float>(overlap_) * 0.5f);
    for (int i = 0; i < fftSize_; ++i)
    {
        state.outputAccum[static_cast<std::size_t>(i)] +=
            state.window[static_cast<std::size_t>(i)] * state.fftData[static_cast<std::size_t>(i)] * scale;
    }

    for (int i = 0; i < hopSize_; ++i)
        state.outputFifo[static_cast<std::size_t>(i)] = state.outputAccum[static_cast<std::size_t>(i)];

    std::move(state.outputAccum.begin() + hopSize_, state.outputAccum.end(), state.outputAccum.begin());
    std::fill(state.outputAccum.end() - hopSize_, state.outputAccum.end(), 0.0f);

    std::move(state.inputFifo.begin() + hopSize_, state.inputFifo.end(), state.inputFifo.begin());
    std::fill(state.inputFifo.end() - hopSize_, state.inputFifo.end(), 0.0f);
}

void PhaseVocoderPitchShifter::clearState(ChannelState& state)
{
    std::fill(state.inputFifo.begin(), state.inputFifo.end(), 0.0f);
    std::fill(state.outputFifo.begin(), state.outputFifo.end(), 0.0f);
    std::fill(state.outputAccum.begin(), state.outputAccum.end(), 0.0f);
    std::fill(state.fftData.begin(), state.fftData.end(), 0.0f);
    std::fill(state.lastPhase.begin(), state.lastPhase.end(), 0.0f);
    std::fill(state.sumPhase.begin(), state.sumPhase.end(), 0.0f);
    state.rover = inputLatency_;
}

} // namespace psbsl
