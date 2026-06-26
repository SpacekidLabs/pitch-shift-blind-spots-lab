#include "RubberBandPitchShifter.h"

#include <algorithm>
#include <cmath>

#if PSBSL_HAS_RUBBERBAND
#include <rubberband/RubberBandStretcher.h>
#endif

namespace psbsl
{

struct RubberBandPitchShifter::Impl
{
#if PSBSL_HAS_RUBBERBAND
    std::unique_ptr<RubberBand::RubberBandStretcher> stretcher;
    std::vector<std::vector<float>> fallback;
#endif
    double sampleRate = 44100.0;
    int channels = 0;
};

RubberBandPitchShifter::RubberBandPitchShifter()
    : impl_(std::make_unique<Impl>())
{
}

RubberBandPitchShifter::~RubberBandPitchShifter() = default;

void RubberBandPitchShifter::prepare(double sampleRate, int channels)
{
    impl_->sampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
    impl_->channels = std::max(1, channels);

#if PSBSL_HAS_RUBBERBAND
    using Stretcher = RubberBand::RubberBandStretcher;
    const auto options =
        Stretcher::OptionProcessRealTime
        | Stretcher::OptionChannelsTogether
        | Stretcher::OptionTransientsCrisp
        | Stretcher::OptionEngineFiner;

    impl_->stretcher = std::make_unique<Stretcher>(
        static_cast<size_t>(impl_->sampleRate),
        static_cast<size_t>(impl_->channels),
        options,
        1.0,
        1.0);
    impl_->fallback.assign(static_cast<std::size_t>(impl_->channels), {});
#endif
}

void RubberBandPitchShifter::reset()
{
#if PSBSL_HAS_RUBBERBAND
    if (impl_->stretcher)
        impl_->stretcher->reset();
#endif
}

bool RubberBandPitchShifter::isAvailable() const
{
#if PSBSL_HAS_RUBBERBAND
    return impl_->stretcher != nullptr;
#else
    return false;
#endif
}

void RubberBandPitchShifter::processBlock(
    const float* const* input,
    float* const* output,
    int numChannels,
    int numSamples,
    float shiftSemitones)
{
    if (input == nullptr || output == nullptr || numChannels <= 0 || numSamples <= 0)
        return;

#if PSBSL_HAS_RUBBERBAND
    if (!impl_->stretcher || numChannels != impl_->channels)
    {
        for (int channel = 0; channel < numChannels; ++channel)
            std::copy(input[channel], input[channel] + numSamples, output[channel]);
        return;
    }

    impl_->stretcher->setPitchScale(std::pow(2.0, static_cast<double>(shiftSemitones) / 12.0));
    impl_->stretcher->setTimeRatio(1.0);
    impl_->stretcher->process(input, static_cast<size_t>(numSamples), false);

    const auto available = static_cast<int>(impl_->stretcher->available());
    if (available >= numSamples)
    {
        impl_->stretcher->retrieve(output, static_cast<size_t>(numSamples));
        return;
    }

    for (int channel = 0; channel < numChannels; ++channel)
        std::copy(input[channel], input[channel] + numSamples, output[channel]);
#else
    for (int channel = 0; channel < numChannels; ++channel)
        std::copy(input[channel], input[channel] + numSamples, output[channel]);
#endif
}

} // namespace psbsl
