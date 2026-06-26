#pragma once

#include <JuceHeader.h>

#include <vector>

namespace psbsl
{

class PhaseVocoderPitchShifter
{
public:
    void prepare(double sampleRate, int maxChannels);
    void reset();

    void processChannel(const float* input, float* output, int numSamples, int channel, float shiftSemitones);

private:
    struct ChannelState
    {
        std::vector<float> inputFifo;
        std::vector<float> outputFifo;
        std::vector<float> outputAccum;
        std::vector<float> fftData;
        std::vector<float> window;
        std::vector<float> lastPhase;
        std::vector<float> sumPhase;
        std::vector<float> analysisMag;
        std::vector<float> analysisFreq;
        std::vector<float> synthMag;
        std::vector<float> synthFreq;
        int rover = 0;
    };

    void processFrame(ChannelState& state, float pitchRatio);
    void clearState(ChannelState& state);

    double sampleRate_ = 44100.0;
    int fftOrder_ = 11;
    int fftSize_ = 2048;
    int hopSize_ = 512;
    int overlap_ = 4;
    int inputLatency_ = 1536;
    juce::dsp::FFT fft_ { fftOrder_ };
    std::vector<ChannelState> channels_;
};

} // namespace psbsl
