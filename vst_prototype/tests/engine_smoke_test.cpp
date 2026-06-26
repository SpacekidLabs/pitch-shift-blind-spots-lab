#include "AdaptivePitchEngine.h"
#include "SimplePitchShifter.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace
{
constexpr float pi = 3.14159265358979323846f;

std::vector<float> makeSine(int sampleRate, int size, float frequency)
{
    std::vector<float> block(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
        block[static_cast<std::size_t>(i)] = 0.35f * std::sin(2.0f * pi * frequency * static_cast<float>(i) / sampleRate);
    return block;
}

std::vector<float> makeTransient(int size)
{
    std::vector<float> block(static_cast<std::size_t>(size), 0.0f);
    if (!block.empty())
        block[0] = 1.0f;
    return block;
}

std::vector<float> makeNoise(int size)
{
    std::vector<float> block(static_cast<std::size_t>(size));
    unsigned int state = 1;
    for (auto& sample : block)
    {
        state = 1664525u * state + 1013904223u;
        const auto normalized = static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
        sample = 0.15f * normalized;
    }
    return block;
}

void printDecision(const char* name, psbsl::AdaptivePitchEngine& engine, const std::vector<float>& block)
{
    psbsl::EngineParams params;
    params.shiftSemitones = 7.0f;
    params.safeMode = true;

    const auto features = engine.analyzeBlock(block.data(), static_cast<int>(block.size()));
    const auto decision = engine.decide(features, params);

    std::cout << name
              << " state=" << psbsl::AdaptivePitchEngine::toString(decision.state)
              << " strategy=" << psbsl::AdaptivePitchEngine::toString(decision.strategy)
              << " disagreement=" << features.observerDisagreement
              << " safe=" << (decision.safeModeActive ? "yes" : "no")
              << "\n";
}
} // namespace

int main()
{
    constexpr int sampleRate = 48000;
    constexpr int blockSize = 1024;

    psbsl::AdaptivePitchEngine engine;
    engine.prepare(sampleRate, blockSize);

    printDecision("sine", engine, makeSine(sampleRate, blockSize, 440.0f));
    printDecision("transient", engine, makeTransient(blockSize));
    printDecision("noise", engine, makeNoise(blockSize));

    psbsl::SimplePitchShifter shifter;
    shifter.prepare(sampleRate, 1);
    std::vector<float> shifted(static_cast<std::size_t>(blockSize), 0.0f);
    for (int block = 0; block < 12; ++block)
    {
        const auto sine = makeSine(sampleRate, blockSize, 440.0f);
        shifter.processChannel(sine.data(), shifted.data(), static_cast<int>(shifted.size()), 0, 12.0f);
    }

    double wetEnergy = 0.0;
    for (const auto sample : shifted)
        wetEnergy += static_cast<double>(sample) * sample;
    std::cout << "pitch_backend rms=" << std::sqrt(wetEnergy / static_cast<double>(shifted.size())) << "\n";

    return 0;
}
