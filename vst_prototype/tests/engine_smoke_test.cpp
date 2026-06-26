#include "AdaptivePitchEngine.h"

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

    return 0;
}
