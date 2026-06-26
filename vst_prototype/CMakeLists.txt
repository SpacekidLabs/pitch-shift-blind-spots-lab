cmake_minimum_required(VERSION 3.22)

project(PitchShiftBlindSpotsVST VERSION 0.3.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSBSL_BUILD_PLUGIN "Build the JUCE VST3 plugin target" ON)
option(PSBSL_BUILD_ENGINE_SMOKE_TEST "Build the standalone adaptive-engine smoke test" ON)
option(PSBSL_ENABLE_RUBBERBAND "Enable the Rubber Band backend when the SDK/library is available" ON)
set(JUCE_SOURCE_DIR "" CACHE PATH "Path to a local JUCE source checkout")
set(RUBBERBAND_ROOT "" CACHE PATH "Path to a local Rubber Band install prefix")

if(JUCE_SOURCE_DIR)
    add_subdirectory("${JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE")
else()
    find_package(JUCE CONFIG REQUIRED)
endif()

add_library(psbsl_adaptive_engine
    Source/AdaptivePitchEngine.cpp
    Source/PhaseVocoderPitchShifter.cpp
    Source/RubberBandPitchShifter.cpp
    Source/SimplePitchShifter.cpp
)
target_include_directories(psbsl_adaptive_engine PUBLIC Source)
target_link_libraries(psbsl_adaptive_engine PUBLIC juce::juce_dsp)

if(PSBSL_ENABLE_RUBBERBAND)
    find_path(RUBBERBAND_INCLUDE_DIR
        NAMES rubberband/RubberBandStretcher.h
        HINTS "${RUBBERBAND_ROOT}/include"
    )
    find_library(RUBBERBAND_LIBRARY
        NAMES rubberband
        HINTS "${RUBBERBAND_ROOT}/lib"
    )
endif()

if(RUBBERBAND_INCLUDE_DIR AND RUBBERBAND_LIBRARY)
    message(STATUS "Rubber Band backend enabled: ${RUBBERBAND_LIBRARY}")
    target_compile_definitions(psbsl_adaptive_engine PUBLIC PSBSL_HAS_RUBBERBAND=1)
    target_include_directories(psbsl_adaptive_engine PUBLIC "${RUBBERBAND_INCLUDE_DIR}")
    target_link_libraries(psbsl_adaptive_engine PUBLIC "${RUBBERBAND_LIBRARY}")
else()
    message(STATUS "Rubber Band backend unavailable; building honest fallback slot")
    target_compile_definitions(psbsl_adaptive_engine PUBLIC PSBSL_HAS_RUBBERBAND=0)
endif()

if(PSBSL_BUILD_ENGINE_SMOKE_TEST)
    add_executable(psbsl_engine_smoke_test
        tests/engine_smoke_test.cpp
    )
    target_link_libraries(psbsl_engine_smoke_test PRIVATE psbsl_adaptive_engine)
endif()

if(PSBSL_BUILD_PLUGIN)
    juce_add_plugin(PitchShiftBlindSpots
        COMPANY_NAME "SpaceKidLabs"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD TRUE
        PLUGIN_MANUFACTURER_CODE SpKl
        PLUGIN_CODE Psbs
        FORMATS VST3
        PRODUCT_NAME "Pitch Shift Blind Spots"
    )
    juce_generate_juce_header(PitchShiftBlindSpots)

    target_sources(PitchShiftBlindSpots PRIVATE
        Source/PluginEditor.cpp
        Source/PluginProcessor.cpp
    )

    target_compile_definitions(PitchShiftBlindSpots PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
    )

    target_link_libraries(PitchShiftBlindSpots PRIVATE
        psbsl_adaptive_engine
        juce::juce_audio_utils
    )
endif()
