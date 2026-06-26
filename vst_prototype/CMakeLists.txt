cmake_minimum_required(VERSION 3.22)

project(PitchShiftBlindSpotsVST VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSBSL_BUILD_PLUGIN "Build the JUCE VST3 plugin target" ON)
option(PSBSL_BUILD_ENGINE_SMOKE_TEST "Build the standalone adaptive-engine smoke test" ON)
set(JUCE_SOURCE_DIR "" CACHE PATH "Path to a local JUCE source checkout")

add_library(psbsl_adaptive_engine
    Source/AdaptivePitchEngine.cpp
)
target_include_directories(psbsl_adaptive_engine PUBLIC Source)

if(PSBSL_BUILD_ENGINE_SMOKE_TEST)
    add_executable(psbsl_engine_smoke_test
        tests/engine_smoke_test.cpp
    )
    target_link_libraries(psbsl_engine_smoke_test PRIVATE psbsl_adaptive_engine)
endif()

if(PSBSL_BUILD_PLUGIN)
    if(JUCE_SOURCE_DIR)
        add_subdirectory("${JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE")
    else()
        find_package(JUCE CONFIG REQUIRED)
    endif()

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
