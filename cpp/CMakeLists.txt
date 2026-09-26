cmake_minimum_required(VERSION 3.22)
project(InverseSynth VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
endif()

option(INVSYN_BUILD_PLUGIN "Build the JUCE plugin" ON)
option(INVSYN_BUILD_CLI "Build the headless CLI" ON)
set(INVSYN_MODEL_PATH "" CACHE FILEPATH "Optional inverse_synth.onnx to bundle inside the plugin")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Pre-extracted ONNX Runtime (leave empty to download)")
set(ORT_VERSION 1.20.1)

include(FetchContent)

# ------------------------------------------------------------ ONNX Runtime
# Headers are used at compile time; the shared library is only COPIED next to
# the plugin and loaded at runtime by full path (see core/Predictor.h).
if(NOT ONNXRUNTIME_ROOT)
  if(WIN32)
    set(_ort_pkg onnxruntime-win-x64-${ORT_VERSION}.zip)
  elseif(APPLE)
    set(_ort_pkg onnxruntime-osx-universal2-${ORT_VERSION}.tgz)
  else()
    set(_ort_pkg onnxruntime-linux-x64-${ORT_VERSION}.tgz)
  endif()
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_pkg}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime)
  set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
endif()

if(WIN32)
  set(ORT_LIB_SRC ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll)
  set(ORT_LIB_NAME onnxruntime.dll)
elseif(APPLE)
  set(ORT_LIB_SRC ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.${ORT_VERSION}.dylib)
  set(ORT_LIB_NAME libonnxruntime.dylib)
else()
  set(ORT_LIB_SRC ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ORT_VERSION})
  set(ORT_LIB_NAME libonnxruntime.so)
endif()

# ------------------------------------------------------------ core (no JUCE)
add_library(invsynth_core STATIC
  core/SynthVoice.cpp core/Effects.cpp core/Features.cpp
  core/Finetuner.cpp core/Predictor.cpp core/Pipeline.cpp)
target_include_directories(invsynth_core PUBLIC core ${ONNXRUNTIME_ROOT}/include)
if(MSVC)
  target_compile_options(invsynth_core PRIVATE /O2 /fp:precise)
else()
  target_compile_options(invsynth_core PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(invsynth_core PUBLIC ${CMAKE_DL_LIBS})
endif()
find_package(Threads REQUIRED)
target_link_libraries(invsynth_core PUBLIC Threads::Threads)

if(INVSYN_BUILD_CLI)
  add_executable(invsynth_cli tools/invsynth_cli.cpp)
  target_link_libraries(invsynth_cli PRIVATE invsynth_core)
  add_custom_command(TARGET invsynth_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_LIB_SRC} $<TARGET_FILE_DIR:invsynth_cli>/${ORT_LIB_NAME})
endif()

# ------------------------------------------------------------ JUCE plugin
if(INVSYN_BUILD_PLUGIN)
  FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(JUCE)

  set(_formats VST3 Standalone)
  if(APPLE)
    list(APPEND _formats AU)
  endif()

  juce_add_plugin(InverseSynth
    COMPANY_NAME "InverseSynth"
    PRODUCT_NAME "InverseSynth"
    PLUGIN_MANUFACTURER_CODE Invs
    PLUGIN_CODE Isyn
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Instrument Synth
    AU_MAIN_TYPE kAudioUnitType_MusicDevice
    FORMATS ${_formats})

  target_sources(InverseSynth PRIVATE plugin/PluginProcessor.cpp plugin/PluginEditor.cpp)
  target_compile_definitions(InverseSynth PUBLIC
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_USE_MP3AUDIOFORMAT=1 JUCE_DISPLAY_SPLASH_SCREEN=0)
  target_link_libraries(InverseSynth
    PRIVATE invsynth_core juce::juce_audio_utils juce::juce_audio_formats juce::juce_dsp
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags
           juce::juce_recommended_warning_flags)

  # Put the ORT library (and optionally the model) inside every built format.
  foreach(fmt IN LISTS _formats)
    set(tgt InverseSynth_${fmt})
    if(NOT TARGET ${tgt})
      continue()
    endif()
    if(APPLE)
      set(_libdir $<TARGET_FILE_DIR:${tgt}>/../Frameworks)
      set(_resdir $<TARGET_FILE_DIR:${tgt}>/../Resources)
    else()
      set(_libdir $<TARGET_FILE_DIR:${tgt}>)
      set(_resdir $<TARGET_FILE_DIR:${tgt}>)
    endif()
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_libdir}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_LIB_SRC} ${_libdir}/${ORT_LIB_NAME})
    if(INVSYN_MODEL_PATH)
      add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_resdir}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INVSYN_MODEL_PATH} ${_resdir}/inverse_synth.onnx)
    endif()
    if(APPLE)   # bundle contents changed after JUCE signed it: re-sign ad hoc
      add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND codesign --force --deep --sign - $<TARGET_BUNDLE_DIR:${tgt}>)
    endif()
  endforeach()
endif()
