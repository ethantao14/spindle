include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
  URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SPINDLE_REFERENCE_DIR "${PROJECT_SOURCE_DIR}/build/reference"
    CACHE PATH "Directory holding dumps from tools/reference.py")

set(SPINDLE_TEST_SOURCES
    version_test.cpp
    cli_test.cpp
    reference_loader_test.cpp
    generate_cli_spec_test.cpp
    sampling_spec_test.cpp
    generate_sampling_spec_test.cpp
    generate_sampling_cli_spec_test.cpp
    gpt2_reference_test.cpp
    gpt2_spec_test.cpp
    gpt2_test.cpp
    tensor_test.cpp
    matmul_test.cpp
    matmul_spec_test.cpp
    ops_test.cpp
    safetensors_spec_test.cpp
    safetensors_test.cpp
    tokenizer_test.cpp
    unicode_test.cpp
    support/reference.cpp)
if(SPINDLE_SANITIZE)
  list(APPEND SPINDLE_TEST_SOURCES sanitizer_test.cpp)
endif()

add_executable(spindle_tests ${SPINDLE_TEST_SOURCES})
target_link_libraries(spindle_tests PRIVATE spindle_core GTest::gtest_main GTest::gmock nlohmann_json::nlohmann_json)
target_compile_options(spindle_tests PRIVATE ${SPINDLE_WARNINGS})
target_include_directories(spindle_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(spindle_tests PRIVATE SPINDLE_CLI_PATH="$<TARGET_FILE:spindle>"
                                                 SPINDLE_REFERENCE_DIR="${SPINDLE_REFERENCE_DIR}")
add_dependencies(spindle_tests spindle)

include(GoogleTest)
gtest_discover_tests(spindle_tests TEST_FILTER "-Tokenizer.LongSinglePieceEncodesQuickly")
add_test(NAME Tokenizer.LongSinglePieceEncodesQuickly
         COMMAND spindle_tests --gtest_filter=Tokenizer.LongSinglePieceEncodesQuickly)
set_tests_properties(Tokenizer.LongSinglePieceEncodesQuickly PROPERTIES TIMEOUT 30)
