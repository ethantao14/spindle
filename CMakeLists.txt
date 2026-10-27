cmake_minimum_required(VERSION 3.25)
project(spindle VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPINDLE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

option(SPINDLE_TSAN "Build with ThreadSanitizer" OFF)
if(SPINDLE_TSAN AND SPINDLE_SANITIZE)
  message(FATAL_ERROR "ThreadSanitizer and AddressSanitizer cannot be combined")
endif()
if(SPINDLE_TSAN)
  add_compile_options(-fsanitize=thread -O1 -g)
  add_link_options(-fsanitize=thread)
endif()

if(SPINDLE_SANITIZE)
  # No recovery: any UB report aborts the test, so it cannot pass silently. -O1, as
  # ASan's docs suggest, keeps full checking while running a GPT-2 forward pass about 9x faster than -O0.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
                      -O1)
  add_link_options(-fsanitize=address,undefined)
endif()

set(SPINDLE_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)

configure_file(src/version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

include(FetchContent)
# Header-only; parses vocab.json here, and safetensors headers and model configs later.
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

add_library(spindle_core src/thread_pool.cpp src/gpt2.cpp src/sampling.cpp src/ops.cpp
                         src/safetensors.cpp src/tensor.cpp src/tokenizer.cpp src/unicode.cpp
                         ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
target_include_directories(spindle_core PUBLIC include)
target_link_libraries(spindle_core PRIVATE nlohmann_json::nlohmann_json PUBLIC Threads::Threads)
target_compile_options(spindle_core PRIVATE ${SPINDLE_WARNINGS})

add_executable(spindle src/main.cpp)
target_link_libraries(spindle PRIVATE spindle_core)
target_compile_options(spindle PRIVATE ${SPINDLE_WARNINGS})

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

option(SPINDLE_BENCHMARKS "Build the Google Benchmark targets" ON)
if(SPINDLE_BENCHMARKS)
  add_subdirectory(bench)
endif()
