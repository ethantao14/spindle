cmake_minimum_required(VERSION 3.25)
project(spindle VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPINDLE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(SPINDLE_SANITIZE)
  # No recovery: any UB report aborts the test, so it cannot pass silently.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

set(SPINDLE_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)

configure_file(src/version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

add_library(spindle_core src/ops.cpp src/tensor.cpp ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
target_include_directories(spindle_core PUBLIC include)
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
