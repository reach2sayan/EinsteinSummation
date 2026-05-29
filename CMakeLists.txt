cmake_minimum_required(VERSION 3.25)
project(EinsteinSummation LANGUAGES CXX)

# ── Global settings ───────────────────────────────────────────────────────────

# Use ccache if available to cache compiled objects across rebuilds.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "ccache found: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for clangd / IDE language servers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Debug when no build type is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
    STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ── Warning flags (used only on OUR targets — never on fetched deps) ──────────
set(EINSUM_WARN_FLAGS -Wall -Wextra -Wshadow -Wconversion)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND EINSUM_WARN_FLAGS -pedantic -fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  list(APPEND EINSUM_WARN_FLAGS -pedantic)
endif()

# Heavy headers shared across all translation units — compiled once as a PCH.
set(EINSUM_PCH_HEADERS
  <algorithm>
  <iterator>
  <ranges>
  <string_view>
  <tuple>
  <type_traits>
  <vector>
  <boost/hana.hpp>
  <experimental/mdspan>
)

function(einsum_target_pch target)
  target_precompile_headers(${target} PRIVATE ${EINSUM_PCH_HEADERS})
endfunction()

function(einsum_target_compile_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W3)
  else()
    target_compile_options(${target} PRIVATE ${EINSUM_WARN_FLAGS})
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:-O3 -march=native>
      $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    )
  endif()
endfunction()

# ── Dependencies ──────────────────────────────────────────────────────────────
include(FetchContent)

# Suppress CMP0167 (FindBoost removed in CMake 3.30+).
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)

# Disable tests, docs, and install targets in all fetched dependencies.
set(BOOST_HANA_ENABLE_TESTING    OFF CACHE BOOL "" FORCE)
set(BOOST_HANA_ENABLE_DOCS       OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS       OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST                OFF CACHE BOOL "" FORCE)

FetchContent_Declare(hana
  GIT_REPOSITORY https://github.com/boostorg/hana.git
  GIT_TAG        v1.7.0
  GIT_SHALLOW    TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG        stable
  GIT_SHALLOW    TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
  GIT_SHALLOW    TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(hana mdspan benchmark)

# ── Library (header-only) ─────────────────────────────────────────────────────
add_library(einsum INTERFACE)

# FILE_SET declares headers for IDE navigation and future install rules.
target_sources(einsum INTERFACE
  FILE_SET HEADERS
  BASE_DIRS ${CMAKE_SOURCE_DIR}/include
  FILES
    include/einsum.hpp
    include/einsum_helpers.hpp
    include/unary_einsum.hpp
    include/n_einsum.hpp
    include/input_handler.hpp
    include/labels.hpp
    include/matrices.hpp
    include/printers.hpp
)

target_include_directories(einsum INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# Mark fetched-dependency includes as SYSTEM so their headers never emit
# warnings when compiled as part of this project.
target_include_directories(einsum SYSTEM INTERFACE
  ${hana_SOURCE_DIR}/include
  ${mdspan_SOURCE_DIR}/include
)

# Propagate C++23 requirement to all consumers.
target_compile_features(einsum INTERFACE cxx_std_23)

# ── Executables ───────────────────────────────────────────────────────────────
if(EXISTS ${CMAKE_SOURCE_DIR}/main.cpp)
  add_executable(main main.cpp)
  target_link_libraries(main PRIVATE einsum)
  einsum_target_compile_options(main)
  einsum_target_pch(main)
endif()

add_executable(einsum_benchmark benchmark/benchmark.cpp)
target_link_libraries(einsum_benchmark PRIVATE einsum benchmark::benchmark)
einsum_target_compile_options(einsum_benchmark)
einsum_target_pch(einsum_benchmark)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

# Boost.Test (header-only, included variant) requires a system Boost install
# because it pulls in more Boost headers than the boostorg/hana FetchContent provides.
find_package(Boost QUIET)
if(Boost_FOUND)
  add_executable(einsum_tests tests/tests.cpp)
  target_link_libraries(einsum_tests PRIVATE einsum)
  # Mark Boost includes as SYSTEM to suppress warnings from Boost headers.
  target_include_directories(einsum_tests SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
  einsum_target_compile_options(einsum_tests)
  einsum_target_pch(einsum_tests)
  add_test(NAME einsum_tests COMMAND einsum_tests)
else()
  message(STATUS "Boost not found — skipping Boost.Test suite (tests/tests.cpp)")
endif()
