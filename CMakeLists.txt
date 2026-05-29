cmake_minimum_required(VERSION 3.14)
project(Einstein_Summation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    if(POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
            "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,\
$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,\
$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif()
    add_compile_options(/W3)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    add_compile_options(-Wall -Wshadow -Wconversion)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wextra -pedantic -fcolor-diagnostics -stdlib=libc++)
        add_link_options(-stdlib=libc++)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wextra -pedantic)
    endif()
endif()

find_package(Boost REQUIRED)

include(FetchContent)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(mdspan googletest benchmark)


add_library(einsum INTERFACE
    ${CMAKE_SOURCE_DIR}/include/einsum.hpp
    ${CMAKE_SOURCE_DIR}/include/input_handler.hpp
    ${CMAKE_SOURCE_DIR}/include/labels.hpp
    ${CMAKE_SOURCE_DIR}/include/matrices.hpp
    ${CMAKE_SOURCE_DIR}/include/printers.hpp
)
target_include_directories(einsum INTERFACE
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${mdspan_SOURCE_DIR}/include
)

# ── Executables ───────────────────────────────────────────────────────────────
add_executable(main main.cpp)
target_link_libraries(main PRIVATE einsum)

add_executable(tests tests/tests.cpp)
target_link_libraries(tests PRIVATE einsum gtest_main)

add_executable(measure benchmark/benchmark.cpp)
target_link_libraries(measure PRIVATE einsum benchmark::benchmark)

enable_testing()
add_test(NAME tests COMMAND tests)
