cmake_minimum_required(VERSION 3.16)

project(parallel_ntt_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
find_package(Threads REQUIRED)

add_executable(ntt_bench src/ntt_bench.cpp)
target_link_libraries(ntt_bench PRIVATE Threads::Threads)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ntt_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ntt_bench PRIVATE HAVE_OPENMP=1)
endif()

if(MSVC)
    target_compile_options(ntt_bench PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(ntt_bench PRIVATE -O3 -march=native -fopenmp-simd -Wall -Wextra)
endif()
