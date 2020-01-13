cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(renderer)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer PRIVATE spdlog::spdlog)

if (NOT PROF)
  target_sources(renderer PRIVATE src/render.hpp src/render.cu)
  set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(render_cpu_ref SHARED IMPORTED)
  target_link_libraries(render_cpu_ref INTERFACE tbb)
  set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)
else()
  target_sources(renderer PRIVATE src/render.hpp src/render_prof.cu)
  set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(render_cpu_ref SHARED IMPORTED)
  target_link_libraries(render_cpu_ref INTERFACE tbb)
  set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)
endif()

#add_library(image_processor src/image_processor.cpp)
#target_compile_features(image_processor PUBLIC cxx_std_17)

add_executable(morpho_cpu src/morpho_cpu.cpp src/image_processor.cpp src/image_processor.hh)
target_compile_features(morpho_cpu PUBLIC cxx_std_17)
target_link_libraries(morpho_cpu PRIVATE render_cpu_ref PNG::PNG spdlog::spdlog renderer)

add_executable(morpho_gpu src/morpho.cpp src/image_processor.cpp src/image_processor.hh)
target_compile_features(morpho_gpu PUBLIC cxx_std_17)
target_link_libraries(morpho_gpu PRIVATE render_cpu_ref PNG::PNG spdlog::spdlog renderer)

add_executable(bench_src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE render_cpu_ref spdlog::spdlog benchmark renderer)
