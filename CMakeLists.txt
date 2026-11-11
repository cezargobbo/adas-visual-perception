cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(adasvp LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ADASVP_BUILD_TESTS "Build the C++ tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

add_library(adasvp SHARED
  src/trt/trt_engine.cpp
  src/preprocess/letterbox.cpp
  src/models/yolo_detector.cpp
  src/models/segmenter.cpp
  src/postprocess/seg_kernels.cu
  src/preprocess/kernels.cu
)
target_include_directories(adasvp
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(adasvp
  PUBLIC  ${OpenCV_LIBS}
  PRIVATE TensorRT::nvinfer TensorRT::nvonnxparser CUDA::cudart
)
target_compile_options(adasvp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

if(ADASVP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
