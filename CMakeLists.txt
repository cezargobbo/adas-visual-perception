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

option(ADASVP_BUILD_TESTS "Build the C++ tests" ${PROJECT_IS_TOP_LEVEL})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

add_library(adasvp SHARED
  src/trt/trt_engine.cpp
  src/preprocess/letterbox.cpp
  src/models/yolo_detector.cpp
  src/models/segmenter.cpp
  src/models/lane_detector.cpp
  src/config.cpp
  src/pipeline.cpp
  src/render/draw_primitives.cpp
  src/render/renderer.cpp
  src/adas/ego_lane.cpp
  src/adas/lead_vehicle.cpp
  src/adas/fcw.cpp
  src/adas/ldw.cpp
  src/adas/traffic_light.cpp
  src/adas/rlr.cpp
  src/track/kalman_xyah.cpp
  src/track/assignment.cpp
  src/track/byte_tracker.cpp
  src/postprocess/seg_kernels.cu
  src/preprocess/kernels.cu
)
target_include_directories(adasvp
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(adasvp
  PUBLIC  opencv_core  # the public header only needs cv::Mat / cv::Point2f
  PRIVATE opencv_imgproc opencv_imgcodecs TensorRT::nvinfer TensorRT::nvonnxparser CUDA::cudart
)
target_compile_options(adasvp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
# decode_polar mirrors numpy float32 arithmetic: no fused multiply-add
# the tracker mirrors numpy float32/float64 arithmetic: no fused multiply-add
set_source_files_properties(src/track/kalman_xyah.cpp src/track/byte_tracker.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
set_source_files_properties(src/models/lane_detector.cpp src/adas/ego_lane.cpp src/adas/lead_vehicle.cpp src/adas/fcw.cpp src/adas/ldw.cpp src/adas/traffic_light.cpp src/adas/rlr.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)

add_executable(adasvp_dump tools/cpp/adasvp_dump.cpp)
target_link_libraries(adasvp_dump PRIVATE adasvp opencv_imgcodecs opencv_videoio)

if(ADASVP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
