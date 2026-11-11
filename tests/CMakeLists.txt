include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1)
FetchContent_MakeAvailable(Catch2)

add_executable(adasvp_tests
  test_trt_engine.cpp
  test_preprocess.cpp
  test_yolo.cpp
  test_segmenter.cpp
  test_lanes.cpp
  test_tracker.cpp
)
target_include_directories(adasvp_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(adasvp_tests PRIVATE ADASVP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(adasvp_tests PRIVATE adasvp Catch2::Catch2WithMain
  TensorRT::nvinfer CUDA::cudart)
find_package(nlohmann_json REQUIRED)
target_link_libraries(adasvp_tests PRIVATE nlohmann_json::nlohmann_json)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(adasvp_tests)
