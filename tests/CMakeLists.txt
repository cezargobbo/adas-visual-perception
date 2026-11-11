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
  test_pipeline.cpp
  test_config.cpp
  test_ego_lane.cpp
)
target_include_directories(adasvp_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(adasvp_tests PRIVATE ADASVP_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(adasvp_tests PRIVATE adasvp Catch2::Catch2WithMain
  TensorRT::nvinfer CUDA::cudart opencv_imgproc opencv_imgcodecs opencv_videoio)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
  FetchContent_MakeAvailable(nlohmann_json)
endif()
target_link_libraries(adasvp_tests PRIVATE nlohmann_json::nlohmann_json)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(adasvp_tests)
