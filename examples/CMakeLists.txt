include(FetchContent)
find_package(CLI11 2.4 QUIET)
if(NOT CLI11_FOUND)
  FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2)
  FetchContent_MakeAvailable(CLI11)
endif()

# Parsing, output plan, CSV and ffmpeg writer: no GPU or display needed (linked into the tests).
add_library(adasvp_example_common STATIC cli.cpp alerts_csv.cpp ffmpeg_writer.cpp)
target_include_directories(adasvp_example_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adasvp_example_common PUBLIC adasvp CLI11::CLI11)
target_compile_options(adasvp_example_common PRIVATE -Wall -Wextra)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio highgui)
add_executable(video_demo video_demo.cpp)
target_link_libraries(video_demo PRIVATE adasvp_example_common CUDA::cudart opencv_imgproc opencv_videoio opencv_highgui)
target_compile_options(video_demo PRIVATE -Wall -Wextra)
