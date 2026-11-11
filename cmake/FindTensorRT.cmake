# Finds TensorRT (nvinfer + nvonnxparser).
# Hints: TENSORRT_ROOT environment variable or CMake variable.
# Provides imported targets TensorRT::nvinfer and TensorRT::nvonnxparser.

find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT} ${TENSORRT_ROOT}
  PATH_SUFFIXES include
  PATHS /usr/include/x86_64-linux-gnu)

find_library(TensorRT_NVINFER_LIBRARY nvinfer
  HINTS $ENV{TENSORRT_ROOT} ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64
  PATHS /usr/lib/x86_64-linux-gnu)

find_library(TensorRT_NVONNXPARSER_LIBRARY nvonnxparser
  HINTS $ENV{TENSORRT_ROOT} ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64
  PATHS /usr/lib/x86_64-linux-gnu)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver
       REGEX "#define (TRT_MAJOR_ENTERPRISE|TRT_MINOR_ENTERPRISE|TRT_PATCH_ENTERPRISE|NV_TENSORRT_(MAJOR|MINOR|PATCH)) ")
  foreach(_c MAJOR MINOR PATCH)
    foreach(_l IN LISTS _trt_ver)
      if(_l MATCHES "#define (NV_TENSORRT|TRT)_${_c}(_ENTERPRISE)? +([0-9]+)")
        set(_trt_${_c} "${CMAKE_MATCH_3}")
      endif()
    endforeach()
  endforeach()
  set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_NVONNXPARSER_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
      IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()
  if(NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
      IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_NVONNXPARSER_LIBRARY)
