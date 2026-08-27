cmake_minimum_required(VERSION 3.16)
project(recastlib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RECASTLIB_ENABLE_LTO "Enable link-time optimization in Release builds" ON)
option(RECASTLIB_WITH_LINUX_AIO
  "Enable Linux O_DIRECT refinement with libaio" ON)

set(RECASTLIB_HAS_LINUX_AIO 0)
if(RECASTLIB_WITH_LINUX_AIO AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(RECASTLIB_HAS_LINUX_AIO 1)
endif()

if(RECASTLIB_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(
    RESULT RECASTLIB_LTO_SUPPORTED
    OUTPUT RECASTLIB_LTO_ERROR
    LANGUAGES CXX)
  if(NOT RECASTLIB_LTO_SUPPORTED)
    message(FATAL_ERROR "Release LTO requested but unsupported: ${RECASTLIB_LTO_ERROR}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(FAISS_ROOT "/usr/local" CACHE PATH "Faiss installation prefix")
find_path(FAISS_INCLUDE_DIR
  NAMES faiss/Index.h
  HINTS "${FAISS_ROOT}/include" /usr/local/include /opt/homebrew/include
  REQUIRED)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  find_library(FAISS_LIBRARY
    NAMES faiss_avx512 faiss_avx2 faiss
    HINTS "${FAISS_ROOT}/lib64" "${FAISS_ROOT}/lib"
          /usr/local/lib64 /usr/local/lib
    REQUIRED)
else()
  find_library(FAISS_LIBRARY
    NAMES faiss
    HINTS "${FAISS_ROOT}/lib" /usr/local/lib /opt/homebrew/lib
    REQUIRED)
endif()

find_package(Threads REQUIRED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT DEFINED BLA_VENDOR)
  # Avoid accidentally linking CentOS's slow reference libblas.
  set(BLA_VENDOR OpenBLAS)
endif()
find_package(BLAS REQUIRED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Dragonfruit's OpenBLAS exports the LAPACK entry points used by Faiss.
  set(RECASTLIB_MATH_LIBRARIES ${BLAS_LIBRARIES})
else()
  find_package(LAPACK REQUIRED)
  set(RECASTLIB_MATH_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

find_package(OpenMP QUIET)
if(APPLE AND NOT OpenMP_CXX_FOUND)
  find_path(LIBOMP_INCLUDE_DIR omp.h
    HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
  find_library(LIBOMP_LIBRARY omp
    HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
endif()
if(NOT OpenMP_CXX_FOUND AND NOT (APPLE AND LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY))
  message(FATAL_ERROR "OpenMP is required by the 8-thread integration test")
endif()

if(RECASTLIB_HAS_LINUX_AIO)
  find_path(LIBAIO_INCLUDE_DIR NAMES libaio.h REQUIRED)
  find_library(LIBAIO_LIBRARY NAMES aio REQUIRED)
endif()

add_library(recastlib STATIC
  src/exact_pq4.cpp
  src/fastscan.cpp
  src/quantizer.cpp
  src/uneven_product_quantizer.cpp
  adapters/external_list_ivf.cpp
  adapters/external_location_directory.cpp
  adapters/faiss_router.cpp
  adapters/list_file_aio_refiner.cpp
  adapters/list_file_record_store.cpp
  adapters/pq4_update_packing.cpp)

target_include_directories(recastlib
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/adapters>"
    "${FAISS_INCLUDE_DIR}")
target_compile_definitions(recastlib
  PUBLIC RECASTLIB_HAS_LINUX_AIO=${RECASTLIB_HAS_LINUX_AIO})
target_link_libraries(recastlib
  PUBLIC
    "${FAISS_LIBRARY}"
    ${RECASTLIB_MATH_LIBRARIES}
    Threads::Threads)
if(RECASTLIB_HAS_LINUX_AIO)
  target_include_directories(recastlib PRIVATE "${LIBAIO_INCLUDE_DIR}")
  target_link_libraries(recastlib PUBLIC "${LIBAIO_LIBRARY}")
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(recastlib PUBLIC OpenMP::OpenMP_CXX)
else()
  target_include_directories(recastlib PUBLIC "${LIBOMP_INCLUDE_DIR}")
  target_compile_options(recastlib PUBLIC -Xpreprocessor -fopenmp)
  target_link_libraries(recastlib PUBLIC "${LIBOMP_LIBRARY}")
endif()
target_compile_options(recastlib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(recastlib_test tests/test.cpp)
target_link_libraries(recastlib_test PRIVATE recastlib)
target_compile_options(recastlib_test PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME sift1m_usage COMMAND recastlib_test --help)

message(STATUS "Faiss headers: ${FAISS_INCLUDE_DIR}")
message(STATUS "Faiss library: ${FAISS_LIBRARY}")
message(STATUS "BLAS vendor: ${BLA_VENDOR}")
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "Linux AIO: ${RECASTLIB_HAS_LINUX_AIO}")
