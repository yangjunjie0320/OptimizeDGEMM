cmake_minimum_required(VERSION 3.10)
project(optimized-gemm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Eigen found: ${EIGEN3_INCLUDE_DIRS}")
endif()

find_package(OpenBLAS REQUIRED)
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
    include_directories(${OpenBLAS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenBLAS not found!")
endif()

# find all the files with name dgemm-*, it can be the subdirectory of src
set(KEY "dgemm")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*/${KEY}-*.c")
list(FILTER SOURCES EXCLUDE REGEX "dgemm-blas.c")

add_library(dgemm-blas)
target_sources(dgemm-blas PRIVATE src/dgemm-blas.c)
target_link_libraries(dgemm-blas ${OpenBLAS_LIBRARIES})

foreach(f IN LISTS SOURCES)
    # get the parent directory name for example src/avx2/dgemm-2x2.c -> avx2-2x2
    get_filename_component(d ${f} DIRECTORY)
    get_filename_component(d ${d} NAME)
    get_filename_component(b ${f} NAME_WE)
    string(REGEX REPLACE "^dgemm-" "" b ${b})
    message(STATUS "preparing ${b}.x")

    # use the dirname-basename as the executable name
    add_executable(${b}.x src/main.cxx ${f})
    # target_compile_options(${b}.x PRIVATE)
    target_link_libraries(${b}.x dgemm-blas)
endforeach()
