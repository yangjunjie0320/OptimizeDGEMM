cmake_minimum_required(VERSION 3.10)
project(optimized-gemm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Eigen found: ${EIGEN3_INCLUDE_DIRS}")
endif()

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    include_directories(${BLAS_INCLUDE_DIRS})
    link_directories(${BLAS_LIBRARY_DIRS})
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
endif()

# find all the files with name dgemm-*, it can be the subdirectory of src
set(KEY "dgemm")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*/${KEY}-*.c")
list(FILTER SOURCES EXCLUDE REGEX "dgemm-blas.c")

add_library(dgemm-blas)
target_sources(dgemm-blas PRIVATE src/dgemm-blas.c)
target_link_libraries(dgemm-blas ${BLAS_LIBRARIES})

foreach(f IN LISTS SOURCES)
    get_filename_component(basename ${f} NAME_WE)
    message(STATUS "preparing ${basename}")

    add_executable(
        main-${basename}
        src/main.cxx 
        ${f}
    )

    target_link_libraries(main-${basename} dgemm-blas)
    target_compile_options(main-${basename} PRIVATE -march=native -O0 -Ofast)
endforeach()
