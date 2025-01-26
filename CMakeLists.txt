cmake_minimum_required(VERSION 3.10)
project(sgemm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CHECK "Check the correctness" ON)
add_compile_options(-DCHECK=${CHECK})

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

set(KEY "sgemm")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/${KEY}-*.c")
list(FILTER SOURCES EXCLUDE REGEX "sgemm-blas.c")

add_library(sgemm-blas)
target_sources(sgemm-blas PRIVATE src/sgemm-blas.c)
target_link_libraries(sgemm-blas ${BLAS_LIBRARIES})

foreach(f IN LISTS SOURCES)
    get_filename_component(basename ${f} NAME_WE)
    message(STATUS "preparing ${basename}")

    add_executable(
        main-${basename}
        src/main.cxx 
        src/${basename}.c
    )

    target_link_libraries(main-${basename} sgemm-blas)
    target_compile_options(main-${basename} PRIVATE -march=native -O3 -Ofast)
endforeach()

