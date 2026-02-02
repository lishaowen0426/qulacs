cmake_minimum_required(VERSION 3.13)

project(insitu_benchmark CXX)

add_executable(insitu_benchmark
    main.cpp
    insitu.cpp
)

find_package(MPI REQUIRED)

set(COMPRESSOR_PREFIX "${CMAKE_SOURCE_DIR}/compressor/install")
set(COMPRESSOR_INCLUDE_DIR "${COMPRESSOR_PREFIX}/include")
set(COMPRESSOR_LIB_DIR "${COMPRESSOR_PREFIX}/lib")
file(GLOB COMPRESSOR_LIBS
    "${COMPRESSOR_LIB_DIR}/*.a"
    "${COMPRESSOR_LIB_DIR}/*.so"
    "${COMPRESSOR_LIB_DIR}/*.dylib"
)

target_compile_features(insitu_benchmark PRIVATE cxx_std_17)
target_link_libraries(insitu_benchmark PRIVATE cppsim_static MPI::MPI_CXX ${COMPRESSOR_LIBS})
target_include_directories(insitu_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src/blaz
    ${CMAKE_SOURCE_DIR}/src/cppsim
    ${CMAKE_SOURCE_DIR}/src/csim
    ${COMPRESSOR_INCLUDE_DIR}
)
