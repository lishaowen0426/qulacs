cmake_minimum_required(VERSION 3.13)

project(insitu_benchmark CXX)

add_executable(insitu_benchmark
    main.cpp
    insitu.cpp
)

find_package(MPI REQUIRED)

set(COMPRESSOR_PREFIX "${CMAKE_SOURCE_DIR}/compressor/install")
set(COMPRESSOR_INCLUDE_DIR "${COMPRESSOR_PREFIX}/include")
set(COMPRESSOR_LIB_DIRS
    "${COMPRESSOR_PREFIX}/lib"
    "${COMPRESSOR_PREFIX}/lib64"
)
set(COMPRESSOR_LIBS "")
foreach(lib_dir IN LISTS COMPRESSOR_LIB_DIRS)
    if (EXISTS "${lib_dir}")
        file(GLOB _libs
            "${lib_dir}/*.a"
            "${lib_dir}/*.so"
            "${lib_dir}/*.dylib"
        )
        list(APPEND COMPRESSOR_LIBS ${_libs})
    endif()
endforeach()

target_compile_features(insitu_benchmark PRIVATE cxx_std_17)
target_link_libraries(insitu_benchmark PRIVATE cppsim_static MPI::MPI_CXX ${COMPRESSOR_LIBS})
target_include_directories(insitu_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src/blaz
    ${CMAKE_SOURCE_DIR}/src/cppsim
    ${CMAKE_SOURCE_DIR}/src/csim
    ${COMPRESSOR_INCLUDE_DIR}
)
