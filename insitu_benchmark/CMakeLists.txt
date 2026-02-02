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

find_library(ZSTD_LIB
    NAMES zstd
    PATHS ${COMPRESSOR_LIB_DIRS}
    NO_DEFAULT_PATH
)
if (NOT ZSTD_LIB)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZSTD REQUIRED libzstd)
    set(ZSTD_LIB ${ZSTD_LINK_LIBRARIES})
endif()

target_compile_features(insitu_benchmark PRIVATE cxx_std_17)
target_link_libraries(insitu_benchmark PRIVATE cppsim_static MPI::MPI_CXX ${COMPRESSOR_LIBS})
if (ZSTD_LIB)
    target_link_libraries(insitu_benchmark PRIVATE ${ZSTD_LIB})
    if (ZSTD_LIBRARY_DIRS)
        target_link_directories(insitu_benchmark PRIVATE ${ZSTD_LIBRARY_DIRS})
    endif()
endif()
target_include_directories(insitu_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src/blaz
    ${CMAKE_SOURCE_DIR}/src/cppsim
    ${CMAKE_SOURCE_DIR}/src/csim
    ${COMPRESSOR_INCLUDE_DIR}
)
