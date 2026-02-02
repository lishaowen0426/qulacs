cmake_minimum_required(VERSION 3.21)

file(GLOB CSIM_SRC
    "*.cpp"
    "*.hpp"
)

if (MSVC)
    #We compile .c files in csim with MSVS C++ compiler, since there is no double _Complex in MSVC
    project(csim_static CXX)
    project(csim_shared CXX)
    file(GLOB CSIM_C_SRC "*.c")
    set_source_files_properties(${CSIM_C_SRC} PROPERTIES LANGUAGE CXX)
endif()

add_library(csim_static STATIC ${CSIM_SRC})
add_library(csim_shared SHARED ${CSIM_SRC})
#target_link_libraries(csim_static eigen)
#target_link_libraries(csim_shared eigen)
add_dependencies(csim_static eigen)
add_dependencies(csim_shared eigen)

set(COMPRESSOR_PREFIX "${CMAKE_SOURCE_DIR}/compressor/install")
set(COMPRESSOR_INCLUDE_DIR "${COMPRESSOR_PREFIX}/include")
set(COMPRESSOR_LIB_DIR "${COMPRESSOR_PREFIX}/lib")
file(GLOB COMPRESSOR_LIBS
    "${COMPRESSOR_LIB_DIR}/*.a"
    "${COMPRESSOR_LIB_DIR}/*.so"
    "${COMPRESSOR_LIB_DIR}/*.dylib"
)

target_include_directories(csim_static PRIVATE ${COMPRESSOR_INCLUDE_DIR})
target_include_directories(csim_shared PRIVATE ${COMPRESSOR_INCLUDE_DIR})
target_link_libraries(csim_static PRIVATE ${COMPRESSOR_LIBS})
target_link_libraries(csim_shared PRIVATE ${COMPRESSOR_LIBS})
target_link_libraries(csim_static PUBLIC blaz_static)
target_link_libraries(csim_shared PUBLIC blaz_static)

if (MSVC)
    set_target_properties(csim_static PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(csim_shared PROPERTIES LINKER_LANGUAGE CXX)
endif()

set_target_properties(csim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
