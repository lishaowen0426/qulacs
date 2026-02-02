cmake_minimum_required(VERSION 3.21)

file(GLOB CPPSIM_SRC
    "*.cpp"
    "*.hpp"
)

set(COMPRESSOR_PREFIX "${CMAKE_SOURCE_DIR}/compressor/install")
set(COMPRESSOR_INCLUDE_DIR "${COMPRESSOR_PREFIX}/include")
set(COMPRESSOR_LIB_DIR "${COMPRESSOR_PREFIX}/lib")
file(GLOB COMPRESSOR_LIBS
    "${COMPRESSOR_LIB_DIR}/*.a"
    "${COMPRESSOR_LIB_DIR}/*.so"
    "${COMPRESSOR_LIB_DIR}/*.dylib"
)

add_library(cppsim_static STATIC ${CPPSIM_SRC})
add_library(cppsim_shared SHARED ${CPPSIM_SRC})
target_include_directories(cppsim_static PRIVATE ${COMPRESSOR_INCLUDE_DIR})
target_include_directories(cppsim_shared PRIVATE ${COMPRESSOR_INCLUDE_DIR})
target_link_libraries(cppsim_static PRIVATE csim_static ${COMPRESSOR_LIBS} )
target_link_libraries(cppsim_shared PRIVATE csim_static ${COMPRESSOR_LIBS} )
add_dependencies(cppsim_static eigen)
add_dependencies(cppsim_shared eigen)




if (USE_GPU)
    target_link_libraries(cppsim_static gpusim_static)
    target_link_libraries(cppsim_shared gpusim_static)
endif()

set_target_properties(cppsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
