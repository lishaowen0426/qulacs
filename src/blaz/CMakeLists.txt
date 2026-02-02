cmake_minimum_required(VERSION 3.21)

file(GLOB BLAZ_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h")

add_library(blaz_static STATIC ${BLAZ_SRC})
