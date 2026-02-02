cmake_minimum_required(VERSION 3.21)

file(GLOB BLAZ_TEST_SRC "*.cpp")

add_executable(blaz_test EXCLUDE_FROM_ALL ${BLAZ_TEST_SRC})
target_link_libraries(blaz_test blaz_static gtest gtest_main)
