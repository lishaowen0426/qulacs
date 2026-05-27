#!/bin/sh

# Build qulacs with MPI on the A64FX (FX700) compute node.
#
# Run this ON THE COMPUTE NODE (aarch64), not the login node (x86), after:
#   module load system/fx700
#   module load FJSVstclanga
#
# Notes:
# - The Fujitsu MPI ships wrappers mpifcc (C) / mpiFCC (C++); there is no
#   mpicc / mpic++.
# - qulacs' CMakeLists only accepts compiler IDs GNU/Clang/AppleClang/MSVC.
#   The native Fujitsu compiler reports as Fujitsu/FujitsuClang and is rejected,
#   so we make the OpenMPI wrappers drive gcc/g++ (detected as GNU, which also
#   enables the aarch64 SVE code path).
export OMPI_CC=gcc
export OMPI_CXX=g++

export C_COMPILER=mpifcc
export CXX_COMPILER=mpiFCC
export USE_GPU=No
export USE_MPI=Yes
export USE_PYTHON=No   # system python is 3.6.8; pybind11 needs >=3.7. C++ lib only.

./script/build_gcc.sh
