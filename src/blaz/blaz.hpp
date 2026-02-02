#pragma once

#include <cstdint>
#include <vector>

#include "csim/type.hpp"

using MaskWord = uint64_t;
using BinType = int32_t;

struct BlazCompressedComplex {
    std::vector<std::size_t> s;
    std::vector<std::size_t> i;
    std::vector<MaskWord> mask;
    std::vector<double> N_r;
    std::vector<double> N_i;
    std::vector<BinType> F_r;  // binned coeff
    std::vector<BinType> F_i;
};

BlazCompressedComplex blaz_compress_1d_complex_array(CTYPE* state, ITYPE dim);
void blaz_decompress_1d_complex_array(
    const BlazCompressedComplex* comp, CTYPE* out_state);
