#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>

using CTYPE = std::complex<double>;
using ITYPE = unsigned long long;

using MaskWord = uint64_t;
using BinType = int32_t;

struct BlazConfig {
    std::size_t block_size = 64;
    std::size_t keep_num = 3;
    std::size_t keep_den = 4;
};

BlazConfig& blaz_config();

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
