#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "blaz/blaz.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: blaz_test <dim>\n";
        return 2;
    }
    const ITYPE dim = static_cast<ITYPE>(std::strtoull(argv[1], nullptr, 10));
    std::vector<CTYPE> state(dim);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (ITYPE i = 0; i < dim; ++i) {
        state[i] = CTYPE(dist(rng), dist(rng));
    }
    double norm_sq = 0.0;
    for (ITYPE i = 0; i < dim; ++i) {
        const double re = std::real(state[i]);
        const double im = std::imag(state[i]);
        norm_sq += re * re + im * im;
    }
    const double norm = std::sqrt(norm_sq);
    for (ITYPE i = 0; i < dim; ++i) {
        state[i] /= norm;
    }

    BlazCompressedComplex comp =
        blaz_compress_1d_complex_array(state.data(), dim);

    std::vector<CTYPE> restored(dim);
    blaz_decompress_1d_complex_array(&comp, restored.data());

    /*
    std::cout << "original:\n";
    for (ITYPE i = 0; i < dim; ++i) {
        std::cout << i << ": " << std::real(state[i]) << " + "
                  << std::imag(state[i]) << "i\n";
    }

    std::cout << "restored:\n";
    for (ITYPE i = 0; i < dim; ++i) {
        std::cout << i << ": " << std::real(restored[i]) << " + "
                  << std::imag(restored[i]) << "i\n";
    }
                  */

    double tvd_prob = 0.0;
    for (ITYPE i = 0; i < dim; ++i) {
        const double re0 = std::real(state[i]);
        const double im0 = std::imag(state[i]);
        const double re1 = std::real(restored[i]);
        const double im1 = std::imag(restored[i]);
        const double p0 = re0 * re0 + im0 * im0;
        const double p1 = re1 * re1 + im1 * im1;
        tvd_prob += std::abs(p0 - p1);
    }
    tvd_prob *= 0.5;
    std::cout << "tvd (prob) = " << tvd_prob << "\n";

    return 0;
}
