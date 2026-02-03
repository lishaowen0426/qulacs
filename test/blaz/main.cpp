#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "blaz/blaz.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: blaz_test <num_qubits> <trials>\n";
        return 2;
    }

     BlazConfig &cfg = blaz_config();
     cfg.keep_num = 15;
     cfg.keep_den=16;

    const ITYPE num_qubits =
        static_cast<ITYPE>(std::strtoull(argv[1], nullptr, 10));
    if (num_qubits >= 63) {
        std::cerr << "num_qubits must be < 63\n";
        return 2;
    }
    const ITYPE dim = 1ULL << num_qubits;
    const std::size_t trials =
        static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    double abs_err_sum = 0.0;
    double rel_err_sum = 0.0;
    double recon_err_sum = 0.0;

    for (std::size_t trial = 0; trial < trials; ++trial) {
        std::vector<CTYPE> state_a(dim);
        std::vector<CTYPE> state_b(dim);
        for (ITYPE i = 0; i < dim; ++i) {
            state_a[i] = CTYPE(dist(rng), dist(rng));
            state_b[i] = CTYPE(dist(rng), dist(rng));
        }

        auto normalize = [&](std::vector<CTYPE>& state) {
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
        };
        normalize(state_a);
        normalize(state_b);

        auto dot = [&](const std::vector<CTYPE>& x,
                       const std::vector<CTYPE>& y) {
            double sum_real = 0.0;
            double sum_imag = 0.0;
            for (ITYPE i = 0; i < dim; ++i) {
                const double xr = std::real(x[i]);
                const double xi = std::imag(x[i]);
                const double yr = std::real(y[i]);
                const double yi = std::imag(y[i]);
                sum_real += (xr * yr) - (xi * yi);
                sum_imag += (xr * yi) + (xi * yr);
            }
            return CTYPE(sum_real, sum_imag);
        };

        const CTYPE exact = dot(state_a, state_b);

        BlazCompressedComplex comp_a =
            blaz_compress_1d_complex_array(state_a.data(), dim);
        BlazCompressedComplex comp_b =
            blaz_compress_1d_complex_array(state_b.data(), dim);

        const CTYPE approx = blaz_dot_product(&comp_a, &comp_b);

        //std::cout << "trial " << trial << " exact=" << exact
        //          << " approx=" << approx << "\n";

        const double abs_err = std::abs(approx - exact);
        const double exact_mag = std::abs(exact);
        const double rel_err = (exact_mag == 0.0) ? 0.0 : (abs_err / exact_mag);

        abs_err_sum += abs_err;
        rel_err_sum += rel_err;
    }

    const double inv_trials = (trials == 0) ? 0.0 : (1.0 / trials);
    std::cout << "avg |approx - exact| = " << abs_err_sum * inv_trials << "\n";
    std::cout << "avg |approx - exact| / |exact| = "
              << rel_err_sum * inv_trials << "\n";

    return 0;
}
