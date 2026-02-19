#include "insitu.hpp"

#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#endif

#include "SZ3/api/sz.hpp"
#include "blaz/blaz.hpp"
#include "cppsim/circuit.hpp"
#include "cppsim/gate_factory.hpp"
#include "cppsim/qasm_loader.hpp"
#include "cppsim/state_quantize.hpp"
#include "csim/MPIutil.hpp"
#include "spdlog/spdlog.h"
#include "mgard/compress_x.hpp"
#include "SPERR_C_API.h"
#include "fpzip.h"
#include "zfp.h"
#include "zstd.h"

QFTCircuitBuilder::QFTCircuitBuilder(UINT qubit_count)
    : qubit_count_(qubit_count) {}

QuantumCircuit *QFTCircuitBuilder::create_circuit(UINT qubit_count) const {
    auto *circuit = new QuantumCircuit(qubit_count);
    const double pi = 3.14159265358979323846;

    for (UINT target = 0; target < qubit_count; ++target) {
        circuit->add_gate(gate::H(target));
        for (UINT control = target + 1; control < qubit_count; ++control) {
            const double angle =
                pi / static_cast<double>(1ULL << (control - target));
            auto *phase = gate::U1(target, angle);
            phase->add_control_qubit(control, 1);
            circuit->add_gate(phase);
        }
    }

    for (UINT i = 0; i < qubit_count / 2; ++i) {
        circuit->add_gate(gate::SWAP(i, qubit_count - 1 - i));
    }

    return circuit;
}

QuantumCircuit *create_custom_h_qft_circuit(UINT qubit_count) {
    auto *circuit = new QuantumCircuit(qubit_count);
    const double pi = 3.14159265358979323846;

    for (UINT target = 0; target < qubit_count; ++target) {
        circuit->add_gate(gate::H_Custom_MPI(target));
        for (UINT control = target + 1; control < qubit_count; ++control) {
            const double angle =
                pi / static_cast<double>(1ULL << (control - target));
            auto *phase = gate::U1(target, angle);
            phase->add_control_qubit(control, 1);
            circuit->add_gate(phase);
        }
    }

    for (UINT i = 0; i < qubit_count / 2; ++i) {
        circuit->add_gate(gate::SWAP(i, qubit_count - 1 - i));
    }

    return circuit;
}

void apply_compression_before_epoch_h(QuantumCircuit *circuit,
    QuantumStateBase *input, UINT epoch, double error_bound, SZ3::EB error_mode,
    double &reduced_ratio) {
    if (circuit == nullptr || input == nullptr) {
        throw std::invalid_argument(
            "apply_compression_before_epoch_h: null circuit or input state");
    }
    if (!input->is_state_vector()) {
        throw std::invalid_argument(
            "apply_compression_before_epoch_h: input must be a state vector");
    }
    if (input->get_device_name() != "cpu") {
        throw std::invalid_argument(
            "apply_compression_before_epoch_h: only CPU states are supported");
    }
    const ITYPE dim = input->dim;
    if (dim <= 0) {
        return;
    }

    SZ3::Config config(static_cast<size_t>(dim));
    config.errorBoundMode = error_mode;
    if (error_mode == SZ3::EB_REL) {
        config.relErrorBound = error_bound;
    } else {
        config.absErrorBound = error_bound;
    }
    config.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    config.lorenzo = true;
    config.lorenzo2 = false;
    config.forceLossy = true;
    config.blockSize = 64;

    std::vector<double> real(static_cast<size_t>(dim));
    std::vector<double> imag(static_cast<size_t>(dim));

    double total_orig_bytes = 0.0;
    double total_cmp_bytes = 0.0;

    auto compress_decompress = [&](std::vector<double> &buffer) {
        size_t cmpSize = 0;

        char *cmp = SZ_compress(config, buffer.data(), cmpSize);
        SZ3::Config decConfig;
        double *dec = SZ_decompress<double>(decConfig, cmp, cmpSize);

        const double original_bytes = static_cast<double>(dim) * sizeof(double);
        total_orig_bytes += original_bytes;
        total_cmp_bytes += static_cast<double>(cmpSize);

        std::copy(dec, dec + static_cast<size_t>(dim), buffer.begin());
        delete[] dec;
        delete[] cmp;
    };

    UINT h_count = 0;
    bool did_compress = false;
    reduced_ratio = 0.0;
    for (auto *gate : circuit->gate_list) {
        if (!did_compress && gate->get_name() == "H") {
            if (h_count == epoch) {
                // std::cout << "Before compress: " << input << std::endl;
                auto *data = input->data_cpp();
                for (ITYPE i = 0; i < dim; ++i) {
                    real[static_cast<size_t>(i)] = data[i].real();
                    imag[static_cast<size_t>(i)] = data[i].imag();
                }

                compress_decompress(real);
                compress_decompress(imag);

                reduced_ratio =
                    (total_orig_bytes - total_cmp_bytes) / total_orig_bytes;

                for (ITYPE i = 0; i < dim; ++i) {
                    const size_t idx = static_cast<size_t>(i);
                    data[i] = CPPCTYPE(real[idx], imag[idx]);
                }

                did_compress = true;

                // std::cout << "After compress: " << input << std::endl;
            }
            ++h_count;
        }
        gate->update_quantum_state(input);
    }
}

void compression_on_h_mpi(QuantumCircuit *circuit, QuantumStateBase *input,
    UINT epoch, double error_bound, SZ3::EB error_mode, double &reduced_ratio) {
    reduced_ratio = 0.0;
    UINT h_count = 0;
    for (auto *gate : circuit->gate_list) {
        bool use_custom_update = false;
        if (gate->get_name() == "H") {
            if (h_count == epoch) {
                // Placeholder: target H located. Compression logic to be added.
                use_custom_update = true;
            }
            ++h_count;
        }

        if (use_custom_update) {
            auto *custom_h =
                gate::H_Custom_MPI(gate->target_qubit_list[0].index());
            custom_h->update_quantum_state(input);
        } else {
            gate->update_quantum_state(input);
        }
    }
}

int run_comp_on_mpi(int argc, char **argv) {
    if (argc < 2) {
        throw std::runtime_error("usage: <program> <num_qubits>");
    }

    UINT qubit_count = 0;
    try {
        qubit_count = static_cast<UINT>(std::stoul(argv[1]));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("invalid num_qubits: ") + e.what());
    }

    QFTCircuitBuilder builder(qubit_count);
    std::unique_ptr<QuantumCircuit> circuit(
        create_custom_h_qft_circuit(qubit_count));

    QuantumState initial_state(qubit_count, true);
    initial_state.set_computational_basis(0);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto sim_start = std::chrono::steady_clock::now();
    circuit->update_quantum_state(&initial_state);
    const auto sim_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> sim_elapsed = sim_end - sim_start;
    const double sim_time = sim_elapsed.count();
    const double comp_time =
        MPIutil::get_inst().get_compress_overhead_time_sum();
    const double ratio = sim_time > 0.0 ? (comp_time / sim_time) : 0.0;
    const double comp_ratio_avg = MPIutil::get_inst().get_compress_ratio_avg();

    double ratio_min = 0.0;
    double ratio_max = 0.0;
    double ratio_sum = 0.0;
    double comp_ratio_min = 0.0;
    double comp_ratio_max = 0.0;
    double comp_ratio_sum = 0.0;
    MPI_Reduce(&ratio, &ratio_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ratio, &ratio_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ratio, &ratio_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_ratio_avg, &comp_ratio_min, 1, MPI_DOUBLE, MPI_MIN, 0,
        MPI_COMM_WORLD);
    MPI_Reduce(&comp_ratio_avg, &comp_ratio_max, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);
    MPI_Reduce(&comp_ratio_avg, &comp_ratio_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD);

    if (rank == 0) {
        const double ratio_avg = ratio_sum / static_cast<double>(size);
        spdlog::info(
            "comp_time_ratio_min={:.6f} comp_time_ratio_max={:.6f} "
            "comp_time_ratio_avg={:.6f}",
            ratio_min, ratio_max, ratio_avg);
        const double comp_ratio_avg_all =
            comp_ratio_sum / static_cast<double>(size);
        spdlog::info(
            "comp_bytes_ratio_min={:.6f} comp_bytes_ratio_max={:.6f} "
            "comp_bytes_ratio_avg={:.6f}",
            comp_ratio_min, comp_ratio_max, comp_ratio_avg_all);
    }

    return 0;
}

namespace {
std::vector<CTYPE> normalize_state(std::vector<CTYPE> state) {
    long double norm_sq = 0.0L;
    for (const auto &v : state) {
        const long double re = std::real(v);
        const long double im = std::imag(v);
        norm_sq += re * re + im * im;
    }
    const long double norm = std::sqrt(norm_sq);
    if (norm == 0.0L) {
        return state;
    }
    for (auto &v : state) {
        v /= static_cast<double>(norm);
    }
    return state;
}

double tvd_prob(const std::vector<CTYPE> &a, const std::vector<CTYPE> &b) {
    long double sum = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const long double ar = std::real(a[i]);
        const long double ai = std::imag(a[i]);
        const long double br = std::real(b[i]);
        const long double bi = std::imag(b[i]);
        const long double pa = ar * ar + ai * ai;
        const long double pb = br * br + bi * bi;
        sum += std::fabs(pa - pb);
    }
    return static_cast<double>(0.5L * sum);
}

CTYPE inner_product(const std::vector<CTYPE> &a, const std::vector<CTYPE> &b) {
    CTYPE inner = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        inner += std::conj(a[i]) * b[i];
    }
    return inner;
}

double fidelity(const std::vector<CTYPE> &a, const std::vector<CTYPE> &b) {
    const CTYPE inner = inner_product(a, b);
    const double mag = std::abs(inner);
    return mag * mag;
}

void write_state(const std::string &path, const std::vector<CTYPE> &state) {
    std::ofstream out(path);
    out << std::setprecision(17);
    for (std::size_t i = 0; i < state.size(); ++i) {
        out << i << " " << std::real(state[i]) << " " << std::imag(state[i])
            << "\n";
    }
}

void ensure_output_dir(const std::string &dir) {
    std::filesystem::create_directories(dir);
}

std::vector<CTYPE> make_random_gaussian(std::size_t dim, std::mt19937 &rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<CTYPE> v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = CTYPE(dist(rng), dist(rng));
    }
    return normalize_state(std::move(v));
}

std::vector<CTYPE> make_low_freq(std::size_t dim) {
    const double two_pi = 2.0 * 3.14159265358979323846;
    std::vector<CTYPE> v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        const double x = two_pi * static_cast<double>(i) / dim;
        const double mag = 1.0 + 0.5 * std::sin(x);
        v[i] = CTYPE(mag * std::sin(x), mag * std::cos(x));
    }
    return normalize_state(std::move(v));
}

std::vector<CTYPE> make_high_freq(std::size_t dim) {
    const double two_pi = 2.0 * 3.14159265358979323846;
    const double freq = static_cast<double>(dim) / 4.0;
    std::vector<CTYPE> v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        const double x = two_pi * freq * static_cast<double>(i) / dim;
        const double mag =
            1.0 +
            0.5 * std::sin(two_pi * static_cast<double>(i) / (dim / 16.0));
        v[i] = CTYPE(mag * std::sin(x), mag * std::cos(x));
    }
    return normalize_state(std::move(v));
}

std::vector<CTYPE> make_sparse(std::size_t dim, std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> idx_dist(0, dim - 1);
    std::normal_distribution<double> val_dist(0.0, 1.0);
    std::vector<CTYPE> v(dim, CTYPE(0.0, 0.0));
    const std::size_t k = std::min<std::size_t>(16, dim);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t idx = idx_dist(rng);
        v[idx] = CTYPE(val_dist(rng), val_dist(rng));
    }
    return normalize_state(std::move(v));
}

std::vector<CTYPE> make_haar_random(std::size_t dim, std::mt19937 &rng) {
    if (dim == 0) {
        return {};
    }
    if ((dim & (dim - 1)) != 0) {
        return make_random_gaussian(dim, rng);
    }

    UINT qubit_count = 0;
    std::size_t t = dim;
    while (t > 1) {
        t >>= 1;
        ++qubit_count;
    }
    QuantumState state(qubit_count);
    state.set_Haar_random_state(static_cast<UINT>(rng()));
    std::vector<CTYPE> v(dim);
    const auto *data = state.data_cpp();
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = data[i];
    }
    return v;
}

std::vector<CTYPE> make_low_freq_with_rng(
    std::size_t dim, std::mt19937 &rng) {
    (void)rng;
    return make_low_freq(dim);
}

std::vector<CTYPE> make_high_freq_with_rng(
    std::size_t dim, std::mt19937 &rng) {
    (void)rng;
    return make_high_freq(dim);
}

}  // namespace

int benchmark_blaz(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const std::size_t dim = 4096;
    const std::vector<std::size_t> block_sizes = {64, 128, 256, 512};
    const std::vector<std::pair<std::size_t, std::size_t>> ratios = {
        {1, 1}, {7, 8}, {3, 4}, {1, 2}, {1, 4}};

    std::mt19937 rng(12345);
    const std::vector<std::pair<const char *, std::vector<CTYPE>>> families = {
        {"random_gaussian", make_random_gaussian(dim, rng)},
        {"haar_random", make_haar_random(dim, rng)},
        {"low_freq", make_low_freq(dim)},
        {"high_freq", make_high_freq(dim)},
        {"sparse", make_sparse(dim, rng)},
    };

    const std::string out_root = "results/benchmark_blaz";
    ensure_output_dir(out_root);
    std::cout << "blaz benchmark dim=" << dim << "\n";
    for (const auto &fam : families) {
        std::cout << "\nfamily: " << fam.first << "\n";
        const std::string fam_dir = out_root + "/" + fam.first;
        ensure_output_dir(fam_dir);
        write_state(fam_dir + "/original.txt", fam.second);
        for (std::size_t bs : block_sizes) {
            for (const auto &ratio : ratios) {
                BlazConfig &cfg = blaz_config();
                cfg.block_size = bs;
                cfg.keep_num = ratio.first;
                cfg.keep_den = ratio.second;
                if (dim % bs != 0 || bs % cfg.keep_den != 0) {
                    continue;
                }

                std::vector<CTYPE> restored(dim);
                BlazCompressedComplex comp = blaz_compress_1d_complex_array(
                    const_cast<CTYPE *>(fam.second.data()),
                    static_cast<ITYPE>(dim));
                blaz_decompress_1d_complex_array(&comp, restored.data());

                const double tvd = tvd_prob(fam.second, restored);
                const double fid = fidelity(fam.second, restored);
                write_state(fam_dir + "/block" + std::to_string(bs) + "_keep" +
                                std::to_string(ratio.first) + "of" +
                                std::to_string(ratio.second) + ".txt",
                    restored);
                std::cout << "  block=" << bs << " keep=" << ratio.first << "/"
                          << ratio.second << " tvd=" << std::setprecision(6)
                          << std::scientific << tvd
                          << " fid=" << std::setprecision(6) << std::scientific
                          << fid << "\n";
            }
        }
    }
    return 0;
}

int benchmark_quantization(int argc, char **argv) {
    if (argc < 2) {
        throw std::runtime_error("usage: <program> <qasm_path> [error_bound]");
    }
    const std::string qasm_path = argv[1];

    if (argc >= 3) {
        const double error_bound = std::stod(argv[2]);
        QuantumStateCpuQuant::set_error_bound(error_bound);
    }

    std::unique_ptr<QuantumCircuit> circuit = load_qasm_file(qasm_path);
    if (!circuit) {
        throw std::runtime_error("failed to load qasm circuit");
    }

    MPIutil &mpiutil = MPIutil::get_inst();
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const UINT qubit_count = circuit->qubit_count;
    QuantumState normal_state(qubit_count, true);
    normal_state.set_computational_basis(0);
    mpiutil.set_quant_comm_enabled(false);
    mpiutil.reset_bytes_exchanged_counters();
    for (auto *gate : circuit->gate_list) {
        gate->update_quantum_state(&normal_state);
    }
    const uint64_t local_bytes_no_quant_run =
        mpiutil.get_bytes_exchanged_no_quant();
    std::vector<CTYPE> normal_output(static_cast<size_t>(normal_state.dim));
    std::copy_n(
        normal_state.data_cpp(), static_cast<size_t>(normal_state.dim), normal_output.data());

    QuantumStateCpuQuant quant_state(qubit_count, true);
    quant_state.set_computational_basis(0);
    mpiutil.set_quant_comm_enabled(true);
    mpiutil.reset_bytes_exchanged_counters();
    for (auto *gate : circuit->gate_list) {
        gate->update_quantum_state(&quant_state);
        quant_state.quantize();
    }
    const uint64_t local_bytes_quant_run = mpiutil.get_bytes_exchanged_quant();
    std::vector<CTYPE> quant_output(static_cast<size_t>(quant_state.dim));
    std::copy_n(
        quant_state.data_cpp(), static_cast<size_t>(quant_state.dim), quant_output.data());

    // Reset default back to non-quantized communication.
    mpiutil.set_quant_comm_enabled(false);

    const double local_tvd = tvd_prob(normal_output, quant_output);
    const CTYPE local_inner = inner_product(normal_output, quant_output);
    const double local_inner_re = std::real(local_inner);
    const double local_inner_im = std::imag(local_inner);
    double local_norm_normal_sq = 0.0;
    double local_norm_quant_sq = 0.0;

    double local_l2_sq = 0.0;
    double local_max_abs = 0.0;
    for (size_t i = 0; i < normal_output.size(); ++i) {
        local_norm_normal_sq += std::norm(normal_output[i]);
        local_norm_quant_sq += std::norm(quant_output[i]);
        const CTYPE diff = normal_output[i] - quant_output[i];
        const double abs_diff = std::abs(diff);
        local_l2_sq += abs_diff * abs_diff;
        if (abs_diff > local_max_abs) {
            local_max_abs = abs_diff;
        }
    }

    double global_l2_sq = 0.0;
    double global_max_abs = 0.0;
    double global_tvd = 0.0;
    double global_inner_re = 0.0;
    double global_inner_im = 0.0;
    double global_norm_normal_sq = 0.0;
    double global_norm_quant_sq = 0.0;
    uint64_t global_bytes_no_quant_run = 0;
    uint64_t global_bytes_quant_run = 0;
    MPI_Reduce(
        &local_l2_sq, &global_l2_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_max_abs, &global_max_abs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_tvd, &global_tvd, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_inner_re, &global_inner_re, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_inner_im, &global_inner_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_norm_normal_sq, &global_norm_normal_sq, 1, MPI_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_norm_quant_sq, &global_norm_quant_sq, 1, MPI_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_bytes_no_quant_run, &global_bytes_no_quant_run, 1,
        MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_bytes_quant_run, &global_bytes_quant_run, 1,
        MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double numerator = global_inner_re * global_inner_re +
                                 global_inner_im * global_inner_im;
        const double denominator = global_norm_normal_sq * global_norm_quant_sq;
        const double global_fidelity =
            (denominator > 0.0) ? (numerator / denominator) : 0.0;
        const std::string qasm_name =
            std::filesystem::path(qasm_path).filename().string();
        std::cout << std::setprecision(17) << "qasm=" << qasm_name
                  << " ranks=" << size << " qubits=" << qubit_count
                  << " gates=" << circuit->gate_list.size()
                  << " error_bound=" << QuantumStateCpuQuant::get_error_bound()
                  << " l2=" << std::sqrt(global_l2_sq)
                  << " max_abs=" << global_max_abs << " tvd=" << global_tvd
                  << " fidelity=" << global_fidelity
                  << " bytes_no_quant=" << global_bytes_no_quant_run
                  << " bytes_quant=" << global_bytes_quant_run << "\n";
    }
    return 0;
}


namespace {
struct BenchResult {
    bool ok = false;
    std::string error;
    double compress_ms = 0.0;
    double decompress_ms = 0.0;
    double ratio = 0.0;
    double tvd = 0.0;
    double fid = 0.0;
};

using CompressorRunner =
    BenchResult (*)(const std::vector<CTYPE> &, std::size_t);

struct CompressorCase {
    const char *compressor;
    const char *config;
    CompressorRunner run;
};

constexpr double kGlobalAbsErrorBound = 1e-4;
constexpr double kGlobalRelErrorBound = 1e-2;
constexpr int kZstdLevel = 3;

int fpzip_precision_from_abs_error(double abs_eb) {
    if (!(abs_eb > 0.0)) {
        return 0;
    }
    int prec = static_cast<int>(std::ceil(-std::log2(abs_eb))) + 2;
    prec = std::max(4, std::min(64, prec));
    if (prec % 2 != 0) {
        ++prec;
    }
    return std::min(64, prec);
}

std::size_t blaz_comp_bytes(const BlazCompressedComplex &comp) {
    return comp.s.size() * sizeof(std::size_t) +
           comp.i.size() * sizeof(std::size_t) +
           comp.mask.size() * sizeof(MaskWord) +
           comp.N_r.size() * sizeof(double) + comp.N_i.size() * sizeof(double) +
           comp.F_r.size() * sizeof(BinType) + comp.F_i.size() * sizeof(BinType);
}

SZ3::Config make_sz_config(
    std::size_t dim, SZ3::EB eb_mode, double eb, SZ3::ALGO algo, int block_size) {
    SZ3::Config cfg(dim);
    cfg.errorBoundMode = eb_mode;
    if (eb_mode == SZ3::EB_REL) {
        cfg.relErrorBound = eb;
        cfg.absErrorBound = 0.0;
    } else {
        cfg.absErrorBound = eb;
        cfg.relErrorBound = 0.0;
    }
    cfg.cmprAlgo = algo;
    cfg.blockSize = block_size;
    return cfg;
}

std::size_t sz_large_buffer_cap(const SZ3::Config &cfg) {
    const std::size_t bound = SZ3::SZ_compress_size_bound<double>(cfg);
    const std::size_t margin = 1ULL * 1024ULL * 1024ULL;
    if (bound > std::numeric_limits<std::size_t>::max() - margin) {
        return bound;
    }
    return bound + margin;
}

bool sz_compress_with_large_buffer(const SZ3::Config &cfg, const double *data,
    std::vector<char> &buffer, std::size_t &cmp_size, std::string &error) {
    try {
        if (buffer.empty()) {
            buffer.resize(sz_large_buffer_cap(cfg));
        }
        cmp_size = SZ_compress(cfg, data, buffer.data(), buffer.size());
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}

uint64_t bits_of_double(double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

double double_of_bits(uint64_t bits) {
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

uint64_t truncate_double_word(uint64_t bits, int drop_bits, double abs_eb) {
    if (drop_bits <= 0) {
        return bits;
    }
    const uint64_t exp_mask = 0x7FF0000000000000ULL;
    const uint64_t mant_mask = 0x000FFFFFFFFFFFFFULL;
    const uint64_t exp_bits = bits & exp_mask;
    if (exp_bits == exp_mask) {
        return bits;
    }

    const double v = double_of_bits(bits);
    if (std::abs(v) < abs_eb) {
        return bits_of_double(0.0);
    }

    const int clamped = std::min(52, drop_bits);
    const uint64_t keep_mant = ~((1ULL << clamped) - 1ULL);
    const uint64_t mant = bits & mant_mask;
    return (bits & ~mant_mask) | (mant & keep_mant);
}

int drop_bits_from_rel_eb(double rel_eb) {
    if (rel_eb <= 0.0) {
        return 0;
    }
    const double raw = std::floor(52.0 + std::log2(rel_eb));
    const int bits = static_cast<int>(raw);
    return std::max(0, std::min(52, bits));
}

std::vector<uint64_t> make_words_for_xor_pipeline(
    const std::vector<CTYPE> &input, bool interleave_ri) {
    const std::size_t n = input.size();
    std::vector<uint64_t> words;
    words.reserve(2 * n);
    const int drop_bits = drop_bits_from_rel_eb(kGlobalRelErrorBound);

    if (interleave_ri) {
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t rb = truncate_double_word(
                bits_of_double(std::real(input[i])), drop_bits,
                kGlobalAbsErrorBound);
            const uint64_t ib = truncate_double_word(
                bits_of_double(std::imag(input[i])), drop_bits,
                kGlobalAbsErrorBound);
            words.push_back(rb);
            words.push_back(ib);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t rb = truncate_double_word(
                bits_of_double(std::real(input[i])), drop_bits,
                kGlobalAbsErrorBound);
            words.push_back(rb);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t ib = truncate_double_word(
                bits_of_double(std::imag(input[i])), drop_bits,
                kGlobalAbsErrorBound);
            words.push_back(ib);
        }
    }
    return words;
}

void restore_words_to_state(const std::vector<uint64_t> &words,
    bool interleave_ri, std::vector<CTYPE> &out) {
    const std::size_t n = out.size();
    if (interleave_ri) {
        for (std::size_t i = 0; i < n; ++i) {
            const double r = double_of_bits(words[2 * i]);
            const double im = double_of_bits(words[2 * i + 1]);
            out[i] = CTYPE(r, im);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const double r = double_of_bits(words[i]);
            const double im = double_of_bits(words[n + i]);
            out[i] = CTYPE(r, im);
        }
    }
}

void append_u64_le(std::vector<uint8_t> &dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFU));
    }
}

uint64_t read_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

std::vector<uint8_t> encode_xor_lz(const std::vector<uint64_t> &words) {
    std::vector<uint8_t> out;
    out.reserve(words.size() * 6 + 16);
    append_u64_le(out, static_cast<uint64_t>(words.size()));

    uint64_t prev = 0;
    for (std::size_t i = 0; i < words.size(); ++i) {
        const uint64_t x = (i == 0) ? words[i] : (words[i] ^ prev);
        prev = words[i];

        uint8_t keep = 0;
        if (x != 0) {
            keep = 8;
            while (keep > 0 && ((x >> (8 * (keep - 1))) & 0xFFU) == 0) {
                --keep;
            }
        }
        out.push_back(keep);
        for (uint8_t b = 0; b < keep; ++b) {
            out.push_back(static_cast<uint8_t>((x >> (8 * b)) & 0xFFU));
        }
    }
    return out;
}

bool decode_xor_lz(const uint8_t *data, std::size_t len, std::vector<uint64_t> &words,
    std::string &error) {
    if (len < 8) {
        error = "xor_lz decode: payload too short";
        return false;
    }
    const uint64_t n64 = read_u64_le(data);
    const std::size_t n = static_cast<std::size_t>(n64);
    words.assign(n, 0);
    std::size_t p = 8;
    uint64_t prev = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (p >= len) {
            error = "xor_lz decode: truncated keep byte";
            return false;
        }
        const uint8_t keep = data[p++];
        if (keep > 8 || p + keep > len) {
            error = "xor_lz decode: invalid keep length";
            return false;
        }
        uint64_t x = 0;
        for (uint8_t b = 0; b < keep; ++b) {
            x |= static_cast<uint64_t>(data[p++]) << (8 * b);
        }
        words[i] = (i == 0) ? x : (x ^ prev);
        prev = words[i];
    }
    return true;
}

BenchResult run_xor_lz_bitplane_zstd(
    const std::vector<CTYPE> &input, std::size_t dim, bool interleave_ri) {
    (void)dim;
    const auto words = make_words_for_xor_pipeline(input, interleave_ri);

    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<uint8_t> encoded = encode_xor_lz(words);
    const size_t cap = ZSTD_compressBound(encoded.size());
    std::vector<uint8_t> cmp(cap);
    const size_t cmp_size = ZSTD_compress(
        cmp.data(), cap, encoded.data(), encoded.size(), kZstdLevel);
    if (ZSTD_isError(cmp_size)) {
        return BenchResult{false, std::string("zstd_compress: ") +
                                      ZSTD_getErrorName(cmp_size)};
    }
    cmp.resize(cmp_size);
    const auto t1 = std::chrono::steady_clock::now();

    const unsigned long long dec_size_ull =
        ZSTD_getFrameContentSize(cmp.data(), cmp.size());
    if (dec_size_ull == ZSTD_CONTENTSIZE_ERROR ||
        dec_size_ull == ZSTD_CONTENTSIZE_UNKNOWN) {
        return BenchResult{false, "zstd frame content size unavailable"};
    }
    const std::size_t dec_size = static_cast<std::size_t>(dec_size_ull);
    std::vector<uint8_t> dec_encoded(dec_size);
    const size_t actual_dec = ZSTD_decompress(
        dec_encoded.data(), dec_encoded.size(), cmp.data(), cmp.size());
    if (ZSTD_isError(actual_dec)) {
        return BenchResult{
            false, std::string("zstd_decompress: ") + ZSTD_getErrorName(actual_dec)};
    }
    if (actual_dec != dec_encoded.size()) {
        return BenchResult{false, "zstd_decompress size mismatch"};
    }

    std::vector<uint64_t> dec_words;
    std::string err;
    if (!decode_xor_lz(dec_encoded.data(), dec_encoded.size(), dec_words, err)) {
        return BenchResult{false, err};
    }
    std::vector<CTYPE> restored(input.size());
    restore_words_to_state(dec_words, interleave_ri, restored);
    const auto t2 = std::chrono::steady_clock::now();

    const double compress_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decompress_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double original_bytes = static_cast<double>(input.size() * sizeof(CTYPE));
    const double ratio =
        cmp_size > 0 ? (original_bytes / static_cast<double>(cmp_size)) : 0.0;
    return BenchResult{true, "", compress_ms, decompress_ms, ratio,
        tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_xor_lz_bitplane_zstd_c(
    const std::vector<CTYPE> &input, std::size_t dim) {
    return run_xor_lz_bitplane_zstd(input, dim, false);
}

BenchResult run_xor_lz_bitplane_zstd_d(
    const std::vector<CTYPE> &input, std::size_t dim) {
    return run_xor_lz_bitplane_zstd(input, dim, true);
}

bool zfp_compress_decompress_channel(const std::vector<double> &in,
    std::vector<double> &out, double &compress_ms, double &decompress_ms,
    std::size_t &compressed_bytes, std::string &error) {
    zfp_field *field =
        zfp_field_1d(const_cast<double *>(in.data()), zfp_type_double, in.size());
    if (!field) {
        error = "zfp_field_1d failed";
        return false;
    }

    zfp_stream *zfp = zfp_stream_open(nullptr);
    if (!zfp) {
        zfp_field_free(field);
        error = "zfp_stream_open failed";
        return false;
    }
    zfp_stream_set_accuracy(zfp, kGlobalAbsErrorBound);

    const size_t max_size = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> buffer(max_size);
    bitstream *stream = stream_open(buffer.data(), max_size);
    if (!stream) {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "zfp stream_open failed";
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    const auto t0 = std::chrono::steady_clock::now();
    const size_t zfp_size = zfp_compress(zfp, field);
    const auto t1 = std::chrono::steady_clock::now();
    if (!zfp_size) {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "zfp_compress failed";
        return false;
    }

    stream_close(stream);
    stream = stream_open(buffer.data(), zfp_size);
    if (!stream) {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "zfp stream_open(decompress) failed";
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);
    zfp_field_set_pointer(field, out.data());
    if (!zfp_decompress(zfp, field)) {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "zfp_decompress failed";
        return false;
    }
    const auto t2 = std::chrono::steady_clock::now();

    compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    decompress_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    compressed_bytes = zfp_size;

    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    return true;
}

BenchResult run_zfp_accuracy_abs(
    const std::vector<CTYPE> &input, std::size_t dim) {
    (void)dim;
    const std::size_t n = input.size();
    std::vector<double> real(n), imag(n), dec_real(n), dec_imag(n);
    for (std::size_t i = 0; i < n; ++i) {
        real[i] = std::real(input[i]);
        imag[i] = std::imag(input[i]);
    }

    double comp_real_ms = 0.0;
    double decomp_real_ms = 0.0;
    double comp_imag_ms = 0.0;
    double decomp_imag_ms = 0.0;
    std::size_t cmp_real_bytes = 0;
    std::size_t cmp_imag_bytes = 0;
    std::string err;

    if (!zfp_compress_decompress_channel(real, dec_real, comp_real_ms,
            decomp_real_ms, cmp_real_bytes, err)) {
        return BenchResult{false, "zfp_real:" + err};
    }
    if (!zfp_compress_decompress_channel(imag, dec_imag, comp_imag_ms,
            decomp_imag_ms, cmp_imag_bytes, err)) {
        return BenchResult{false, "zfp_imag:" + err};
    }

    std::vector<CTYPE> restored(n);
    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(dec_real[i], dec_imag[i]);
    }

    const double original_bytes = static_cast<double>(n * sizeof(CTYPE));
    const std::size_t cmp_bytes = cmp_real_bytes + cmp_imag_bytes;
    const double ratio =
        cmp_bytes > 0 ? (original_bytes / static_cast<double>(cmp_bytes)) : 0.0;
    return BenchResult{true, "", comp_real_ms + comp_imag_ms,
        decomp_real_ms + decomp_imag_ms, ratio, tvd_prob(input, restored),
        fidelity(input, restored)};
}

BenchResult run_mgard_abs_global(
    const std::vector<CTYPE> &input, std::size_t dim) {
    (void)dim;
    const std::size_t n = input.size();
    std::vector<double> interleaved(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        interleaved[2 * i] = std::real(input[i]);
        interleaved[2 * i + 1] = std::imag(input[i]);
    }

    if (n < 6 || (n % 2) != 0) {
        return BenchResult{false, "mgard requires n>=6 and even for 2d packing"};
    }

    void *compressed_data = nullptr;
    void *decompressed_data = nullptr;
    std::size_t compressed_bytes = 0;
    const std::vector<mgard_x::SIZE> shape = {static_cast<mgard_x::SIZE>(n / 2), 4};

    const auto t0 = std::chrono::steady_clock::now();
    const mgard_x::compress_status_type cstat = mgard_x::compress(2,
        mgard_x::data_type::Double, shape, kGlobalAbsErrorBound, 0.0,
        mgard_x::error_bound_type::ABS, interleaved.data(), compressed_data,
        compressed_bytes, false);
    const auto t1 = std::chrono::steady_clock::now();
    if (cstat != mgard_x::compress_status_type::Success || !compressed_data) {
        return BenchResult{false, "mgard compress failed"};
    }

    std::vector<mgard_x::SIZE> dec_shape;
    mgard_x::data_type dec_dtype = mgard_x::data_type::Double;
    const mgard_x::compress_status_type dstat = mgard_x::decompress(
        compressed_data, compressed_bytes, decompressed_data, dec_shape, dec_dtype, false);
    const auto t2 = std::chrono::steady_clock::now();
    if (dstat != mgard_x::compress_status_type::Success || !decompressed_data) {
        free(compressed_data);
        return BenchResult{false, "mgard decompress failed"};
    }
    if (dec_dtype != mgard_x::data_type::Double || dec_shape.size() != 2 ||
        dec_shape[0] != shape[0] || dec_shape[1] != shape[1]) {
        free(compressed_data);
        free(decompressed_data);
        return BenchResult{false, "mgard decompressed metadata mismatch"};
    }

    std::vector<double> restored_interleaved(2 * n);
    std::memcpy(restored_interleaved.data(), decompressed_data, restored_interleaved.size() * sizeof(double));
    std::vector<CTYPE> restored(n);
    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(restored_interleaved[2 * i], restored_interleaved[2 * i + 1]);
    }

    free(compressed_data);
    free(decompressed_data);

    const double original_bytes = static_cast<double>(n * sizeof(CTYPE));
    const double ratio =
        compressed_bytes > 0 ? (original_bytes / static_cast<double>(compressed_bytes))
                             : 0.0;
    const double compress_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decompress_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    return BenchResult{true, "", compress_ms, decompress_ms, ratio,
        tvd_prob(input, restored),
        fidelity(input, restored)};
}

BenchResult run_sperr_2d_pwe_abs_global(
    const std::vector<CTYPE> &input, std::size_t dim) {
    const std::size_t n = input.size();
    if (n != dim) {
        return BenchResult{false, "sperr dim mismatch"};
    }
    std::vector<double> interleaved(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        interleaved[2 * i] = std::real(input[i]);
        interleaved[2 * i + 1] = std::imag(input[i]);
    }

    void *cmp = nullptr;
    std::size_t cmp_len = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = C_API::sperr_comp_2d(interleaved.data(), 0, 2, n,
        3 /* mode=pwe */, kGlobalAbsErrorBound, 0 /* no header */, &cmp, &cmp_len);
    const auto t1 = std::chrono::steady_clock::now();
    if (rc != 0 || cmp == nullptr || cmp_len == 0) {
        if (cmp) {
            free(cmp);
        }
        return BenchResult{false, "sperr_comp_2d failed"};
    }

    void *dec = nullptr;
    const int rd =
        C_API::sperr_decomp_2d(cmp, cmp_len, 0, 2, n, &dec);
    const auto t2 = std::chrono::steady_clock::now();
    if (rd != 0 || dec == nullptr) {
        free(cmp);
        if (dec) {
            free(dec);
        }
        return BenchResult{false, "sperr_decomp_2d failed"};
    }

    const auto *dec_d = static_cast<const double *>(dec);
    std::vector<CTYPE> restored(n);
    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(dec_d[2 * i], dec_d[2 * i + 1]);
    }

    free(cmp);
    free(dec);

    const double compress_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decompress_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double original_bytes = static_cast<double>(n * sizeof(CTYPE));
    const double ratio =
        cmp_len > 0 ? (original_bytes / static_cast<double>(cmp_len)) : 0.0;
    return BenchResult{true, "", compress_ms, decompress_ms, ratio,
        tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_fpzip_2d_precision_abs(
    const std::vector<CTYPE> &input, std::size_t dim) {
    const std::size_t n = input.size();
    if (n != dim) {
        return BenchResult{false, "fpzip dim mismatch"};
    }

    std::vector<double> interleaved(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        interleaved[2 * i] = std::real(input[i]);
        interleaved[2 * i + 1] = std::imag(input[i]);
    }

    const std::size_t raw_bytes = interleaved.size() * sizeof(double);
    std::size_t cmp_cap = raw_bytes + raw_bytes / 2 + 4096;
    if (cmp_cap < raw_bytes) {
        cmp_cap = raw_bytes;
    }
    std::vector<unsigned char> cmp(cmp_cap);

    FPZ *writer = fpzip_write_to_buffer(cmp.data(), cmp.size());
    if (!writer) {
        return BenchResult{false, "fpzip_write_to_buffer failed"};
    }
    writer->type = FPZIP_TYPE_DOUBLE;
    writer->prec = fpzip_precision_from_abs_error(kGlobalAbsErrorBound);
    writer->nx = 2;
    writer->ny = static_cast<int>(n);
    writer->nz = 1;
    writer->nf = 1;

    const auto t0 = std::chrono::steady_clock::now();
    if (!fpzip_write_header(writer)) {
        fpzip_write_close(writer);
        return BenchResult{false, "fpzip_write_header failed"};
    }
    const size_t cmp_size = fpzip_write(writer, interleaved.data());
    const auto t1 = std::chrono::steady_clock::now();
    fpzip_write_close(writer);
    if (cmp_size == 0) {
        return BenchResult{false, "fpzip_write failed"};
    }

    FPZ *reader = fpzip_read_from_buffer(cmp.data());
    if (!reader) {
        return BenchResult{false, "fpzip_read_from_buffer failed"};
    }
    if (!fpzip_read_header(reader)) {
        fpzip_read_close(reader);
        return BenchResult{false, "fpzip_read_header failed"};
    }
    if (reader->type != FPZIP_TYPE_DOUBLE || reader->nx != 2 ||
        reader->ny != static_cast<int>(n) || reader->nz != 1 ||
        reader->nf != 1) {
        fpzip_read_close(reader);
        return BenchResult{false, "fpzip header mismatch"};
    }

    std::vector<double> dec_interleaved(2 * n, 0.0);
    const size_t dec_read = fpzip_read(reader, dec_interleaved.data());
    const auto t2 = std::chrono::steady_clock::now();
    fpzip_read_close(reader);
    if (dec_read == 0) {
        return BenchResult{false, "fpzip_read failed"};
    }

    std::vector<CTYPE> restored(n);
    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(dec_interleaved[2 * i], dec_interleaved[2 * i + 1]);
    }

    const double compress_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decompress_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double original_bytes = static_cast<double>(n * sizeof(CTYPE));
    const double ratio =
        cmp_size > 0 ? (original_bytes / static_cast<double>(cmp_size)) : 0.0;
    return BenchResult{true, "", compress_ms, decompress_ms, ratio,
        tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_blaz_block64_keep3of4(
    const std::vector<CTYPE> &input, std::size_t dim) {
    if (dim % 64 != 0) {
        return BenchResult{false, "dim_not_divisible_by_block"};
    }
    BlazConfig &cfg = blaz_config();
    cfg.block_size = 64;
    cfg.keep_num = 3;
    cfg.keep_den = 4;
    if (cfg.block_size % cfg.keep_den != 0) {
        return BenchResult{false, "invalid_blaz_config"};
    }

    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    double total_cmp_bytes = 0.0;
    std::vector<CTYPE> restored(dim);
    std::vector<CTYPE> work = input;
    const auto t0 = std::chrono::steady_clock::now();
    BlazCompressedComplex comp =
        blaz_compress_1d_complex_array(work.data(), static_cast<ITYPE>(dim));
    const auto t1 = std::chrono::steady_clock::now();
    blaz_decompress_1d_complex_array(&comp, restored.data());
    const auto t2 = std::chrono::steady_clock::now();
    total_comp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_decomp_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    total_cmp_bytes += static_cast<double>(blaz_comp_bytes(comp));

    const double avg_cmp_bytes = total_cmp_bytes;
    const double original_bytes = static_cast<double>(dim * sizeof(CTYPE));
    const double ratio =
        avg_cmp_bytes > 0.0 ? (original_bytes / avg_cmp_bytes) : 0.0;
    return BenchResult{true, "", total_comp_ms, total_decomp_ms,
        ratio, tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_blaz_block128_keep1of2(
    const std::vector<CTYPE> &input, std::size_t dim) {
    if (dim % 128 != 0) {
        return BenchResult{false, "dim_not_divisible_by_block"};
    }
    BlazConfig &cfg = blaz_config();
    cfg.block_size = 128;
    cfg.keep_num = 1;
    cfg.keep_den = 2;
    if (cfg.block_size % cfg.keep_den != 0) {
        return BenchResult{false, "invalid_blaz_config"};
    }

    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    double total_cmp_bytes = 0.0;
    std::vector<CTYPE> restored(dim);
    std::vector<CTYPE> work = input;
    const auto t0 = std::chrono::steady_clock::now();
    BlazCompressedComplex comp =
        blaz_compress_1d_complex_array(work.data(), static_cast<ITYPE>(dim));
    const auto t1 = std::chrono::steady_clock::now();
    blaz_decompress_1d_complex_array(&comp, restored.data());
    const auto t2 = std::chrono::steady_clock::now();
    total_comp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_decomp_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    total_cmp_bytes += static_cast<double>(blaz_comp_bytes(comp));

    const double avg_cmp_bytes = total_cmp_bytes;
    const double original_bytes = static_cast<double>(dim * sizeof(CTYPE));
    const double ratio =
        avg_cmp_bytes > 0.0 ? (original_bytes / avg_cmp_bytes) : 0.0;
    return BenchResult{true, "", total_comp_ms, total_decomp_ms,
        ratio, tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_sz3_interp_lorenzo_abs1e6_b64(
    const std::vector<CTYPE> &input, std::size_t dim) {
    const std::size_t n = input.size();
    std::vector<double> real(n), imag(n);
    for (std::size_t i = 0; i < n; ++i) {
        real[i] = std::real(input[i]);
        imag[i] = std::imag(input[i]);
    }

    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    double total_cmp_bytes = 0.0;
    std::vector<CTYPE> restored(n);
    SZ3::Config cfg_real =
        make_sz_config(dim, SZ3::EB_ABS, kGlobalAbsErrorBound,
            SZ3::ALGO_INTERP_LORENZO, 64);
    SZ3::Config cfg_imag =
        make_sz_config(dim, SZ3::EB_ABS, kGlobalAbsErrorBound,
            SZ3::ALGO_INTERP_LORENZO, 64);

    const auto t0 = std::chrono::steady_clock::now();
    std::size_t cmp_size_real = 0;
    std::size_t cmp_size_imag = 0;
    std::vector<char> cmp_real(sz_large_buffer_cap(cfg_real));
    std::vector<char> cmp_imag(sz_large_buffer_cap(cfg_imag));
    std::string err;
    if (!sz_compress_with_large_buffer(
            cfg_real, real.data(), cmp_real, cmp_size_real, err)) {
        return BenchResult{false, "sz3_real:" + err};
    }
    if (!sz_compress_with_large_buffer(
            cfg_imag, imag.data(), cmp_imag, cmp_size_imag, err)) {
        return BenchResult{false, "sz3_imag:" + err};
    }
    const auto t1 = std::chrono::steady_clock::now();

    SZ3::Config dec_cfg_real;
    SZ3::Config dec_cfg_imag;
    double *dec_real =
        SZ_decompress<double>(dec_cfg_real, cmp_real.data(), cmp_size_real);
    double *dec_imag =
        SZ_decompress<double>(dec_cfg_imag, cmp_imag.data(), cmp_size_imag);
    const auto t2 = std::chrono::steady_clock::now();

    total_comp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_decomp_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    total_cmp_bytes += static_cast<double>(cmp_size_real + cmp_size_imag);

    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(dec_real[i], dec_imag[i]);
    }

    delete[] dec_real;
    delete[] dec_imag;

    const double avg_cmp_bytes = total_cmp_bytes;
    const double original_bytes = static_cast<double>(dim * sizeof(CTYPE));
    const double ratio =
        avg_cmp_bytes > 0.0 ? (original_bytes / avg_cmp_bytes) : 0.0;
    return BenchResult{true, "", total_comp_ms, total_decomp_ms,
        ratio, tvd_prob(input, restored), fidelity(input, restored)};
}

BenchResult run_sz3_lorenzo_reg_rel1e4_b64(
    const std::vector<CTYPE> &input, std::size_t dim) {
    const std::size_t n = input.size();
    std::vector<double> real(n), imag(n);
    for (std::size_t i = 0; i < n; ++i) {
        real[i] = std::real(input[i]);
        imag[i] = std::imag(input[i]);
    }

    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    double total_cmp_bytes = 0.0;
    std::vector<CTYPE> restored(n);
    SZ3::Config cfg_real =
        make_sz_config(dim, SZ3::EB_REL, kGlobalRelErrorBound,
            SZ3::ALGO_LORENZO_REG, 64);
    SZ3::Config cfg_imag =
        make_sz_config(dim, SZ3::EB_REL, kGlobalRelErrorBound,
            SZ3::ALGO_LORENZO_REG, 64);

    const auto t0 = std::chrono::steady_clock::now();
    std::size_t cmp_size_real = 0;
    std::size_t cmp_size_imag = 0;
    std::vector<char> cmp_real(sz_large_buffer_cap(cfg_real));
    std::vector<char> cmp_imag(sz_large_buffer_cap(cfg_imag));
    std::string err;
    if (!sz_compress_with_large_buffer(
            cfg_real, real.data(), cmp_real, cmp_size_real, err)) {
        return BenchResult{false, "sz3_real:" + err};
    }
    if (!sz_compress_with_large_buffer(
            cfg_imag, imag.data(), cmp_imag, cmp_size_imag, err)) {
        return BenchResult{false, "sz3_imag:" + err};
    }
    const auto t1 = std::chrono::steady_clock::now();

    SZ3::Config dec_cfg_real;
    SZ3::Config dec_cfg_imag;
    double *dec_real =
        SZ_decompress<double>(dec_cfg_real, cmp_real.data(), cmp_size_real);
    double *dec_imag =
        SZ_decompress<double>(dec_cfg_imag, cmp_imag.data(), cmp_size_imag);
    const auto t2 = std::chrono::steady_clock::now();

    total_comp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_decomp_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    total_cmp_bytes += static_cast<double>(cmp_size_real + cmp_size_imag);

    for (std::size_t i = 0; i < n; ++i) {
        restored[i] = CTYPE(dec_real[i], dec_imag[i]);
    }

    delete[] dec_real;
    delete[] dec_imag;

    const double avg_cmp_bytes = total_cmp_bytes;
    const double original_bytes = static_cast<double>(dim * sizeof(CTYPE));
    const double ratio =
        avg_cmp_bytes > 0.0 ? (original_bytes / avg_cmp_bytes) : 0.0;
    return BenchResult{true, "", total_comp_ms, total_decomp_ms,
        ratio, tvd_prob(input, restored), fidelity(input, restored)};
}
}  // namespace

int benchmark_compressor(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const std::vector<UINT> qubits = {20, 22, 24, 26, 28};
    std::vector<std::size_t> dims;
    dims.reserve(qubits.size());
    for (UINT q : qubits) {
        dims.push_back(static_cast<std::size_t>(1ULL << q));
    }

    const std::vector<CompressorCase> cases = {
        {"blaz", "block64_keep3of4", &run_blaz_block64_keep3of4},
        {"blaz", "block128_keep1of2", &run_blaz_block128_keep1of2},
        {"sz3", "interp_lorenzo_abs1e-6_b64", &run_sz3_interp_lorenzo_abs1e6_b64},
        {"sz3", "lorenzo_reg_rel1e-4_b64", &run_sz3_lorenzo_reg_rel1e4_b64},
        {"zfp", "accuracy_abs_global", &run_zfp_accuracy_abs},
        {"mgard", "abs_global", &run_mgard_abs_global},
        {"sperr", "2d_pwe_abs_global", &run_sperr_2d_pwe_abs_global},
        {"fpzip", "2d_prec_abs_global", &run_fpzip_2d_precision_abs},
        {"xor_lz_zstd", "C_separate_ri", &run_xor_lz_bitplane_zstd_c},
        {"xor_lz_zstd", "D_interleave_ri", &run_xor_lz_bitplane_zstd_d},
    };

    const std::string out_root = "results/benchmark_compressor";
    ensure_output_dir(out_root);
    std::ofstream csv(out_root + "/metrics.csv");
    csv << "compressor,config,dim,family,compress_ms,decompress_ms,ratio,tvd,fidelity,status\n";

    std::cout << "compressor benchmark abs_eb=" << kGlobalAbsErrorBound
              << " rel_eb=" << kGlobalRelErrorBound << " dims=[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << dims[i];
    }
    std::cout << "]\n";

    std::mt19937 rng(12345);
    std::cout << std::setprecision(6) << std::scientific;
    using FamilyMaker = std::vector<CTYPE> (*)(std::size_t, std::mt19937 &);
    struct FamilyCase {
        const char *name;
        FamilyMaker make;
    };
    const std::vector<FamilyCase> family_cases = {
        {"random_gaussian", &make_random_gaussian},
        {"haar_random", &make_haar_random},
        {"low_freq", &make_low_freq_with_rng},
        {"high_freq", &make_high_freq_with_rng},
        {"sparse", &make_sparse},
    };

    for (std::size_t dim : dims) {
        for (const auto &fam_case : family_cases) {
            std::vector<CTYPE> family_state;
            try {
                family_state = fam_case.make(dim, rng);
            } catch (const std::bad_alloc &) {
                csv << "all,all," << dim << "," << fam_case.name
                    << ",0,0,0,0,0,skip:oom_family_build\n";
                std::cout << "dim=" << dim << " family=" << fam_case.name
                          << " status=skip:oom_family_build\n";
                continue;
            } catch (const std::exception &e) {
                csv << "all,all," << dim << "," << fam_case.name
                    << ",0,0,0,0,0,skip:family_build_error:" << e.what()
                    << "\n";
                std::cout << "dim=" << dim << " family=" << fam_case.name
                          << " status=skip:family_build_error:" << e.what()
                          << "\n";
                continue;
            } catch (...) {
                csv << "all,all," << dim << "," << fam_case.name
                    << ",0,0,0,0,0,skip:family_build_unknown\n";
                std::cout << "dim=" << dim << " family=" << fam_case.name
                          << " status=skip:family_build_unknown\n";
                continue;
            }
            for (const auto &c : cases) {
                BenchResult r;
                try {
                    r = c.run(family_state, dim);
                } catch (const std::bad_alloc &) {
                    r = BenchResult{false, "oom"};
                } catch (const std::exception &e) {
                    r = BenchResult{false, e.what()};
                } catch (...) {
                    r = BenchResult{false, "unknown_error"};
                }
                if (!r.ok) {
                    csv << c.compressor << "," << c.config << "," << dim << ","
                        << fam_case.name << ",0,0,0,0,0,skip:" << r.error << "\n";
                    std::cout << "dim=" << dim << " family=" << fam_case.name
                              << " compressor=" << c.compressor
                              << " config=" << c.config
                              << " status=skip:" << r.error << "\n";
                    continue;
                }

                csv << c.compressor << "," << c.config << "," << dim << ","
                    << fam_case.name << "," << r.compress_ms << "," << r.decompress_ms
                    << "," << r.ratio << "," << r.tvd << "," << r.fid << ",ok\n";
                std::cout << "dim=" << dim << " family=" << fam_case.name
                          << " compressor=" << c.compressor
                          << " config=" << c.config
                          << " compress_ms=" << r.compress_ms
                          << " decompress_ms=" << r.decompress_ms
                          << " ratio=" << r.ratio << " tvd=" << r.tvd
                          << " fidelity=" << r.fid << "\n";
            }
        }
    }
    std::cout << "wrote " << out_root << "/metrics.csv\n";
    return 0;
}
