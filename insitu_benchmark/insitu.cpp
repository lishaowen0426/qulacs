#include "insitu.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "SZ3/api/sz.hpp"
#include "cppsim/circuit.hpp"
#include "cppsim/gate_factory.hpp"
#include "csim/MPIutil.hpp"
#include "spdlog/spdlog.h"

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
    spdlog::set_pattern("%v");
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
            "comp_time_ratio_min={:.6f} comp_time_ratio_max={:.6f} comp_time_ratio_avg={:.6f}",
            ratio_min, ratio_max, ratio_avg);
        const double comp_ratio_avg_all =
            comp_ratio_sum / static_cast<double>(size);
        spdlog::info(
            "comp_bytes_ratio_min={:.6f} comp_bytes_ratio_max={:.6f} comp_bytes_ratio_avg={:.6f}",
            comp_ratio_min, comp_ratio_max, comp_ratio_avg_all);
    }

    return 0;
}
