#pragma once

#include <functional>

#include "SZ3/api/sz.hpp"
#include "cppsim/circuit.hpp"
#include "cppsim/circuit_builder.hpp"
#include "cppsim/gate.hpp"
#include "cppsim/state.hpp"

class RikenCircuitBuilder : public QuantumCircuitBuilder {
public:
    void step_gate(const QuantumCircuit *circuit, QuantumStateBase *state,
        const std::function<void(const QuantumGateBase *,
            const QuantumStateBase *)> &on_step) const {
        for (auto *gate : circuit->gate_list) {
            gate->update_quantum_state(state);
            on_step(gate, state);
        }
    }
};

class QFTCircuitBuilder : public RikenCircuitBuilder {
public:
    QFTCircuitBuilder(UINT qubit_count);
    QuantumCircuit *create_circuit(UINT qubit_count) const override;

private:
    UINT qubit_count_;
};

/*
void apply_compression_before_epoch_h(QuantumCircuit *circuit,
    QuantumStateBase *input, UINT epoch, double error_bound, SZ3::EB error_mode,
    double &reduced_ratio);

void compression_on_h_mpi(QuantumCircuit *circuit, QuantumStateBase *input,
    UINT epoch, double error_bound, SZ3::EB error_mode, double &reduced_ratio);
    */

int run_comp_on_mpi(int argc, char **argv);
int benchmark_blaz(int argc, char **argv);
int benchmark_quantization(int argc, char **argv);

int benchmark_compressor(int argc, char **argv);

