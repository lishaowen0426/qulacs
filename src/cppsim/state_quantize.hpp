#pragma once

#include "state.hpp"

class QuantumStateCpuQuant : public QuantumStateCpu {
public:
    using QuantumStateCpu::QuantumStateCpu;
    virtual ~QuantumStateCpuQuant() override = default;

    static void set_error_bound(double error_bound);
    static double get_error_bound();
    void quantize();
};
