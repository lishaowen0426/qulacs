#include "state_quantize.hpp"

#include <cmath>
#include <csim/quantize_config.hpp>
#include <csim/utility.hpp>

void QuantumStateCpuQuant::set_error_bound(double error_bound) {
    double synced_error_bound = error_bound;
#ifdef _USE_MPI
    // In MPI mode, keep a single error-bound value across all ranks.
    MPIutil& mpiutil = MPIutil::get_inst();
    if (mpiutil.get_size() > 1) {
        mpiutil.s_D_bcast(&synced_error_bound);
    }
#endif
    if (synced_error_bound < 0.0) {
        throw std::invalid_argument(
            "Error: QuantumStateCpuQuant::set_error_bound(double): "
            "error_bound must be non-negative");
    }
    set_quantize_error_bound(synced_error_bound);
}

double QuantumStateCpuQuant::get_error_bound() {
    return get_quantize_error_bound();
}

void QuantumStateCpuQuant::quantize() {
    const double error_bound = get_quantize_error_bound();
    if (error_bound <= 0.0) return;

    const double step = get_quantize_step();
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(this->_dim, 10);
#pragma omp parallel for
#endif
    for (ITYPE idx = 0; idx < this->_dim; ++idx) {
        const double real_part = std::real(this->_state_vector[idx]);
        const double imag_part = std::imag(this->_state_vector[idx]);
        const double snapped_real = step * std::round(real_part / step);
        const double snapped_imag = step * std::round(imag_part / step);
        this->_state_vector[idx] = CPPCTYPE(snapped_real, snapped_imag);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}
