#include "quantize_config.hpp"

namespace {
double global_quantize_error_bound = 1e-6;
}

double get_quantize_error_bound() { return global_quantize_error_bound; }

void set_quantize_error_bound(double error_bound) {
    global_quantize_error_bound = error_bound;
}

double get_quantize_step() { return 2.0 * global_quantize_error_bound; }
