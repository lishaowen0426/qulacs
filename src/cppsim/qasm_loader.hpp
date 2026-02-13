#pragma once

#include <memory>
#include <string>
#include <vector>

#include "circuit.hpp"

/**
 * Parse OpenQASM lines and build a QuantumCircuit.
 *
 * Supported instructions are a minimal subset aligned with the Python
 * converter: qreg, x/y/z/h/s/sdg/t/tdg/sx/sxdg, cx/cz/swap, rx/ry/rz,
 * p/u1/u2/u3/u.
 */
DllExport std::unique_ptr<QuantumCircuit> convert_QASM_to_quantum_circuit(
    const std::vector<std::string>& input_lines, bool remap_remove = false);

/**
 * Load a .qasm file and build a QuantumCircuit.
 */
DllExport std::unique_ptr<QuantumCircuit> load_qasm_file(
    const std::string& qasm_file_path, bool remap_remove = false);
