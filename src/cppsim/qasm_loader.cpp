#include "qasm_loader.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gate_factory.hpp"

namespace {
std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() &&
           std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

std::string normalize_qasm_instruction(const std::string& line) {
    std::string out;
    out.reserve(line.size());
    for (char c : line) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        out.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

constexpr double kQasmPi = 3.141592653589793238462643383279502884;

[[noreturn]] void throw_parse_error(
    size_t line_no_1based, const std::string& message) {
    std::stringstream ss;
    ss << "QASM parse error at line " << line_no_1based << ": " << message;
    throw std::runtime_error(ss.str());
}

UINT parse_uint_strict(
    const std::string& token, size_t line_no_1based, const std::string& ctx) {
    if (token.empty()) throw_parse_error(line_no_1based, "empty integer in " + ctx);
    size_t consumed = 0;
    UINT value = 0;
    try {
        value = static_cast<UINT>(std::stoull(token, &consumed));
    } catch (...) {
        throw_parse_error(line_no_1based, "invalid integer '" + token + "' in " + ctx);
    }
    if (consumed != token.size()) {
        throw_parse_error(line_no_1based, "invalid integer '" + token + "' in " + ctx);
    }
    return value;
}

double parse_double_strict(
    const std::string& token, size_t line_no_1based, const std::string& ctx) {
    if (token.empty()) throw_parse_error(line_no_1based, "empty number in " + ctx);

    std::function<double(size_t&)> parse_expr;
    std::function<double(size_t&)> parse_term;
    std::function<double(size_t&)> parse_factor;

    parse_factor = [&](size_t& pos) -> double {
        if (pos >= token.size()) {
            throw_parse_error(line_no_1based, "invalid number '" + token + "' in " + ctx);
        }

        if (token[pos] == '+') {
            ++pos;
            return parse_factor(pos);
        }
        if (token[pos] == '-') {
            ++pos;
            return -parse_factor(pos);
        }
        if (token[pos] == '(') {
            ++pos;
            const double value = parse_expr(pos);
            if (pos >= token.size() || token[pos] != ')') {
                throw_parse_error(
                    line_no_1based, "missing ')' in number '" + token + "' in " + ctx);
            }
            ++pos;
            return value;
        }
        if (token.compare(pos, 2, "pi") == 0) {
            pos += 2;
            return kQasmPi;
        }

        char* end_ptr = nullptr;
        const char* begin_ptr = token.c_str() + pos;
        const double value = std::strtod(begin_ptr, &end_ptr);
        if (end_ptr == begin_ptr) {
            throw_parse_error(line_no_1based, "invalid number '" + token + "' in " + ctx);
        }
        pos = static_cast<size_t>(end_ptr - token.c_str());
        return value;
    };

    parse_term = [&](size_t& pos) -> double {
        double value = parse_factor(pos);
        while (pos < token.size()) {
            const char op = token[pos];
            if (op != '*' && op != '/') break;
            ++pos;
            const double rhs = parse_factor(pos);
            if (op == '*') {
                value *= rhs;
            } else {
                if (rhs == 0.0) {
                    throw_parse_error(
                        line_no_1based, "division by zero in number '" + token + "'");
                }
                value /= rhs;
            }
        }
        return value;
    };

    parse_expr = [&](size_t& pos) -> double {
        double value = parse_term(pos);
        while (pos < token.size()) {
            const char op = token[pos];
            if (op != '+' && op != '-') break;
            ++pos;
            const double rhs = parse_term(pos);
            if (op == '+') {
                value += rhs;
            } else {
                value -= rhs;
            }
        }
        return value;
    };

    size_t pos = 0;
    const double value = parse_expr(pos);
    if (pos != token.size()) {
        throw_parse_error(line_no_1based, "invalid number '" + token + "' in " + ctx);
    }
    return value;
}

UINT mapped_qubit_index(const std::vector<UINT>& mapping, UINT raw_index,
    size_t line_no_1based, const std::string& ctx) {
    if (raw_index >= mapping.size()) {
        throw_parse_error(line_no_1based,
            "qubit index " + std::to_string(raw_index) +
                " out of mapping range in " + ctx);
    }
    return mapping[raw_index];
}

bool parse_qreg(
    const std::string& instr, UINT* qubit_count, size_t line_no_1based) {
    static const std::string prefix = "qregq[";
    if (!starts_with(instr, prefix)) return false;
    const size_t close = instr.find("];", prefix.size());
    if (close == std::string::npos || close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid qreg format");
    }
    *qubit_count = parse_uint_strict(
        instr.substr(prefix.size(), close - prefix.size()), line_no_1based, "qreg");
    return true;
}

bool parse_one_qubit_gate_no_param(const std::string& instr, const std::string& op,
    UINT* target, size_t line_no_1based) {
    const std::string prefix = op + "q[";
    if (!starts_with(instr, prefix)) return false;
    const size_t close = instr.find("];", prefix.size());
    if (close == std::string::npos || close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid " + op + " format");
    }
    *target = parse_uint_strict(
        instr.substr(prefix.size(), close - prefix.size()), line_no_1based, op);
    return true;
}

bool parse_two_qubit_gate_no_param(const std::string& instr, const std::string& op,
    UINT* q0, UINT* q1, size_t line_no_1based) {
    const std::string prefix = op + "q[";
    if (!starts_with(instr, prefix)) return false;
    const size_t mid = instr.find("],q[", prefix.size());
    const size_t close = (mid == std::string::npos) ? std::string::npos
                                                     : instr.find("];", mid + 4);
    if (mid == std::string::npos || close == std::string::npos ||
        close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid " + op + " format");
    }
    *q0 = parse_uint_strict(
        instr.substr(prefix.size(), mid - prefix.size()), line_no_1based, op);
    *q1 = parse_uint_strict(
        instr.substr(mid + 4, close - (mid + 4)), line_no_1based, op);
    return true;
}

bool parse_one_param_one_qubit_gate(const std::string& instr,
    const std::string& op, double* param, UINT* target, size_t line_no_1based) {
    const std::string prefix = op + "(";
    if (!starts_with(instr, prefix)) return false;
    const size_t sep = instr.find(")q[", prefix.size());
    const size_t close =
        (sep == std::string::npos) ? std::string::npos : instr.find("];", sep + 3);
    if (sep == std::string::npos || close == std::string::npos ||
        close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid " + op + " format");
    }
    *param = parse_double_strict(
        instr.substr(prefix.size(), sep - prefix.size()), line_no_1based, op);
    *target = parse_uint_strict(
        instr.substr(sep + 3, close - (sep + 3)), line_no_1based, op);
    return true;
}

bool parse_three_param_one_qubit_gate(const std::string& instr,
    const std::string& op, double* p0, double* p1, double* p2, UINT* target,
    size_t line_no_1based) {
    const std::string prefix = op + "(";
    if (!starts_with(instr, prefix)) return false;
    const size_t sep = instr.find(")q[", prefix.size());
    const size_t close =
        (sep == std::string::npos) ? std::string::npos : instr.find("];", sep + 3);
    if (sep == std::string::npos || close == std::string::npos ||
        close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid " + op + " format");
    }
    const std::string params = instr.substr(prefix.size(), sep - prefix.size());
    const size_t comma0 = params.find(',');
    const size_t comma1 = (comma0 == std::string::npos)
                              ? std::string::npos
                              : params.find(',', comma0 + 1);
    if (comma0 == std::string::npos || comma1 == std::string::npos ||
        params.find(',', comma1 + 1) != std::string::npos) {
        throw_parse_error(line_no_1based, "invalid " + op + " parameter count");
    }
    *p0 = parse_double_strict(params.substr(0, comma0), line_no_1based, op);
    *p1 = parse_double_strict(
        params.substr(comma0 + 1, comma1 - comma0 - 1), line_no_1based, op);
    *p2 = parse_double_strict(params.substr(comma1 + 1), line_no_1based, op);
    *target = parse_uint_strict(
        instr.substr(sep + 3, close - (sep + 3)), line_no_1based, op);
    return true;
}

bool parse_two_param_one_qubit_gate(const std::string& instr,
    const std::string& op, double* p0, double* p1, UINT* target,
    size_t line_no_1based) {
    const std::string prefix = op + "(";
    if (!starts_with(instr, prefix)) return false;
    const size_t sep = instr.find(")q[", prefix.size());
    const size_t close =
        (sep == std::string::npos) ? std::string::npos : instr.find("];", sep + 3);
    if (sep == std::string::npos || close == std::string::npos ||
        close + 2 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid " + op + " format");
    }
    const std::string params = instr.substr(prefix.size(), sep - prefix.size());
    const size_t comma = params.find(',');
    if (comma == std::string::npos || params.find(',', comma + 1) != std::string::npos) {
        throw_parse_error(line_no_1based, "invalid " + op + " parameter count");
    }
    *p0 = parse_double_strict(params.substr(0, comma), line_no_1based, op);
    *p1 = parse_double_strict(params.substr(comma + 1), line_no_1based, op);
    *target = parse_uint_strict(
        instr.substr(sep + 3, close - (sep + 3)), line_no_1based, op);
    return true;
}

bool parse_qubits_remap(
    const std::string& instr, UINT* src, UINT* dst, size_t line_no_1based) {
    if (!starts_with(instr, "//q[")) return false;
    const size_t mid = instr.find("]-->q[");
    const size_t close = (mid == std::string::npos) ? std::string::npos
                                                     : instr.find(']', mid + 6);
    if (mid == std::string::npos || close == std::string::npos ||
        close + 1 != instr.size()) {
        throw_parse_error(line_no_1based, "invalid remap comment format");
    }
    *src = parse_uint_strict(instr.substr(4, mid - 4), line_no_1based, "remap src");
    *dst = parse_uint_strict(
        instr.substr(mid + 6, close - (mid + 6)), line_no_1based, "remap dst");
    return true;
}

bool parse_qubits_count_comment(
    const std::string& instr, UINT* qubit_count, size_t line_no_1based) {
    static const std::string prefix = "//qubits:";
    if (!starts_with(instr, prefix)) return false;
    *qubit_count = parse_uint_strict(
        instr.substr(prefix.size()), line_no_1based, "qubits comment");
    return true;
}

}  // namespace

std::unique_ptr<QuantumCircuit> convert_QASM_to_quantum_circuit(
    const std::vector<std::string>& input_lines, bool remap_remove) {
    std::unique_ptr<QuantumCircuit> circuit;
    std::vector<UINT> mapping;

    for (size_t line_no = 0; line_no < input_lines.size(); ++line_no) {
        const size_t line_no_1based = line_no + 1;
        std::string line = trim(input_lines[line_no]);
        if (line.empty()) continue;

        // Handle full-line comments before removing inline comments.
        if (starts_with(line, "//")) {
            const std::string comment = normalize_qasm_instruction(line);
            if (remap_remove) {
                UINT raw = 0, mapped = 0;
                if (parse_qubits_remap(comment, &raw, &mapped, line_no_1based)) {
                    if (raw >= mapping.size()) {
                        throw_parse_error(line_no_1based,
                            "remap source index out of range: " + std::to_string(raw));
                    }
                    mapping[raw] = mapped;
                    continue;
                }
                UINT qubit_count = 0;
                if (parse_qubits_count_comment(comment, &qubit_count, line_no_1based)) {
                    mapping.resize(qubit_count);
                    for (UINT i = 0; i < qubit_count; ++i) mapping[i] = i;
                    continue;
                }
            }
            continue;
        }

        const size_t inline_comment_pos = line.find("//");
        if (inline_comment_pos != std::string::npos) {
            line = line.substr(0, inline_comment_pos);
        }

        const std::string instr = normalize_qasm_instruction(line);
        if (instr.empty()) continue;

        if (instr == "openqasm2.0;" || instr == "openqasm3.0;" ||
            instr == "include\"qelib1.inc\";" || instr == "include\"stdgates.inc\";") {
            continue;
        }
        if (starts_with(instr, "creg") || starts_with(instr, "measure") ||
            starts_with(instr, "barrier")) {
            continue;
        }

        UINT qcount = 0;
        if (parse_qreg(instr, &qcount, line_no_1based)) {
            circuit = std::unique_ptr<QuantumCircuit>(new QuantumCircuit(qcount));
            if (mapping.empty()) {
                mapping.resize(qcount);
                for (UINT i = 0; i < qcount; ++i) mapping[i] = i;
            }
            continue;
        }
        if (!circuit) {
            throw_parse_error(
                line_no_1based, "qreg must appear before gate instructions");
        }
        if (mapping.empty()) {
            mapping.resize(circuit->qubit_count);
            for (UINT i = 0; i < circuit->qubit_count; ++i) mapping[i] = i;
        }

        UINT q0 = 0, q1 = 0;
        if (parse_two_qubit_gate_no_param(instr, "cx", &q0, &q1, line_no_1based)) {
            circuit->add_CNOT_gate(mapped_qubit_index(mapping, q0, line_no_1based, "cx"),
                mapped_qubit_index(mapping, q1, line_no_1based, "cx"));
            continue;
        }
        if (parse_two_qubit_gate_no_param(instr, "cz", &q0, &q1, line_no_1based)) {
            circuit->add_CZ_gate(mapped_qubit_index(mapping, q0, line_no_1based, "cz"),
                mapped_qubit_index(mapping, q1, line_no_1based, "cz"));
            continue;
        }
        if (parse_two_qubit_gate_no_param(instr, "swap", &q0, &q1, line_no_1based)) {
            circuit->add_SWAP_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "swap"),
                mapped_qubit_index(mapping, q1, line_no_1based, "swap"));
            continue;
        }

        if (parse_one_qubit_gate_no_param(instr, "id", &q0, line_no_1based)) {
            circuit->add_gate(gate::Identity(
                mapped_qubit_index(mapping, q0, line_no_1based, "id")));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "x", &q0, line_no_1based)) {
            circuit->add_X_gate(mapped_qubit_index(mapping, q0, line_no_1based, "x"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "y", &q0, line_no_1based)) {
            circuit->add_Y_gate(mapped_qubit_index(mapping, q0, line_no_1based, "y"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "z", &q0, line_no_1based)) {
            circuit->add_Z_gate(mapped_qubit_index(mapping, q0, line_no_1based, "z"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "h", &q0, line_no_1based)) {
            circuit->add_H_gate(mapped_qubit_index(mapping, q0, line_no_1based, "h"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "s", &q0, line_no_1based)) {
            circuit->add_S_gate(mapped_qubit_index(mapping, q0, line_no_1based, "s"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "sdg", &q0, line_no_1based)) {
            circuit->add_Sdag_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "sdg"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "t", &q0, line_no_1based)) {
            circuit->add_T_gate(mapped_qubit_index(mapping, q0, line_no_1based, "t"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "tdg", &q0, line_no_1based)) {
            circuit->add_Tdag_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "tdg"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "sx", &q0, line_no_1based)) {
            circuit->add_sqrtX_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "sx"));
            continue;
        }
        if (parse_one_qubit_gate_no_param(instr, "sxdg", &q0, line_no_1based)) {
            circuit->add_sqrtXdag_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "sxdg"));
            continue;
        }

        double p0 = 0.0, p1 = 0.0, p2 = 0.0;
        if (parse_one_param_one_qubit_gate(
                instr, "rx", &p0, &q0, line_no_1based)) {
            circuit->add_RX_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "rx"), -p0);
            continue;
        }
        if (parse_one_param_one_qubit_gate(
                instr, "ry", &p0, &q0, line_no_1based)) {
            circuit->add_RY_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "ry"), -p0);
            continue;
        }
        if (parse_one_param_one_qubit_gate(
                instr, "rz", &p0, &q0, line_no_1based)) {
            circuit->add_RZ_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "rz"), -p0);
            continue;
        }
        if (parse_one_param_one_qubit_gate(instr, "p", &p0, &q0, line_no_1based) ||
            parse_one_param_one_qubit_gate(
                instr, "u1", &p0, &q0, line_no_1based)) {
            circuit->add_U1_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "u1/p"), p0);
            continue;
        }
        if (parse_two_param_one_qubit_gate(
                instr, "u2", &p0, &p1, &q0, line_no_1based)) {
            circuit->add_U2_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "u2"), p0, p1);
            continue;
        }
        if (parse_three_param_one_qubit_gate(
                instr, "u3", &p0, &p1, &p2, &q0, line_no_1based) ||
            parse_three_param_one_qubit_gate(
                instr, "u", &p0, &p1, &p2, &q0, line_no_1based)) {
            circuit->add_U3_gate(
                mapped_qubit_index(mapping, q0, line_no_1based, "u3/u"), p0, p1, p2);
            continue;
        }

        throw_parse_error(line_no_1based, "unknown or unsupported instruction: " + instr);
    }

    if (!circuit) {
        throw std::runtime_error("QASM parse error: no qreg found");
    }
    return circuit;
}

std::unique_ptr<QuantumCircuit> load_qasm_file(
    const std::string& qasm_file_path, bool remap_remove) {
    std::ifstream ifs(qasm_file_path);
    if (!ifs.good()) {
        throw std::runtime_error(
            "failed to open qasm file: " + qasm_file_path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        lines.push_back(line);
    }
    return convert_QASM_to_quantum_circuit(lines, remap_remove);
}
