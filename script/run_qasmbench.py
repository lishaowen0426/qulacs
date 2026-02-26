#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

QASMBENCH_ROOT = Path("compressor/QASMBench")
OUTPUT_ROOT = Path("results/QASMBenchOutput")
QASMBENCH_FOLDERS = [
    QASMBENCH_ROOT / "small",
    QASMBENCH_ROOT / "medium",
    QASMBENCH_ROOT / "large",
]


def create_circuit_from_qasm(qasm_path: str | Path) -> QuantumCircuit:
    path = Path(qasm_path)
    if not path.is_file():
        raise FileNotFoundError(f"QASM file not found: {path}")
    return QuantumCircuit.from_qasm_file(str(path))


def run_circuit_from_zero_state(circuit: QuantumCircuit) -> Statevector:
    unitary_circuit = circuit.remove_final_measurements(inplace=False)
    input_state = Statevector.from_label("0" * unitary_circuit.num_qubits)
    return input_state.evolve(unitary_circuit)


def get_output_path_for_qasm(
    qasm_path: str | Path,
    qasmbench_root: Path = QASMBENCH_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    qasm_path_obj = Path(qasm_path)
    try:
        relative_path = qasm_path_obj.relative_to(qasmbench_root)
    except ValueError:
        qasm_abs = qasm_path_obj.resolve()
        root_abs = qasmbench_root.resolve()
        relative_path = qasm_abs.relative_to(root_abs)
    return output_root / relative_path.parent / "output"


def cached_result_matches_input(output_path: Path, input_basis_state: str) -> bool:
    if not output_path.is_file():
        return False
    with output_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    return first_line == input_basis_state


def write_statevector_output(
    output_path: Path, input_basis_state: str, final_state: Statevector
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        # First line stores the basis-state input bitstring.
        f.write(f"{input_basis_state}\n")
        for amp in final_state.data:
            f.write(f"{amp.real:.17g} {amp.imag:.17g}\n")


def _select_normal_qasm(circuit_dir: Path) -> Path | None:
    preferred = circuit_dir / f"{circuit_dir.name}.qasm"
    if preferred.is_file():
        return preferred

    candidates = sorted(
        p
        for p in circuit_dir.iterdir()
        if p.is_file() and p.suffix == ".qasm" and "_transpiled" not in p.stem
    )
    if not candidates:
        return None
    return candidates[0]


def iter_qasmbench_circuits() -> list[Path]:
    selected: list[Path] = []
    for folder in QASMBENCH_FOLDERS:
        if not folder.is_dir():
            print(f"Skip missing folder: {folder}")
            continue
        for circuit_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            selected_qasm = _select_normal_qasm(circuit_dir)
            if selected_qasm is None:
                print(f"[skip] no normal .qasm found in {circuit_dir}")
                continue
            selected.append(selected_qasm)
    return selected


def process_single_circuit(qasm_path: Path) -> str:
    circuit = create_circuit_from_qasm(qasm_path)
    unitary_circuit = circuit.remove_final_measurements(inplace=False)
    input_basis_state = "0" * unitary_circuit.num_qubits
    output_path = get_output_path_for_qasm(qasm_path)

    if cached_result_matches_input(output_path, input_basis_state):
        print(f"[cached] {qasm_path} -> {output_path}")
        return "cached"

    final_state = run_circuit_from_zero_state(circuit)
    write_statevector_output(output_path, input_basis_state, final_state)
    print(
        f"[saved] {qasm_path} -> {output_path} "
        f"(qubits={circuit.num_qubits}, dim={len(final_state)})"
    )
    return "computed"


def main() -> None:
    circuit_paths = iter_qasmbench_circuits()
    print(f"Discovered {len(circuit_paths)} circuit folders to process.")

    cached_count = 0
    computed_count = 0
    failed_count = 0

    for i, qasm_path in enumerate(circuit_paths, start=1):
        print(f"[{i}/{len(circuit_paths)}] {qasm_path}")
        try:
            status = process_single_circuit(qasm_path)
            if status == "cached":
                cached_count += 1
            else:
                computed_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"[failed] {qasm_path}: {type(exc).__name__}: {exc}")

    print(
        "Done. "
        f"computed={computed_count}, cached={cached_count}, failed={failed_count}"
    )


if __name__ == "__main__":
    main()
