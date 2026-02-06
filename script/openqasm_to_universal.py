#!/usr/bin/env python3

from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump as qasm2_dump


BASIS_GATES = ["rz", "sx", "cx"]
SOURCE_ROOT = Path("compressor/QASMBench/medium")
OUTPUT_SUFFIX = "_" + "_".join(BASIS_GATES)


def pick_source_qasm(folder: Path) -> Path | None:
    candidates = sorted(
        qasm_file
        for qasm_file in folder.glob("*.qasm")
        if not qasm_file.stem.endswith("_transpiled")
        and not qasm_file.stem.endswith(OUTPUT_SUFFIX)
    )
    if not candidates:
        return None

    preferred_name = f"{folder.name}.qasm"
    for qasm_file in candidates:
        if qasm_file.name == preferred_name:
            return qasm_file
    return candidates[0]


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Input directory not found: {SOURCE_ROOT}")

    processed = 0
    skipped = 0

    for subfolder in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir()):
        source_qasm = pick_source_qasm(subfolder)
        if source_qasm is None:
            print(f"[skip] {subfolder}: no source .qasm file found")
            skipped += 1
            continue

        output_qasm = subfolder / f"{source_qasm.stem}{OUTPUT_SUFFIX}.qasm"

        try:
            circuit = QuantumCircuit.from_qasm_file(str(source_qasm))
            transpiled_circuit = transpile(circuit, basis_gates=BASIS_GATES)
            with output_qasm.open("w", encoding="utf-8") as f:
                qasm2_dump(transpiled_circuit, f)
            print(f"[ok] {source_qasm} -> {output_qasm}")
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {source_qasm}: {exc}")
            skipped += 1

    print(f"Done. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
