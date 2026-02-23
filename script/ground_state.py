#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml


DEFAULT_ATTRIBUTES = [
    "ground_states",
    "shadow_basis",
    "shadow_meas",
    "spin_system",
]


def _as_array(value: Any) -> np.ndarray | None:
    """Best-effort conversion to a NumPy array."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object and arr.size > 0:
        return None
    return arr


def _dataset_file_path(dataset: qml.data.Dataset) -> str:
    """Return the backing HDF5 file path if available."""
    try:
        return str(dataset.bind.file.filename)
    except Exception:
        return "<unknown>"


def _summarize_array(name: str, arr: np.ndarray) -> None:
    print(f"[{name}] type=np.ndarray shape={arr.shape} dtype={arr.dtype}")
    if arr.size == 0:
        return
    if np.issubdtype(arr.dtype, np.number):
        if np.iscomplexobj(arr):
            mag = np.abs(arr)
            print(
                f"[{name}] |x| min={mag.min():.6g}, max={mag.max():.6g}, mean={mag.mean():.6g}"
            )
        else:
            print(
                f"[{name}] min={arr.min():.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}"
            )


def _print_value_summary(name: str, value: Any) -> None:
    arr = _as_array(value)
    if arr is not None:
        _summarize_array(name, arr)
        return

    if isinstance(value, (list, tuple)):
        print(f"[{name}] type={type(value).__name__} len={len(value)}")
        if len(value) > 0:
            first = value[0]
            first_arr = _as_array(first)
            if first_arr is not None:
                print(
                    f"[{name}] first element array shape={first_arr.shape} dtype={first_arr.dtype}"
                )
            else:
                print(f"[{name}] first element type={type(first).__name__}")
        return

    print(f"[{name}] type={type(value).__name__}")


def _statevector_rows(ground_states: Any) -> np.ndarray | None:
    arr = _as_array(ground_states)
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim >= 2:
        last_dim = arr.shape[-1]
        return arr.reshape(-1, last_dim)
    return None


def _top_basis_probabilities(state: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
    probs = np.abs(state) ** 2
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [(int(i), float(probs[i])) for i in idx]


def _n_qubits_from_state(state: np.ndarray) -> int | None:
    dim = state.size
    if dim <= 0:
        return None
    n = int(round(math.log2(dim)))
    if 2**n != dim:
        return None
    return n


def _z_expectations_from_probs(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    expvals = np.zeros(n_qubits, dtype=float)
    for basis_index, p in enumerate(probs):
        for q in range(n_qubits):
            bit = (basis_index >> (n_qubits - 1 - q)) & 1
            expvals[q] += p if bit == 0 else -p
    return expvals


def analyze_ground_states(ground_states: Any, top_k: int) -> None:
    rows = _statevector_rows(ground_states)
    if rows is None:
        print("[ground_states] could not parse as numeric array")
        return

    print(f"[ground_states] parsed as {rows.shape[0]} state vector(s), dim={rows.shape[1]}")
    max_rows = min(rows.shape[0], 3)
    for i in range(max_rows):
        state = rows[i]
        norm = float(np.vdot(state, state).real)
        print(f"[ground_states] state[{i}] norm={norm:.12f}")

        n_qubits = _n_qubits_from_state(state)
        if n_qubits is not None:
            probs = np.abs(state) ** 2
            z_exp = _z_expectations_from_probs(probs, n_qubits)
            z_exp_str = ", ".join(f"{v:.4f}" for v in z_exp)
            print(f"[ground_states] state[{i}] <Z_q> = [{z_exp_str}]")

            top = _top_basis_probabilities(state, k=min(top_k, state.size))
            for basis_idx, p in top:
                bitstring = format(basis_idx, f"0{n_qubits}b")
                print(f"[ground_states] state[{i}] top basis |{bitstring}> prob={p:.6f}")
        else:
            print("[ground_states] state dimension is not a power of 2; skipped qubit-wise summary")


def print_entire_statevector(ground_states: Any, state_index: int) -> None:
    rows = _statevector_rows(ground_states)
    if rows is None:
        print("[statevector] could not parse ground_states as a numeric array")
        return

    if state_index < 0 or state_index >= rows.shape[0]:
        print(
            f"[statevector] invalid --state-index={state_index}; valid range is [0, {rows.shape[0] - 1}]"
        )
        return

    state = rows[state_index]
    with np.printoptions(threshold=np.inf, linewidth=200, precision=12, suppress=False):
        print(f"[statevector] index={state_index} dim={state.size}")
        print(state)


def analyze_shadow_data(shadow_basis: Any, shadow_meas: Any) -> None:
    basis = _as_array(shadow_basis)
    meas = _as_array(shadow_meas)
    if basis is None or meas is None:
        print("[shadow] could not parse shadow_basis or shadow_meas as numeric arrays")
        return

    print(f"[shadow] basis shape={basis.shape}, meas shape={meas.shape}")
    if basis.shape != meas.shape:
        print("[shadow] warning: basis and measurement arrays have different shapes")

    flat_basis = basis.reshape(-1)
    flat_meas = meas.reshape(-1)
    basis_counts = Counter(int(x) for x in flat_basis.tolist())
    meas_counts = Counter(int(x) for x in flat_meas.tolist())
    print(f"[shadow] basis label counts: {dict(sorted(basis_counts.items()))}")
    print(f"[shadow] meas value counts: {dict(sorted(meas_counts.items()))}")

    if basis.ndim == 2 and meas.ndim == 2 and basis.shape == meas.shape:
        n_shots, n_qubits = basis.shape
        print(f"[shadow] interpreted as {n_shots} shots on {n_qubits} qubits")
        z_mask = basis == 2
        if np.any(z_mask):
            z_meas = meas[z_mask]
            z_avg = float(np.mean(1 - 2 * z_meas))
            print(f"[shadow] global <Z> estimate from Z-basis measurements only: {z_avg:.6f}")
        else:
            print("[shadow] no Z-basis measurements found (basis label 2)")


def analyze_spin_system(spin_system: Any) -> None:
    print(f"[spin_system] type={type(spin_system).__name__}")

    if hasattr(spin_system, "terms"):
        try:
            coeffs, ops = spin_system.terms()
            coeff_arr = np.asarray(coeffs, dtype=np.complex128)
            print(f"[spin_system] number of terms={len(coeff_arr)}")
            print(f"[spin_system] coeff |x| sum={np.abs(coeff_arr).sum():.6g}")
            if len(ops) > 0:
                print(f"[spin_system] first term operator={ops[0]}")
            return
        except Exception:
            pass

    if hasattr(spin_system, "shape"):
        shape = getattr(spin_system, "shape")
        print(f"[spin_system] shape={shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and analyze a PennyLane qspin/FermiHubbard dataset."
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Analyze an existing local .h5 dataset file directly (skip qml.data.load).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--sysname", type=str, default="FermiHubbard")
    parser.add_argument("--periodicity", type=str, default="open")
    parser.add_argument("--lattice", type=str, default="chain")
    parser.add_argument("--layout", type=str, default="1x4")
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=DEFAULT_ATTRIBUTES,
        help="Dataset attributes to request.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top basis states to print.")
    parser.add_argument(
        "--print-statevector",
        action="store_true",
        help="Print the full complex amplitudes of one ground-state vector.",
    )
    parser.add_argument(
        "--state-index",
        type=int,
        default=0,
        help="Ground-state index to print when --print-statevector is used.",
    )
    parser.add_argument(
        "--num-threads", type=int, default=20, help="Max download threads for qml.data.load."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when data already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_file is not None:
        ds = qml.data.Dataset.open(args.dataset_file.expanduser(), mode="r")
        datasets = [ds]
    else:
        try:
            datasets = qml.data.load(
                "qspin",
                sysname=args.sysname,
                periodicity=args.periodicity,
                lattice=args.lattice,
                layout=args.layout,
                attributes=args.attributes,
                folder_path=args.data_dir,
                num_threads=args.num_threads,
                force=args.force,
            )
        except Exception as exc:
            raise RuntimeError(
                "qml.data.load failed. If you already have the dataset downloaded, run this script "
                "with --dataset-file /path/to/file.h5 to analyze it offline."
            ) from exc

    if not datasets:
        raise RuntimeError("No datasets were returned by qml.data.load.")

    ds = datasets[0]
    print(f"[dataset] loaded {len(datasets)} dataset(s)")
    print(f"[dataset] local file: {_dataset_file_path(ds)}")
    print(f"[dataset] identifiers: {dict(ds.identifiers)}")
    print(f"[dataset] available attributes in file: {ds.list_attributes()}")

    if "ground_states" in ds.list_attributes():
        _print_value_summary("ground_states", ds.ground_states)
        analyze_ground_states(ds.ground_states, top_k=args.top_k)
        if args.print_statevector:
            print_entire_statevector(ds.ground_states, state_index=args.state_index)
    else:
        print("[ground_states] not present")

    if "shadow_basis" in ds.list_attributes():
        _print_value_summary("shadow_basis", ds.shadow_basis)
    else:
        print("[shadow_basis] not present")

    if "shadow_meas" in ds.list_attributes():
        _print_value_summary("shadow_meas", ds.shadow_meas)
    else:
        print("[shadow_meas] not present")

    if "shadow_basis" in ds.list_attributes() and "shadow_meas" in ds.list_attributes():
        analyze_shadow_data(ds.shadow_basis, ds.shadow_meas)

    if "spin_system" in ds.list_attributes():
        _print_value_summary("spin_system", ds.spin_system)
        analyze_spin_system(ds.spin_system)
    else:
        print("[spin_system] not present")

    ds.close()


if __name__ == "__main__":
    main()
