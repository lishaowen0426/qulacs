#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, TypeVar

import pyzx as zx
import pyzx.todd as zx_todd
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGDependency
from qiskit.qasm2 import dump as qasm2_dump
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.qasm2 import loads as qasm2_loads

# Clifford+T with CZ as the only 2-qubit primitive to favor diagonal structure.
CLIFFORD_T_BASIS = ["h", "s", "sdg", "t", "tdg", "z", "cz"]
DIAGONAL_GATES = {
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "cz",
    # Extra names in case intermediate circuits contain them.
    "p",
    "cp",
    "rz",
    "rzz",
    "u1",
}
IGNORED_OPS = {"barrier"}
T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPT_BIN = REPO_ROOT / "compressor" / "TOpt" / "bin" / "TOpt"
QASMBENCH_BATCH_ROOTS = [
    Path("compressor/QASMBench/small"),
    Path("compressor/QASMBench/medium"),
]


def _node_id(node: Any) -> int:
    if hasattr(node, "node_id"):
        return int(node.node_id)
    raise TypeError(f"Cannot determine node_id for node: {node}")


def _direct_predecessors(dag: DAGDependency, node_id: int) -> list[int]:
    if hasattr(dag, "direct_predecessors"):
        return [int(nid) for nid in dag.direct_predecessors(node_id)]
    if hasattr(dag, "predecessors"):
        return [int(nid) for nid in dag.predecessors(node_id)]
    raise AttributeError(
        "DAGDependency has neither direct_predecessors nor predecessors"
    )


def _direct_successors(dag: DAGDependency, node_id: int) -> list[int]:
    if hasattr(dag, "direct_successors"):
        return [int(nid) for nid in dag.direct_successors(node_id)]
    if hasattr(dag, "successors"):
        return [int(nid) for nid in dag.successors(node_id)]
    raise AttributeError("DAGDependency has neither direct_successors nor successors")


def is_diagonal_gate(name: str) -> bool:
    return name.lower() in DIAGONAL_GATES


def count_ops_sorted(circuit: QuantumCircuit) -> OrderedDict[str, int]:
    ops = circuit.count_ops()
    return OrderedDict(sorted((str(name), int(count)) for name, count in ops.items()))


def diagonal_metrics(circuit: QuantumCircuit) -> dict[str, float | int]:
    total = 0
    diagonal = 0
    run_count = 0
    max_run = 0
    current_run = 0

    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in IGNORED_OPS or name == "measure":
            continue
        total += 1
        if is_diagonal_gate(name):
            diagonal += 1
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            if current_run > 0:
                run_count += 1
                current_run = 0
    if current_run > 0:
        run_count += 1

    ratio = float(diagonal / total) if total else 0.0
    return {
        "total_gates": total,
        "diagonal_gates": diagonal,
        "diagonal_ratio": ratio,
        "diagonal_runs": run_count,
        "max_diagonal_run": max_run,
    }


def extract_final_measurements(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[tuple[int, int]], int]:
    has_seen_measure = False
    measurements: list[tuple[int, int]] = []

    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name == "measure":
            has_seen_measure = True
            q_index = circuit.find_bit(inst.qubits[0]).index
            c_index = circuit.find_bit(inst.clbits[0]).index
            measurements.append((q_index, c_index))
            continue
        if name == "barrier":
            continue
        if has_seen_measure:
            raise ValueError(
                "Mid-circuit measurement detected. "
                "This script currently supports only final measurements."
            )

    unitary = circuit.remove_final_measurements(inplace=False)
    return unitary, measurements, circuit.num_clbits


def restore_final_measurements(
    circuit: QuantumCircuit,
    measurements: list[tuple[int, int]],
    num_clbits: int,
) -> QuantumCircuit:
    if num_clbits <= 0 and not measurements:
        return circuit

    restored = QuantumCircuit(circuit.num_qubits, num_clbits)
    restored.global_phase = circuit.global_phase
    for inst in circuit.data:
        qargs = [
            restored.qubits[circuit.find_bit(qubit).index] for qubit in inst.qubits
        ]
        cargs = [restored.clbits[circuit.find_bit(cbit).index] for cbit in inst.clbits]
        restored.append(inst.operation.copy(), qargs, cargs)

    for q_index, c_index in measurements:
        restored.measure(q_index, c_index)
    return restored


def reorder_to_prioritize_diagonal(circuit: QuantumCircuit) -> QuantumCircuit:
    if not circuit.data:
        return circuit

    dag = circuit_to_dagdependency(circuit)
    nodes = sorted(list(dag.get_nodes()), key=_node_id)
    if not nodes:
        return circuit

    node_ids = [_node_id(node) for node in nodes]
    node_id_set = set(node_ids)
    node_by_id = {_node_id(node): node for node in nodes}

    indegree = {
        nid: len(
            [pred for pred in _direct_predecessors(dag, nid) if pred in node_id_set]
        )
        for nid in node_ids
    }
    ready = {nid for nid in node_ids if indegree[nid] == 0}
    schedule: list[int] = []
    scheduled: set[int] = set()

    while len(schedule) < len(node_ids):
        if not ready:
            raise RuntimeError(
                "No ready nodes left before schedule completion (cycle or DAG issue)."
            )
        diagonal_ready = sorted(
            nid for nid in ready if is_diagonal_gate(node_by_id[nid].name)
        )
        next_nid = diagonal_ready[0] if diagonal_ready else min(ready)

        ready.remove(next_nid)
        scheduled.add(next_nid)
        schedule.append(next_nid)

        for succ in _direct_successors(dag, next_nid):
            if succ not in indegree:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0 and succ not in scheduled:
                ready.add(succ)

    reordered = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    reordered.global_phase = circuit.global_phase
    for nid in schedule:
        node = node_by_id[nid]
        qargs = [
            reordered.qubits[circuit.find_bit(qubit).index] for qubit in node.qargs
        ]
        cargs = [reordered.clbits[circuit.find_bit(cbit).index] for cbit in node.cargs]
        reordered.append(node.op.copy(), qargs, cargs)

    return reordered


def print_stage_stats(label: str, circuit: QuantumCircuit) -> None:
    metrics = diagonal_metrics(circuit)
    print(f"\n[{label}]")
    print(f"  ops: {count_ops_sorted(circuit)}")
    print(
        "  diagonal: "
        f"{metrics['diagonal_gates']}/{metrics['total_gates']} "
        f"({metrics['diagonal_ratio']:.3f})"
    )
    print(
        "  grouping: "
        f"runs={metrics['diagonal_runs']}, "
        f"max_run={metrics['max_diagonal_run']}"
    )


def run_with_progress(
    label: str,
    fn: Callable[[], T],
    heartbeat_sec: float,
) -> T:
    print(f"[start] {label}", flush=True)
    started = time.perf_counter()
    done = threading.Event()

    def _heartbeat() -> None:
        while not done.wait(timeout=heartbeat_sec):
            elapsed = time.perf_counter() - started
            print(f"[running] {label} ... {elapsed:.1f}s elapsed", flush=True)

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = fn()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"[error] {label} failed after {elapsed:.1f}s", flush=True)
        done.set()
        heartbeat_thread.join(timeout=0.2)
        raise

    done.set()
    heartbeat_thread.join(timeout=0.2)
    elapsed = time.perf_counter() - started
    print(f"[done] {label} in {elapsed:.1f}s", flush=True)
    return result


def patch_pyzx_topt_parser() -> None:
    # PyZX 0.9.0 parses TOpt rows as strings, which can crash in todd_simp on
    # newer TOpt output. This patch converts parsed bits to ints.
    def _call_topt_patched(m: Any, quiet: bool = True) -> Any:
        assert zx.settings.topt_command is not None
        if not quiet:
            print("TOpt: ", end="")

        t_start = m.cols()
        gsm_text = "\n".join(" ".join(str(i) for i in r) for r in m.data)

        with tempfile.NamedTemporaryFile(suffix=".gsm") as f:
            f.write(gsm_text.encode("ascii"))
            f.flush()
            time.sleep(0.01)

            if zx.settings.topt_command[0].find("wsl") != -1:
                fname = "/mnt/c" + f.name.replace("\\", "/")[2:]
            else:
                fname = f.name

            cmd = [*zx.settings.topt_command, "gsm", fname]
            if zx_todd.USE_REED_MULLER:
                cmd.extend(["-a", "rm"])

            output = subprocess.check_output(cmd)
            out = output.decode()

        start = out.find("Output gate")
        end = out.find("Successful")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("Failed to parse TOpt output block.")

        rows = out[start:end].strip().splitlines()[2:]
        i = out.find("Total time")
        if i != -1:
            t = out[i + 10 : out.find("s", i)]
            if not quiet:
                print(t)

        data: list[list[int]] = []
        for row in rows:
            raw = row.strip()
            if not raw:
                continue
            if all(ch in "01" for ch in raw):
                bits = [int(ch) for ch in raw]
            else:
                bits = [int(tok) for tok in raw.split()]
            data.append(bits)

        m2 = zx_todd.Mat2(data)
        if zx_todd.USE_REED_MULLER:
            m_tmp = m2.transpose()
            zx_todd.remove_trivial_cols(m_tmp)
            m2 = m_tmp.transpose()

        t_end = m2.cols()
        if t_end < t_start and not quiet:
            print("Found reduction: ", t_start - t_end)
        return m2

    zx_todd.call_topt = _call_topt_patched


def configure_topt() -> None:
    candidate = DEFAULT_TOPT_BIN
    if not candidate.exists():
        zx.settings.topt_command = None
        print(
            "[config] TOpt not found. "
            f"Expected at: {candidate}. Falling back to built-in TODD.",
            flush=True,
        )
        return
    if not candidate.is_file():
        zx.settings.topt_command = None
        print(
            f"[config] TOpt path is not a file: {candidate}. "
            "Falling back to built-in TODD.",
            flush=True,
        )
        return

    zx.settings.topt_command = [str(candidate.resolve())]
    patch_pyzx_topt_parser()
    print(f"[config] Using TOpt: {zx.settings.topt_command}", flush=True)


def cut_phase_blocks_only(
    circuit: zx.Circuit, quiet: bool
) -> tuple[list[list[Any]], list[list[Any]], list[Any], int, list[tuple[int, int]]]:
    normalized = circuit.to_basic_gates().split_phase_gates()
    correction: list[Any] = []
    qubits = normalized.qubits

    gates = {i: [] for i in range(qubits)}
    for idx, gate in enumerate(normalized.gates):
        gate_copy = gate.copy()
        gate_copy.index = idx
        if gate_copy.name in ("CNOT", "CZ"):
            gates[gate_copy.control].append(gate_copy)
            gates[gate_copy.target].append(gate_copy)
        elif gate_copy.name != "HAD":
            if not isinstance(gate_copy, zx.circuit.gates.ZPhase):
                raise TypeError(
                    f"Unknown gate {gate_copy}. "
                    "Maybe simplify with circuit.to_basic_gates()?"
                )
            gates[gate_copy.target].append(gate_copy)
        else:
            gates[gate_copy.target].append(gate_copy)

    blocks: list[list[Any]] = []
    hadamard_layers: list[list[Any]] = []
    layer_summaries: list[tuple[int, int]] = []
    layer_index = 0
    while any(gates.values()):
        if not quiet:
            print("new block", flush=True)
        block, hadamards = zx.optimize.greedy_consume_gates(gates, qubits)
        hadamard_count = sum(1 for h in hadamards if getattr(h, "name", "") == "HAD")
        layer_summaries.append((len(block), hadamard_count))
        print(
            f"[phase-block-layer {layer_index}] "
            f"block_gates={len(block)} hadamards={hadamard_count}",
            flush=True,
        )
        if block:
            blocks.append([g.copy() for g in block])
        if hadamards:
            hadamard_layers.append([h.copy() for h in hadamards])
        layer_index += 1

    return blocks, hadamard_layers, correction, qubits, layer_summaries


def write_phase_block_report(
    report_path: Path,
    layer_summaries: list[tuple[int, int]],
    blocks_count: int,
    hadamard_layers_count: int,
    correction_count: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for idx, (block_gates, hadamards) in enumerate(layer_summaries):
            f.write(f"layer {idx}: block_gates={block_gates} hadamards={hadamards}\n")
        f.write(
            f"summary: blocks={blocks_count} "
            f"hadamard_layers={hadamard_layers_count} "
            f"correction_gates={correction_count}\n"
        )


def rebuild_from_phase_blocks(
    blocks: list[list[Any]],
    hadamard_layers: list[list[Any]],
    correction: list[Any],
    qubits: int,
) -> zx.Circuit:
    rebuilt = zx.Circuit(qubits)
    layers = max(len(blocks), len(hadamard_layers))
    for idx in range(layers):
        if idx < len(blocks):
            rebuilt.gates.extend(g.copy() for g in blocks[idx])
        if idx < len(hadamard_layers):
            rebuilt.gates.extend(h.copy() for h in hadamard_layers[idx])
    rebuilt.gates.extend(g.copy() for g in correction)
    return rebuilt


def draw_circuit_with_qiskit(circuit: QuantumCircuit, draw_path: Path) -> Path:
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = draw_path.suffix.lower()
    if suffix in {".txt", ".text"}:
        draw_path.write_text(
            str(circuit.draw(output="text", fold=-1)), encoding="utf-8"
        )
        return draw_path

    try:
        circuit.draw(output="mpl", filename=str(draw_path), fold=-1)
        return draw_path
    except Exception as exc:
        fallback = draw_path.with_suffix(".txt")
        fallback.write_text(str(circuit.draw(output="text", fold=-1)), encoding="utf-8")
        print(
            f"[warn] Failed to generate mpl figure ({exc}). "
            f"Wrote text diagram to: {fallback}",
            flush=True,
        )
        return fallback


def generate_iqp_style_circuit(num_qubits: int, seed: int) -> QuantumCircuit:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)

    # IQP-style: H^n -> diagonal (commuting) phase polynomial -> H^n.
    for q in range(num_qubits):
        circuit.h(q)

    # Random 1-qubit diagonal phases.
    for q in range(num_qubits):
        r = rng.random()
        if r < 0.45:
            circuit.t(q)
        elif r < 0.75:
            circuit.s(q)
        elif r < 0.90:
            circuit.z(q)

    # Random 2-qubit diagonal interactions (CZ), also commuting.
    edge_prob = min(0.35, 2.0 / max(num_qubits, 2))
    for q0 in range(num_qubits):
        for q1 in range(q0 + 1, num_qubits):
            if rng.random() < edge_prob:
                circuit.cz(q0, q1)

    for q in range(num_qubits):
        circuit.h(q)
    return circuit


def optimize_qasm_with_pyzx(
    original: QuantumCircuit,
    output_qasm: Path,
    qiskit_opt_level: int,
    skip_diagonal_reorder: bool,
    heartbeat_sec: float,
    phase_blocks_only: bool,
    draw_path: Path,
    phase_report_path: Path | None = None,
) -> None:
    unitary, measurements, num_clbits = extract_final_measurements(original)
    if measurements:
        print(
            f"[prep] removed {len(measurements)} final measurements before phase processing.",
            flush=True,
        )

    # Stage 1: force a Clifford+T(+CZ) representation with Qiskit.
    # Note: for circuits with non-Clifford angles (for example QFT with pi/8 terms),
    # this approximation can cause very large gate-count blowups.
    clifford_t = run_with_progress(
        "Qiskit transpile to Clifford+T(+CZ)",
        lambda: transpile(
            unitary,
            basis_gates=CLIFFORD_T_BASIS,
            optimization_level=qiskit_opt_level,
        ),
        heartbeat_sec=heartbeat_sec,
    )

    # Stage 2: run PyZX optimizers that target phase polynomial structure.
    zx_circuit = run_with_progress(
        "PyZX parse and normalize",
        lambda: zx.Circuit.from_qasm(qasm2_dumps(clifford_t))
        .to_basic_gates()
        .split_phase_gates(),
        heartbeat_sec=heartbeat_sec,
    )
    if phase_blocks_only:
        (
            blocks,
            hadamard_layers,
            correction,
            _qubits,
            layer_summaries,
        ) = run_with_progress(
            "PyZX cut into phase-polynomial blocks",
            lambda: cut_phase_blocks_only(zx_circuit, quiet=True),
            heartbeat_sec=heartbeat_sec,
        )
        print(
            "[phase-blocks] "
            f"blocks={len(blocks)}, "
            f"hadamard_layers={len(hadamard_layers)}, "
            f"correction_gates={len(correction)}",
            flush=True,
        )
        print(
            "[phase-blocks] analysis-only: keeping normalized circuit unchanged.",
            flush=True,
        )
        if phase_report_path is not None:
            write_phase_block_report(
                report_path=phase_report_path,
                layer_summaries=layer_summaries,
                blocks_count=len(blocks),
                hadamard_layers_count=len(hadamard_layers),
                correction_count=len(correction),
            )
            print(f"[phase-blocks] report written to: {phase_report_path}", flush=True)
        zx_optimized = zx_circuit
    else:
        zx_optimized = run_with_progress(
            "PyZX phase_block_optimize",
            lambda: zx.optimize.phase_block_optimize(
                zx_circuit, pre_optimize=False, quiet=True
            ),
            heartbeat_sec=heartbeat_sec,
        )
        zx_optimized = run_with_progress(
            "PyZX full_optimize",
            lambda: zx.optimize.full_optimize(zx_optimized, quiet=True),
            heartbeat_sec=heartbeat_sec,
        )
    zx_optimized = zx_optimized.split_phase_gates()
    after_pyzx = run_with_progress(
        "Convert PyZX output back to Qiskit",
        lambda: qasm2_loads(zx_optimized.to_qasm()),
        heartbeat_sec=heartbeat_sec,
    )

    # Stage 3: optional post-processing.
    if phase_blocks_only:
        final_unitary = after_pyzx
    else:
        final_unitary = run_with_progress(
            "Qiskit normalize back to Clifford+T(+CZ)",
            lambda: transpile(
                after_pyzx,
                basis_gates=CLIFFORD_T_BASIS,
                optimization_level=0,
            ),
            heartbeat_sec=heartbeat_sec,
        )
        if not skip_diagonal_reorder:
            final_unitary = run_with_progress(
                "Diagonal-priority dependency scheduling",
                lambda: reorder_to_prioritize_diagonal(final_unitary),
                heartbeat_sec=heartbeat_sec,
            )

    if phase_blocks_only:
        # Keep pure unitary output for phase-block analysis mode.
        final_circuit = final_unitary
    else:
        final_circuit = restore_final_measurements(
            final_unitary, measurements, num_clbits
        )

    print_stage_stats("input", original)
    print_stage_stats("qiskit_clifford_t", clifford_t)
    print_stage_stats("after_pyzx", after_pyzx)
    print_stage_stats("final", final_circuit)

    output_qasm.parent.mkdir(parents=True, exist_ok=True)
    with output_qasm.open("w", encoding="utf-8") as f:
        qasm2_dump(final_circuit, f)
    print(f"\nWrote optimized QASM to: {output_qasm}")

    written_draw = run_with_progress(
        "Draw final circuit with Qiskit",
        lambda: draw_circuit_with_qiskit(final_circuit, draw_path),
        heartbeat_sec=heartbeat_sec,
    )
    print(f"Wrote circuit drawing to: {written_draw}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transpile a QASM circuit to Clifford+T, optimize with PyZX, "
            "and maximize diagonal gate grouping."
        )
    )
    parser.add_argument(
        "input_qasm",
        nargs="?",
        type=Path,
        default=None,
        help="Input OpenQASM file path. Not required when using --iqp mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all artifacts (QASM and circuit drawing).",
    )
    parser.add_argument(
        "--qiskit-opt-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Qiskit transpiler optimization level for the first Clifford+T stage.",
    )
    parser.add_argument(
        "--skip-diagonal-reorder",
        action="store_true",
        help="Skip dependency-safe diagonal-priority scheduling at the end.",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=10.0,
        help="Seconds between progress heartbeat logs while a stage is running.",
    )
    parser.add_argument(
        "--phase-blocks-only",
        action="store_true",
        help="Cut into phase-polynomial blocks and rebuild without TODD optimization.",
    )
    parser.add_argument(
        "--iqp",
        action="store_true",
        help=(
            "Generate a Qiskit IQP-style circuit instead of reading input_qasm. "
            "In this mode, phase-block cutting is applied automatically."
        ),
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=None,
        help="Number of qubits for --iqp mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for --iqp generation.",
    )
    parser.add_argument(
        "--phase-report-path",
        type=Path,
        default=None,
        help="Optional path to write phase block/layer report (used in phase-block mode).",
    )
    parser.add_argument(
        "--qasmbench-batch",
        action="store_true",
        help=(
            "Process all circuits in compressor/QASMBench/small and /medium, "
            "run transpile + phase-block cutting, and mirror output folder structure."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Per-circuit timeout in seconds for --qasmbench-batch mode.",
    )
    return parser.parse_args()


def find_qasmbench_input(subdir: Path) -> Path | None:
    expected = subdir / f"{subdir.name}_transpiled.qasm"
    if expected.exists():
        return expected
    matches = sorted(subdir.glob("*_transpiled.qasm"))
    if len(matches) == 1:
        return matches[0]
    return None


def run_qasmbench_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    total = 0
    done = 0
    skipped = 0
    timed_out = 0
    failed = 0

    for root in QASMBENCH_BATCH_ROOTS:
        if not root.exists():
            print(f"[batch-skip] missing root: {root}", flush=True)
            continue
        for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
            total += 1
            circuit_name = subdir.name
            input_qasm = find_qasmbench_input(subdir)
            if input_qasm is None:
                print(
                    f"[batch-skip] {subdir}: no unique *_transpiled.qasm found",
                    flush=True,
                )
                skipped += 1
                continue

            out_subdir = output_dir / root.name / circuit_name
            out_subdir.mkdir(parents=True, exist_ok=True)
            report_path = out_subdir / f"{circuit_name}.txt"

            cmd = [
                sys.executable,
                str(script_path),
                str(input_qasm),
                "--output-dir",
                str(out_subdir),
                "--phase-blocks-only",
                "--phase-report-path",
                str(report_path),
                "--qiskit-opt-level",
                str(args.qiskit_opt_level),
                "--heartbeat-sec",
                str(args.heartbeat_sec),
            ]
            if args.skip_diagonal_reorder:
                cmd.append("--skip-diagonal-reorder")

            print(f"[batch] start {root.name}/{circuit_name}", flush=True)
            try:
                result = subprocess.run(
                    cmd,
                    timeout=args.timeout_sec,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                timed_out += 1
                print(
                    f"[batch-timeout] {root.name}/{circuit_name} (> {args.timeout_sec}s), skipped",
                    flush=True,
                )
                continue

            if result.returncode == 0:
                done += 1
                print(f"[batch-ok] {root.name}/{circuit_name}", flush=True)
            else:
                failed += 1
                err = (result.stderr or "").strip().splitlines()
                err_head = f" err={err[0]}" if err else ""
                print(
                    f"[batch-fail] {root.name}/{circuit_name} (rc={result.returncode}){err_head}",
                    flush=True,
                )

    print(
        "[batch-summary] "
        f"total={total} done={done} timed_out={timed_out} failed={failed} skipped={skipped}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.qasmbench_batch:
        run_qasmbench_batch(args)
        return

    configure_topt()

    if args.iqp:
        if args.num_qubits is None:
            raise ValueError("--num-qubits is required when --iqp is set.")
        if args.input_qasm is not None:
            print(
                "[warn] input_qasm is ignored in --iqp mode.",
                flush=True,
            )
        original = run_with_progress(
            f"Generate IQP-style circuit (n={args.num_qubits}, seed={args.seed})",
            lambda: generate_iqp_style_circuit(args.num_qubits, args.seed),
            heartbeat_sec=args.heartbeat_sec,
        )
        output_stem = f"iqp_n{args.num_qubits}_zx_diag_opt"
        phase_blocks_only = True
    else:
        if args.input_qasm is None:
            raise ValueError("input_qasm is required unless --iqp is set.")
        input_qasm = args.input_qasm
        if not input_qasm.exists():
            raise FileNotFoundError(f"Input QASM file not found: {input_qasm}")
        original = run_with_progress(
            "Load input QASM",
            lambda: QuantumCircuit.from_qasm_file(str(input_qasm)),
            heartbeat_sec=args.heartbeat_sec,
        )
        input_stem = (
            input_qasm.stem[:-11]
            if input_qasm.stem.endswith("_transpiled")
            else input_qasm.stem
        )
        output_stem = f"{input_stem}_zx_diag_opt"
        phase_blocks_only = args.phase_blocks_only

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_qasm = output_dir / f"{output_stem}.qasm"
    draw_path = output_dir / f"{output_stem}.png"

    optimize_qasm_with_pyzx(
        original=original,
        output_qasm=output_qasm,
        qiskit_opt_level=args.qiskit_opt_level,
        skip_diagonal_reorder=args.skip_diagonal_reorder,
        heartbeat_sec=args.heartbeat_sec,
        phase_blocks_only=phase_blocks_only,
        draw_path=draw_path,
        phase_report_path=args.phase_report_path,
    )


if __name__ == "__main__":
    main()
