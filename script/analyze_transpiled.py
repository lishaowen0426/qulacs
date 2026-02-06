#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path
import pickle
import signal
from typing import Any

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGDependency
from qiskit.version import VERSION as QISKIT_VERSION


# Set this path before running.
ALLOWED_GATES = {"rz", "sx", "cx", "x"}
IGNORED_OPS = {"barrier", "measure"}
OUTPUT_FOLDER = Path("results/rz_tile")
DEFAULT_DAG_CACHE_FOLDER = Path("results/rz_tile/.dag_cache")
QFT_QASM_PATH = Path("compressor/QASMBench/medium/qft_n18/qft_n18_transpiled.qasm")
QFT_OUTPUT_FOLDER = Path("results/rz_tile/medium/qft_n18")
INPUT_DATASET_FOLDERS = [
    Path("compressor/QASMBench/medium"),
    # Path("compressor/QASMBench/large"),
]
DAG_BUILD_TIMEOUT_SEC = 120


def qasm_to_dag_dependency(qasm_file: str | Path) -> DAGDependency:
    """Load an OpenQASM file and return its DAGDependency graph."""
    qasm_path = Path(qasm_file)
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    return circuit_to_dagdependency(circuit)


def _dag_cache_path(qasm_file: str | Path, cache_folder: Path) -> Path:
    qasm_path = Path(qasm_file)
    stat = qasm_path.stat()
    key_src = (
        f"{qasm_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{QISKIT_VERSION}"
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return cache_folder / f"{key}.pkl"


def qasm_to_dag_dependency_cached(
    qasm_file: str | Path, use_cache: bool, cache_folder: Path
) -> DAGDependency:
    if not use_cache:
        return qasm_to_dag_dependency(qasm_file)

    cache_path = _dag_cache_path(qasm_file, cache_folder)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                dag = pickle.load(f)
            if isinstance(dag, DAGDependency):
                print(f"[cache-hit] {qasm_file}")
                return dag
        except Exception:
            pass

    dag = qasm_to_dag_dependency(qasm_file)
    try:
        cache_folder.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(dag, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return dag


class _DagBuildTimeout(Exception):
    pass


def _handle_alarm(_signum: int, _frame: Any) -> None:
    raise _DagBuildTimeout(
        f"Timed out while building DAGDependency (>{DAG_BUILD_TIMEOUT_SEC}s)"
    )


def _node_name(node: Any) -> str:
    if hasattr(node, "name"):
        return str(node.name).lower()
    if hasattr(node, "op") and hasattr(node.op, "name"):
        return str(node.op.name).lower()
    raise TypeError(f"Cannot determine gate name for node: {node}")


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


def _node_by_id_map(dag: DAGDependency) -> dict[int, Any]:
    return {_node_id(node): node for node in dag.get_nodes()}


def validate_allowed_gates(
    dag: DAGDependency, allowed_gates: set[str] | None = None
) -> None:
    allowed = allowed_gates or ALLOWED_GATES
    for node in dag.get_nodes():
        gate = _node_name(node)
        if gate in IGNORED_OPS:
            continue
        if gate not in allowed:
            raise ValueError(
                f"Unsupported gate '{gate}' at node_id={_node_id(node)}. "
                f"Allowed gates: {sorted(allowed)}"
            )


def tile_dag_max_rz(dag: DAGDependency) -> list[dict[str, Any]]:
    """
    Dependency-respecting tiling that prioritizes grouping as many RZ gates as possible.

    Returns a list of tiles. Each tile is a dict:
    - kind: "rz" or "non_rz"
    - node_ids: list[int]
    - gate_names: list[str]
    """
    validate_allowed_gates(dag, ALLOWED_GATES)

    nodes = sorted(
        [node for node in dag.get_nodes() if _node_name(node) not in IGNORED_OPS],
        key=_node_id,
    )
    node_ids = [_node_id(node) for node in nodes]
    node_id_set = set(node_ids)
    node_names = {_node_id(node): _node_name(node) for node in nodes}

    indegree = {
        nid: len(
            [pred for pred in _direct_predecessors(dag, nid) if pred in node_id_set]
        )
        for nid in node_ids
    }
    ready = {nid for nid in node_ids if indegree[nid] == 0}
    scheduled: set[int] = set()
    tiles: list[dict[str, Any]] = []

    def schedule(
        nid: int, tile_node_ids: list[int], tile_gate_names: list[str]
    ) -> None:
        ready.remove(nid)
        scheduled.add(nid)
        tile_node_ids.append(nid)
        tile_gate_names.append(node_names[nid])
        for succ in _direct_successors(dag, nid):
            if succ not in indegree:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0 and succ not in scheduled:
                ready.add(succ)

    while len(scheduled) < len(node_ids):
        if not ready:
            raise RuntimeError(
                "No ready nodes left before schedule completion (cycle or DAG issue)."
            )

        rz_ready = sorted(nid for nid in ready if node_names[nid] == "rz")

        if rz_ready:
            tile_node_ids: list[int] = []
            tile_gate_names: list[str] = []
            while True:
                rz_ready = sorted(
                    nid for nid in ready if node_names[nid] == "rz"
                )  # tie-breaking
                if not rz_ready:
                    break
                schedule(rz_ready[0], tile_node_ids, tile_gate_names)
            tiles.append(
                {"kind": "rz", "node_ids": tile_node_ids, "gate_names": tile_gate_names}
            )
            continue

        tile_node_ids = []
        tile_gate_names = []
        while ready:
            rz_ready = [nid for nid in ready if node_names[nid] == "rz"]
            if rz_ready:
                break
            next_nid = min(
                ready
            )  # when no rz is ready, takes min node id from the ready set.
            schedule(next_nid, tile_node_ids, tile_gate_names)
        tiles.append(
            {"kind": "non_rz", "node_ids": tile_node_ids, "gate_names": tile_gate_names}
        )

    return tiles


def dag_num_qubits(dag: DAGDependency) -> int:
    """Version-safe qubit count."""
    if hasattr(dag, "num_qubits"):
        num_qubits_attr = getattr(dag, "num_qubits")
        return int(num_qubits_attr() if callable(num_qubits_attr) else num_qubits_attr)
    return len(dag.qubits)


def dag_num_nodes(dag: DAGDependency) -> int:
    """Version-safe operation node count."""
    if hasattr(dag, "size"):
        size_attr = getattr(dag, "size")
        return int(size_attr() if callable(size_attr) else size_attr)
    if hasattr(dag, "get_nodes"):
        return len(list(dag.get_nodes()))
    return len(list(dag.op_nodes()))


def plot_tile_size_distribution(
    tiles: list[dict[str, Any]], qasm_path: str | Path, output_folder: Path, kind: str
) -> Path:
    if kind not in {"rz", "non_rz"}:
        raise ValueError("kind must be 'rz' or 'non_rz'")

    sizes = [len(tile["node_ids"]) for tile in tiles if tile["kind"] == kind]
    output_folder.mkdir(parents=True, exist_ok=True)

    label = "RZ" if kind == "rz" else "Non-RZ"
    suffix = "rz" if kind == "rz" else "non_rz"
    circuit_name = Path(qasm_path).stem
    output_path = output_folder / f"{circuit_name}_{suffix}_tile_size_distribution.png"
    if not sizes:
        plt.figure(figsize=(8, 4))
        plt.title(f"{circuit_name}: {label} Tile Size Distribution (no {label} tiles)")
        plt.xlabel(f"{label} tile size")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    max_size = max(sizes)
    size_values = list(range(1, max_size + 1))
    counts = [sizes.count(size) for size in size_values]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(size_values, counts, width=0.8)
    plt.title(f"{circuit_name}: {label} Tile Size Distribution")
    plt.xlabel(f"{label} tile size")
    plt.ylabel("Count")
    plt.xticks(size_values)
    for bar, count in zip(bars, counts):
        if count == 0:
            continue
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def build_tiled_circuit(
    dag: DAGDependency, tiles: list[dict[str, Any]]
) -> QuantumCircuit:
    node_map = _node_by_id_map(dag)
    qubits = list(dag.qubits)
    clbits = list(dag.clbits)
    qubit_to_index = {qubit: i for i, qubit in enumerate(qubits)}
    clbit_to_index = {clbit: i for i, clbit in enumerate(clbits)}

    tiled_circuit = QuantumCircuit(len(qubits), len(clbits), name="tiled")
    for tile in tiles:
        for nid in tile["node_ids"]:
            node = node_map[nid]
            qargs = [tiled_circuit.qubits[qubit_to_index[q]] for q in node.qargs]
            cargs = [tiled_circuit.clbits[clbit_to_index[c]] for c in node.cargs]
            tiled_circuit.append(node.op, qargs, cargs)
        if tile["node_ids"]:
            tiled_circuit.barrier()
    return tiled_circuit


def save_tiled_circuit_plot(
    dag: DAGDependency,
    tiles: list[dict[str, Any]],
    qasm_path: Path,
    output_folder: Path,
) -> Path:
    tiled_circuit = build_tiled_circuit(dag, tiles)
    output_path = output_folder / f"{qasm_path.stem}_tiled_circuit.png"
    figure = tiled_circuit.draw(output="mpl", fold=-1)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def optimize_cx_rz_cx_to_rzz(circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
    optimized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name="optimized")
    qubit_to_index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    clbit_to_index = {clbit: i for i, clbit in enumerate(circuit.clbits)}
    data = circuit.data
    i = 0
    replacements = 0
    while i < len(data):
        inst0 = data[i].operation
        qargs0 = [optimized.qubits[qubit_to_index[q]] for q in data[i].qubits]
        cargs0 = [optimized.clbits[clbit_to_index[c]] for c in data[i].clbits]
        if (
            inst0.name == "cx"
            and i + 2 < len(data)
            and data[i + 1].operation.name == "rz"
            and data[i + 2].operation.name == "cx"
        ):
            inst1 = data[i + 1].operation
            qargs1 = [optimized.qubits[qubit_to_index[q]] for q in data[i + 1].qubits]
            cargs1 = [optimized.clbits[clbit_to_index[c]] for c in data[i + 1].clbits]
            inst2 = data[i + 2].operation
            qargs2 = [optimized.qubits[qubit_to_index[q]] for q in data[i + 2].qubits]
            cargs2 = [optimized.clbits[clbit_to_index[c]] for c in data[i + 2].clbits]

            if (
                qargs0 == qargs2
                and qargs1
                and qargs1[0] == qargs0[1]
                and cargs0 == cargs2
                and len(inst1.params) == 1
            ):
                theta = inst1.params[0]
                optimized.append(RZZGate(theta), [qargs0[0], qargs0[1]], [])
                replacements += 1
                i += 3
                continue

        optimized.append(inst0, qargs0, cargs0)
        i += 1

    return optimized, replacements


def optimize_qft_and_plot() -> None:
    print(f"[start] {QFT_QASM_PATH}")
    old_handler = signal.signal(signal.SIGALRM, _handle_alarm)
    signal.alarm(DAG_BUILD_TIMEOUT_SEC)
    try:
        dag = qasm_to_dag_dependency_cached(
            QFT_QASM_PATH, use_cache=True, cache_folder=DEFAULT_DAG_CACHE_FOLDER
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    tiles = tile_dag_max_rz(dag)
    tiled_circuit = build_tiled_circuit(dag, tiles)
    # Drop tile barriers to allow pattern matching within/between tiles.
    no_barrier = QuantumCircuit(tiled_circuit.num_qubits, tiled_circuit.num_clbits)
    for item in tiled_circuit.data:
        inst = item.operation
        if inst.name == "barrier":
            continue
        qargs = item.qubits
        cargs = item.clbits
        no_barrier.append(inst, qargs, cargs)
    optimized, replacements = optimize_cx_rz_cx_to_rzz(no_barrier)

    QFT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = QFT_OUTPUT_FOLDER / f"{QFT_QASM_PATH.stem}_optimized.png"
    figure = optimized.draw(output="mpl", fold=-1)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Applied CX-RZ-CX -> RZZ replacements: {replacements}")
    print(f"Saved optimized QFT circuit plot to: {output_path}")


def find_transpiled_qasm_in_folder(circuit_folder: Path) -> Path | None:
    # Prefer standard .qasm extension; also accept .qsm if present.
    candidates = sorted(circuit_folder.glob("*_transpiled.qasm"))
    if candidates:
        return candidates[0]
    legacy_candidates = sorted(circuit_folder.glob("*_transpiled.qsm"))
    if legacy_candidates:
        return legacy_candidates[0]
    return None


def analyze_one_qasm_file(
    qasm_path: Path,
    output_folder: Path,
    draw_circuit: bool,
    use_cache: bool,
    cache_folder: Path,
) -> None:
    print(f"[start] {qasm_path}")
    old_handler = signal.signal(signal.SIGALRM, _handle_alarm)
    signal.alarm(DAG_BUILD_TIMEOUT_SEC)
    try:
        dag = qasm_to_dag_dependency_cached(qasm_path, use_cache, cache_folder)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    tiles = tile_dag_max_rz(dag)
    rz_tiles = sum(1 for tile in tiles if tile["kind"] == "rz")
    non_rz_tiles = sum(1 for tile in tiles if tile["kind"] == "non_rz")
    rz_ops = sum(len(tile["node_ids"]) for tile in tiles if tile["kind"] == "rz")

    print(f"Loaded DAGDependency from: {qasm_path}")
    print(f"num_qubits={dag_num_qubits(dag)}, num_nodes={dag_num_nodes(dag)}")
    print(
        f"num_tiles={len(tiles)}, rz_tiles={rz_tiles}, non_rz_tiles={non_rz_tiles}, rz_ops={rz_ops}"
    )

    if draw_circuit:
        tiled_circuit_path = save_tiled_circuit_plot(
            dag, tiles, qasm_path, output_folder
        )
        print(f"Saved tiled circuit plot to: {tiled_circuit_path}")
        return

    rz_plot_path = plot_tile_size_distribution(tiles, qasm_path, output_folder, "rz")
    non_rz_plot_path = plot_tile_size_distribution(
        tiles, qasm_path, output_folder, "non_rz"
    )
    print(f"Saved RZ tile size distribution plot to: {rz_plot_path}")
    print(f"Saved non-RZ tile size distribution plot to: {non_rz_plot_path}")


def analyze_all_datasets(
    draw_circuit: bool, use_cache: bool, cache_folder: Path
) -> None:
    for dataset_root in INPUT_DATASET_FOLDERS:
        if not dataset_root.exists():
            print(f"[skip] dataset folder not found: {dataset_root}")
            continue

        for circuit_folder in sorted(
            path for path in dataset_root.iterdir() if path.is_dir()
        ):
            qasm_path = find_transpiled_qasm_in_folder(circuit_folder)
            if qasm_path is None:
                print(f"[skip] no *_transpiled.qasm in: {circuit_folder}")
                continue

            relative_folder = circuit_folder.relative_to(Path("compressor/QASMBench"))
            output_folder = OUTPUT_FOLDER / relative_folder
            try:
                analyze_one_qasm_file(
                    qasm_path, output_folder, draw_circuit, use_cache, cache_folder
                )
            except _DagBuildTimeout as exc:
                print(f"[skip] {qasm_path}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {qasm_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout",
        type=int,
        default=DAG_BUILD_TIMEOUT_SEC,
        help="Per-circuit timeout in seconds when building DAGDependency.",
    )
    parser.add_argument(
        "--draw-circuit",
        action="store_true",
        help="Also draw and save the tiled circuit as a PNG.",
    )
    parser.add_argument(
        "--no-dag-cache",
        action="store_true",
        help="Disable DAGDependency disk cache.",
    )
    parser.add_argument(
        "--dag-cache-folder",
        type=Path,
        default=DEFAULT_DAG_CACHE_FOLDER,
        help="Folder for serialized DAGDependency cache files.",
    )
    parser.add_argument(
        "--qft-opt",
        action="store_true",
        help="Run QFT-specific CX-RZ-CX to RZZ optimization and save the circuit plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DAG_BUILD_TIMEOUT_SEC = args.timeout
    if args.qft_opt:
        optimize_qft_and_plot()
    else:
        analyze_all_datasets(
            draw_circuit=args.draw_circuit,
            use_cache=not args.no_dag_cache,
            cache_folder=args.dag_cache_folder,
        )
