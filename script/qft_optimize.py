#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
import pickle
import signal
from typing import Any, Optional

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.converters import circuit_to_dag, circuit_to_dagdependency
from qiskit.dagcircuit import DAGCircuit, DAGDependency, DAGOpNode
from qiskit.qasm2 import dump as qasm2_dump
from qiskit.version import VERSION as QISKIT_VERSION


# QFT_QASM_PATH = Path("compressor/QASMBench/medium/qft_n18/qft_n18_transpiled.qasm")
QFT_QASM_PATH = Path("compressor/QASMBench/large/qft_n29/qft_n29_transpiled.qasm")
OUTPUT_FOLDER = Path("results/qft_opt")
DEFAULT_DAG_CACHE_FOLDER = Path("results/qft_opt/.dag_cache")
ALLOWED_GATES = {"rz", "sx", "cx", "x"}
IGNORED_OPS = {"barrier", "measure"}
DAG_BUILD_TIMEOUT_SEC = 120


class BitSet:
    def __init__(self, size: int, value: bool = False) -> None:
        self._bits = [value] * size

    def set_all(self) -> None:
        for i in range(len(self._bits)):
            self._bits[i] = True

    def none(self) -> bool:
        return not any(self._bits)

    def clear_bits(self, mask: "BitSet") -> None:
        for i, bit in enumerate(mask._bits):
            if bit:
                self._bits[i] = False

    def set(self, idx: int) -> None:
        self._bits[idx] = True

    def __len__(self) -> int:
        return len(self._bits)

    def __getitem__(self, idx: int) -> bool:
        return self._bits[idx]


@dataclass
class Group:
    depth: int
    diag_ops: list[DAGOpNode]
    terminator: Optional[DAGOpNode]
    S_out: set[int]


class _DagBuildTimeout(Exception):
    pass


def _handle_alarm(_signum: int, _frame: Any) -> None:
    raise _DagBuildTimeout(
        f"Timed out while building DAGDependency (>{DAG_BUILD_TIMEOUT_SEC}s)"
    )


def _qasm_to_dag_dependency(qasm_file: str | Path) -> DAGDependency:
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


def _qasm_to_dag_dependency_cached(
    qasm_file: str | Path, use_cache: bool, cache_folder: Path
) -> DAGDependency:
    if not use_cache:
        return _qasm_to_dag_dependency(qasm_file)

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

    dag = _qasm_to_dag_dependency(qasm_file)
    try:
        cache_folder.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(dag, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return dag


def _circuit_cache_path(
    circuit: QuantumCircuit, cache_folder: Path, label: str
) -> Path:
    # Use a stable, versioned hash of the circuit's QASM2 representation.
    from qiskit.qasm2 import dumps as qasm2_dumps

    qasm_text = qasm2_dumps(circuit)
    key_src = f"{label}|{QISKIT_VERSION}|{hashlib.sha256(qasm_text.encode('utf-8')).hexdigest()}"
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return cache_folder / f"{key}.pkl"


def _circuit_to_dag_dependency_cached(
    circuit: QuantumCircuit, use_cache: bool, cache_folder: Path, label: str
) -> DAGDependency:
    if not use_cache:
        return circuit_to_dagdependency(circuit)

    cache_path = _circuit_cache_path(circuit, cache_folder, label)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                dag = pickle.load(f)
            if isinstance(dag, DAGDependency):
                print(f"[cache-hit] {label}")
                return dag
        except Exception:
            pass

    dag = circuit_to_dagdependency(circuit)
    try:
        cache_folder.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(dag, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return dag


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


def _validate_allowed_gates(
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


def _tile_dag_max_rz(dag: DAGDependency) -> list[dict[str, Any]]:
    _validate_allowed_gates(dag, ALLOWED_GATES)

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
                rz_ready = sorted(nid for nid in ready if node_names[nid] == "rz")
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
            next_nid = min(ready)
            schedule(next_nid, tile_node_ids, tile_gate_names)
        tiles.append(
            {"kind": "non_rz", "node_ids": tile_node_ids, "gate_names": tile_gate_names}
        )

    return tiles


def _build_tiled_circuit(
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


def _optimize_cx_rz_cx_to_rzz(circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
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


def _fuse_consecutive_rz(circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
    fused = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name="fused_rz")
    qubit_to_index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    clbit_to_index = {clbit: i for i, clbit in enumerate(circuit.clbits)}

    data = circuit.data
    i = 0
    fused_count = 0
    while i < len(data):
        inst = data[i].operation
        qargs = [fused.qubits[qubit_to_index[q]] for q in data[i].qubits]
        cargs = [fused.clbits[clbit_to_index[c]] for c in data[i].clbits]

        if inst.name == "rz" and len(qargs) == 1:
            theta = inst.params[0] if inst.params else 0.0
            j = i + 1
            while j < len(data):
                next_inst = data[j].operation
                next_qargs = data[j].qubits
                if (
                    next_inst.name != "rz"
                    or len(next_qargs) != 1
                    or next_qargs[0] != data[i].qubits[0]
                ):
                    break
                theta = theta + (next_inst.params[0] if next_inst.params else 0.0)
                j += 1
            if j > i + 1:
                fused_count += (j - i) - 1
            fused.rz(theta, qargs[0])
            i = j
            continue

        fused.append(inst, qargs, cargs)
        i += 1

    return fused, fused_count


def compress_group_construction(
    sorted_ops: dict[int, list[DAGOpNode]],
    remaining: dict[int, BitSet],
    depth: int,
    S: set[int],
    K: int,
) -> tuple[Group, BitSet]:
    ops = sorted_ops[depth]
    mask = remaining[depth]
    base_support = set(S)

    scheduled_mask = BitSet(len(ops))
    diag_ops: list[DAGOpNode] = []
    diag_indices: list[int] = []
    targets_by_idx: dict[int, set[int]] = {}

    def gate_name(node: DAGOpNode) -> str:
        return node.op.name

    def qubits(node: DAGOpNode) -> list[int]:
        qs = []
        for q in node.qargs:
            if hasattr(q, "index"):
                qs.append(q.index)
            elif hasattr(q, "_index"):
                qs.append(q._index)
            else:
                raise AttributeError("Qubit has no index attribute")
        return qs

    def targets(node: DAGOpNode) -> set[int]:
        name = gate_name(node)
        qs = qubits(node)
        if name == "rz":
            return {qs[0]}
        if name == "rzz":
            return {qs[0], qs[1]}
        raise ValueError(f"Unexpected gate type in compress_group_construction: {name}")

    for i in range(len(ops)):
        if not mask[i]:
            continue
        node = ops[i]
        name = gate_name(node)
        if name not in {"rz", "rzz"}:
            continue
        T = targets(node)
        diag_indices.append(i)
        targets_by_idx[i] = T

    if len(base_support) > K:
        raise RuntimeError(
            f"Base support exceeds K at depth={depth}: |S|={len(base_support)} K={K}"
        )

    if diag_indices:
        from itertools import combinations

        all_targets: set[int] = set()
        for T in targets_by_idx.values():
            all_targets |= T
        candidate_add = sorted(all_targets - base_support)
        max_add = min(K - len(base_support), len(candidate_add))

        best_support: Optional[set[int]] = None
        best_count = -1
        best_added = None

        for r in range(max_add + 1):
            for combo in combinations(candidate_add, r):
                support = base_support | set(combo)
                count = 0
                for idx in diag_indices:
                    if targets_by_idx[idx].issubset(support):
                        count += 1
                if best_added is None:
                    best_added = r
                if count > best_count or (count == best_count and r < best_added):
                    best_count = count
                    best_support = support
                    best_added = r
            if best_count == len(diag_indices):
                break

        if best_support is not None and best_count > 0:
            for idx in diag_indices:
                if targets_by_idx[idx].issubset(best_support):
                    scheduled_mask.set(idx)
                    diag_ops.append(ops[idx])

    S_diag: set[int] = set()
    for n in diag_ops:
        S_diag |= targets(n)
    group = Group(
        depth=depth,
        diag_ops=diag_ops,
        terminator=None,
        S_out=S_diag,
    )
    return group, scheduled_mask


def choose_next_support(
    sorted_ops: dict[int, list[DAGOpNode]],
    remaining: dict[int, BitSet],
    start_depth: int,
    K: int,
    num_qubits: int,
    prev_support: set[int],
) -> set[int]:
    del num_qubits

    if start_depth not in sorted_ops:
        return set()

    if len(prev_support) > K:
        prev_support = set()

    ops = sorted_ops[start_depth]
    mask = remaining[start_depth]

    def gate_name(node: DAGOpNode) -> str:
        return node.op.name

    def qubits(node: DAGOpNode) -> list[int]:
        qs = []
        for q in node.qargs:
            if hasattr(q, "index"):
                qs.append(q.index)
            elif hasattr(q, "_index"):
                qs.append(q._index)
            else:
                raise AttributeError("Qubit has no index attribute")
        return qs

    def targets(node: DAGOpNode) -> set[int]:
        name = gate_name(node)
        qs = qubits(node)
        if name == "rz":
            return {qs[0]}
        if name == "rzz":
            return {qs[0], qs[1]}
        raise ValueError(f"Unexpected gate type in choose_next_support: {name}")

    def can_add_with_support(support: set[int]) -> bool:
        for i in range(len(ops)):
            if not mask[i]:
                continue
            node = ops[i]
            name = gate_name(node)
            if name not in {"rz", "rzz"}:
                continue
            T = targets(node)
            if len(support | T) <= K:
                return True
        return False

    if prev_support and can_add_with_support(prev_support):
        return set(prev_support)

    if can_add_with_support(set()):
        return set()

    raise RuntimeError(f"No diagonal gate can fit within K={K} at depth={start_depth}.")


class TopologicalSorterRZ_RZZ_SX_CX:
    def sort(self, circuit: QuantumCircuit) -> dict[int, list[DAGOpNode]]:
        dag: DAGCircuit = circuit_to_dag(circuit)
        qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}

        buckets: dict[int, list[DAGOpNode]] = {}
        depth_idx = 0
        for layer in dag.layers():
            ops = list(layer["graph"].op_nodes())
            if not ops:
                continue
            buckets[depth_idx] = ops
            depth_idx += 1

        def gate_class(n: DAGOpNode) -> int:
            name = n.op.name
            if name in {"rz", "rzz"}:
                return 0
            if name in {"sx", "cx"}:
                return 1
            raise ValueError(f"Unexpected gate type in sorter: {name}")

        def target_signature(n: DAGOpNode) -> tuple[int, ...]:
            name = n.op.name
            qs = [qubit_to_index[q] for q in n.qargs]
            if name == "rz":
                return (qs[0],)
            if name == "rzz":
                a, b = qs[0], qs[1]
                return (min(a, b), max(a, b))
            if name == "sx":
                return (qs[0],)
            if name == "cx":
                c, t = qs[0], qs[1]
                return (c, t)
            raise ValueError(f"Unexpected gate type in sorter: {name}")

        def stable_id(n: DAGOpNode) -> int:
            # node_id is assigned in insertion order when building the DAG
            return int(n._node_id)

        for d in buckets:
            buckets[d].sort(
                key=lambda n: (gate_class(n), target_signature(n), stable_id(n))
            )

        return buckets


class QFTOptimizer:
    def __init__(
        self,
        qasm_path: Path,
        use_cache: bool,
        cache_folder: Path,
        output_folder: Path,
        K: int,
    ) -> None:
        self.qasm_path = qasm_path
        self.use_cache = use_cache
        self.cache_folder = cache_folder
        self.output_folder = output_folder
        self.K = K

        self.dag: DAGDependency | None = None
        self.tiles: list[dict[str, Any]] | None = None
        self.tiled_circuit: QuantumCircuit | None = None
        self.optimized_circuit: QuantumCircuit | None = None
        self.fused_circuit: QuantumCircuit | None = None
        self.fused_dag: DAGDependency | None = None
        self.sorted_ops: dict[int, list[DAGOpNode]] | None = None
        self.groups: list[Group] | None = None
        self.replacements: int = 0
        self.fused_count: int = 0

    def preprocess(self) -> QuantumCircuit:
        print(f"[start] {self.qasm_path}")
        old_handler = signal.signal(signal.SIGALRM, _handle_alarm)
        signal.alarm(DAG_BUILD_TIMEOUT_SEC)
        try:
            self.dag = _qasm_to_dag_dependency_cached(
                self.qasm_path, self.use_cache, self.cache_folder
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        self.tiles = _tile_dag_max_rz(self.dag)
        self.tiled_circuit = _build_tiled_circuit(self.dag, self.tiles)

        no_barrier = QuantumCircuit(
            self.tiled_circuit.num_qubits, self.tiled_circuit.num_clbits
        )
        for item in self.tiled_circuit.data:
            inst = item.operation
            if inst.name == "barrier":
                continue
            qargs = item.qubits
            cargs = item.clbits
            no_barrier.append(inst, qargs, cargs)

        self.optimized_circuit, self.replacements = _optimize_cx_rz_cx_to_rzz(
            no_barrier
        )
        print(f"Applied CX-RZ-CX -> RZZ replacements: {self.replacements}")

        self.fused_circuit, self.fused_count = _fuse_consecutive_rz(
            self.optimized_circuit
        )
        print(f"Fused consecutive RZ gates: {self.fused_count}")

        self.fused_dag = _circuit_to_dag_dependency_cached(
            self.fused_circuit,
            use_cache=self.use_cache,
            cache_folder=self.cache_folder,
            label="fused_circuit",
        )
        print("[info] Built DAGDependency for fused circuit")
        self.save_preprocess_outputs()
        return self.fused_circuit

    def save_preprocess_outputs(self) -> None:
        if self.optimized_circuit is None or self.fused_circuit is None:
            raise RuntimeError("Call preprocess() before saving outputs.")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        qasm_stem = self.qasm_path.stem

        output_qasm = self.output_folder / f"{qasm_stem}_optimized.qasm"
        with output_qasm.open("w", encoding="utf-8") as f:
            qasm2_dump(self.optimized_circuit, f)
        print(f"Saved optimized QASM to: {output_qasm}")

        output_png = self.output_folder / f"{qasm_stem}_optimized.png"
        figure = self.optimized_circuit.draw(output="mpl", fold=-1)
        figure.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved optimized circuit plot to: {output_png}")

        fused_qasm = self.output_folder / f"{qasm_stem}_optimized_fused.qasm"
        with fused_qasm.open("w", encoding="utf-8") as f:
            qasm2_dump(self.fused_circuit, f)
        print(f"Saved fused QASM to: {fused_qasm}")

        fused_png = self.output_folder / f"{qasm_stem}_optimized_fused.png"
        fused_fig = self.fused_circuit.draw(output="mpl", fold=-1)
        fused_fig.savefig(fused_png, dpi=150, bbox_inches="tight")
        plt.close(fused_fig)
        print(f"Saved fused circuit plot to: {fused_png}")

    def topological_sort(self) -> None:
        if self.fused_circuit is None:
            raise RuntimeError("Call preprocess() before topological_sort().")
        sorter = TopologicalSorterRZ_RZZ_SX_CX()
        self.sorted_ops = sorter.sort(self.fused_circuit)

    def compression_tile(self) -> None:
        if self.sorted_ops is None:
            raise RuntimeError("Call topological_sort() before compression_tile().")

        remaining: dict[int, BitSet] = {
            depth: BitSet(len(ops)) for depth, ops in self.sorted_ops.items()
        }
        for depth in remaining:
            remaining[depth].set_all()

        groups: list[Group] = []
        support: set[int] = set()

        def has_remaining(depth: int) -> bool:
            return not remaining[depth].none()

        def has_diagonal_remaining(depth: int) -> bool:
            ops = self.sorted_ops[depth]
            mask = remaining[depth]
            for i in range(len(ops)):
                if not mask[i]:
                    continue
                if ops[i].op.name in {"rz", "rzz"}:
                    return True
            return False

        def take_first_nondiag(depth: int) -> int:
            ops = self.sorted_ops[depth]
            mask = remaining[depth]
            for i in range(len(ops)):
                if not mask[i]:
                    continue
                if ops[i].op.name in {"sx", "cx"}:
                    return i
            raise RuntimeError(
                f"No non-diagonal gate found at depth={depth} when expected."
            )

        depths = sorted(self.sorted_ops.keys())
        for depth in depths:
            while has_remaining(depth):
                if has_diagonal_remaining(depth):
                    support = choose_next_support(
                        self.sorted_ops,
                        remaining,
                        depth,
                        self.K,
                        self.fused_circuit.num_qubits,
                        support,
                    )
                    group, scheduled_mask = compress_group_construction(
                        self.sorted_ops,
                        remaining,
                        depth,
                        support,
                        self.K,
                    )
                    if not group.diag_ops and support:
                        support = set()
                        group, scheduled_mask = compress_group_construction(
                            self.sorted_ops,
                            remaining,
                            depth,
                            support,
                            self.K,
                        )
                    if not group.diag_ops:
                        raise RuntimeError(
                            f"Failed to schedule any diagonal gate at depth={depth}."
                        )

                    remaining[depth].clear_bits(scheduled_mask)
                    groups.append(group)
                    support = set(group.S_out)
                    continue

                idx = take_first_nondiag(depth)
                node = self.sorted_ops[depth][idx]
                if node.op.name not in {"sx", "cx"}:
                    raise RuntimeError(
                        f"Unexpected gate type at depth={depth}: {node.op.name}"
                    )
                scheduled_mask = BitSet(len(self.sorted_ops[depth]))
                scheduled_mask.set(idx)
                remaining[depth].clear_bits(scheduled_mask)
                groups.append(
                    Group(depth=depth, diag_ops=[], terminator=node, S_out=set())
                )
                support = set()

        self.groups = groups

    def validate_group(self):
        if self.sorted_ops is None:
            raise RuntimeError("Call topological_sort() before validate_group().")
        if self.groups is None:
            raise RuntimeError("Call compression_tile() before validate_group().")

        node_to_depth: dict[int, int] = {}
        for depth, nodes in self.sorted_ops.items():
            for node in nodes:
                node_to_depth[id(node)] = depth

        scheduled: set[int] = set()
        for group in self.groups:
            for node in group.diag_ops:
                node_id = id(node)
                if node_id not in node_to_depth:
                    raise RuntimeError("Group contains node not in sorted_ops.")
                if node.op.name not in {"rz", "rzz"}:
                    raise RuntimeError("diag_ops contains non-diagonal gate.")
                if node_id in scheduled:
                    raise RuntimeError("Node scheduled more than once.")
                scheduled.add(node_id)

            if group.terminator is not None:
                node_id = id(group.terminator)
                if node_id not in node_to_depth:
                    raise RuntimeError("Group terminator not in sorted_ops.")
                if group.terminator.op.name not in {"sx", "cx"}:
                    raise RuntimeError("Terminator is not sx/cx.")
                if group.diag_ops:
                    raise RuntimeError("sx/cx group should not contain diagonal ops.")
                if node_id in scheduled:
                    raise RuntimeError("Node scheduled more than once.")
                scheduled.add(node_id)
            else:
                if not group.diag_ops:
                    raise RuntimeError("Empty group without terminator.")

        expected = set(node_to_depth.keys())
        missing = expected - scheduled
        extra = scheduled - expected
        if missing:
            raise RuntimeError(f"Missing scheduled nodes: {len(missing)}")
        if extra:
            raise RuntimeError(f"Extra scheduled nodes: {len(extra)}")

    def post_processing(self) -> None:
        if self.groups is None:
            raise RuntimeError("Call compression_tile() before post_processing().")

        merged: list[Group] = []
        for group in self.groups:
            if group.terminator is not None:
                merged.append(group)
                continue

            if merged and merged[-1].terminator is None:
                prev = merged[-1]
                union_support = prev.S_out | group.S_out
                if len(union_support) <= self.K:
                    prev.diag_ops.extend(group.diag_ops)
                    prev.S_out = union_support
                    continue

            merged.append(group)

        self.groups = merged

    def print_groups(self) -> None:
        if self.groups is None:
            raise RuntimeError("Call compression_tile() before print_groups().")

        for group in self.groups:
            depth = group.depth
            diag_count = len(group.diag_ops)
            s_card = len(group.S_out)

            s_out_list = sorted(group.S_out)
            if group.terminator is None:
                print(
                    f"depth={depth} diag_ops={diag_count} "
                    f"S_out={s_card} S_out_set={s_out_list}"
                )
                continue

            term_gate = group.terminator.op.name
            if term_gate == "sx":
                q = group.terminator.qargs[0]
                q_idx = q.index if hasattr(q, "index") else q._index
                print(f"depth={depth} sx target={q_idx}")
                continue
            if term_gate == "cx":
                c_q = group.terminator.qargs[0]
                t_q = group.terminator.qargs[1]
                c_idx = c_q.index if hasattr(c_q, "index") else c_q._index
                t_idx = t_q.index if hasattr(t_q, "index") else t_q._index
                print(f"depth={depth} cx control={c_idx} target={t_idx}")
                continue

            raise RuntimeError(f"Unexpected terminator gate: {term_gate}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qasm",
        type=Path,
        default=QFT_QASM_PATH,
        help="Path to transpiled QFT OpenQASM file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DAG_BUILD_TIMEOUT_SEC,
        help="Per-circuit timeout in seconds when building DAGDependency.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DAG_BUILD_TIMEOUT_SEC = args.timeout
    optimizer = QFTOptimizer(
        qasm_path=args.qasm,
        use_cache=not args.no_dag_cache,
        cache_folder=args.dag_cache_folder,
        output_folder=OUTPUT_FOLDER,
        K=4,
    )
    optimizer.preprocess()
    optimizer.topological_sort()
    optimizer.compression_tile()
    # optimizer.validate_group()
    optimizer.post_processing()
    optimizer.print_groups()
