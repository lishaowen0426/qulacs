#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
import pickle
import signal
from typing import Any, Optional, Literal

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
    kind: Literal["diagonal"]
    S_out: set[int]
    stop_reason: str


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
    S_cur = set(S)

    scheduled_mask = BitSet(len(ops))
    diag_ops: list[DAGOpNode] = []
    terminator: Optional[DAGOpNode] = None

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
        if name == "cx":
            return {qs[0], qs[1]}
        if name == "sx":
            return {qs[0]}
        raise ValueError(f"Unexpected gate type in compress_group_construction: {name}")

    # Step 1: greedily select diagonal gates that fit
    while True:
        best_i: Optional[int] = None
        best_cost = float("inf")

        for i in range(len(ops)):
            if not mask[i] or scheduled_mask[i]:
                continue
            node = ops[i]
            name = gate_name(node)
            if name in {"sx", "cx"}:
                continue

            T = targets(node)
            new_needed = len(T - S_cur)
            if len(S_cur) + new_needed <= K:
                if new_needed < best_cost:
                    best_cost = new_needed
                    best_i = i
                    if best_cost == 0:
                        break

        if best_i is None:
            break

        node = ops[best_i]
        scheduled_mask.set(best_i)
        diag_ops.append(node)
        S_cur |= targets(node)

    # Step 2: select one barrier (sx or cx) as terminator if any remains
    for i in range(len(ops)):
        if not mask[i] or scheduled_mask[i]:
            continue
        node = ops[i]
        name = gate_name(node)
        if name not in {"sx", "cx"}:
            continue

        T = targets(node)
        terminator = node
        scheduled_mask.set(i)
        is_conflict = any(q in S_cur for q in T)
        if name == "sx":
            stop_reason = "hit_sx_conflict" if is_conflict else "hit_sx_nonconflict"
        else:
            stop_reason = "hit_cx_conflict" if is_conflict else "hit_cx_nonconflict"
        group = Group(
            depth=depth,
            diag_ops=diag_ops,
            terminator=terminator,
            kind="diagonal",
            S_out=S_cur,
            stop_reason=stop_reason,
        )
        return group, scheduled_mask

    # Step 3: no sx taken -> diagonal group ends naturally
    if diag_ops:
        group = Group(
            depth=depth,
            diag_ops=diag_ops,
            terminator=None,
            kind="diagonal",
            S_out=S_cur,
            stop_reason="support_saturated",
        )
        return group, scheduled_mask

    # Step 4: safety fallback
    group = Group(
        depth=depth,
        diag_ops=[],
        terminator=None,
        kind="diagonal",
        S_out=S_cur,
        stop_reason="bucket_exhausted",
    )
    return group, scheduled_mask


def choose_next_support(
    sorted_ops: dict[int, list[DAGOpNode]],
    remaining: dict[int, BitSet],
    start_depth: int,
    K: int,
    num_qubits: int,
) -> set[int]:
    def _gate_name(node: DAGOpNode) -> str:
        return node.op.name

    def _qubits(node: DAGOpNode) -> list[int]:
        qs = []
        for q in node.qargs:
            if hasattr(q, "index"):
                qs.append(q.index)
            elif hasattr(q, "_index"):
                qs.append(q._index)
            else:
                raise AttributeError("Qubit has no index attribute")
        return qs

    def _targets(node: DAGOpNode) -> set[int]:
        name = _gate_name(node)
        qs = _qubits(node)
        if name == "rz":
            return {qs[0]}
        if name == "rzz":
            return {qs[0], qs[1]}
        if name == "cx":
            return {qs[0], qs[1]}
        if name == "sx":
            return {qs[0]}
        raise ValueError(f"Unexpected gate type in choose_next_support: {name}")

    S_next: set[int] = set()
    blocked = [False] * num_qubits

    depths = sorted(d for d in sorted_ops.keys() if d >= start_depth)
    for d in depths:
        ops_d = sorted_ops[d]
        mask = remaining.get(d)
        if mask is None or mask.none():
            continue

        for i in range(len(ops_d)):
            if not mask[i]:
                continue

            node = ops_d[i]
            name = _gate_name(node)

            if name in {"sx", "cx"}:
                T = _targets(node)
                if any(q in S_next for q in T):
                    return S_next
                for q in T:
                    blocked[q] = True
                continue

            T = _targets(node)
            if any(blocked[q] for q in T):
                continue

            if len(S_next | T) <= K:
                S_next |= T
                if len(S_next) == K:
                    return S_next

    return S_next


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
        if self.fused_circuit is None:
            raise RuntimeError("Call preprocess() before compression_tile().")

        sorted_ops = self.sorted_ops
        depths = sorted(sorted_ops.keys())
        if not depths:
            self.groups = []
            return

        remaining: dict[int, BitSet] = {}
        for d in depths:
            n = len(sorted_ops[d])
            mask = BitSet(n)
            mask.set_all()
            remaining[d] = mask

        groups: list[Group] = []
        S: set[int] = set()
        num_qubits = self.fused_circuit.num_qubits
        open_start_depth: int | None = None
        open_diag_ops: list[DAGOpNode] = []
        open_terminator: Optional[DAGOpNode] = None
        open_stop_reason: str = "end"
        open_S_out: set[int] = set()

        def diag_targets(nodes: list[DAGOpNode]) -> set[int]:
            out: set[int] = set()
            for n in nodes:
                qs = [q.index if hasattr(q, "index") else q._index for q in n.qargs]
                if n.op.name == "rz":
                    out.add(qs[0])
                elif n.op.name == "rzz":
                    out.update(qs)
                else:
                    raise RuntimeError("diag_targets called on non-diagonal op")
            return out

        def flush_open_group() -> None:
            nonlocal open_start_depth, open_diag_ops, open_terminator, open_stop_reason, open_S_out
            if open_start_depth is None:
                return
            if len(open_S_out) > self.K:
                raise RuntimeError(
                    f"Internal error: open_S_out exceeds K: {len(open_S_out)} > {self.K}"
                )
            groups.append(
                Group(
                    depth=open_start_depth,
                    diag_ops=open_diag_ops,
                    terminator=open_terminator,
                    kind="diagonal",
                    S_out=open_S_out,
                    stop_reason=open_stop_reason,
                )
            )
            open_start_depth = None
            open_diag_ops = []
            open_terminator = None
            open_stop_reason = "end"
            open_S_out = set()

        def next_nonempty_depth(start_d: int) -> int | None:
            for d in depths:
                if d < start_d:
                    continue
                if not remaining[d].none():
                    return d
            return None

        depth = next_nonempty_depth(start_d=depths[0])
        while depth is not None:
            while not remaining[depth].none():
                group, scheduled_mask = compress_group_construction(
                    sorted_ops=sorted_ops,
                    remaining=remaining,
                    depth=depth,
                    S=S,
                    K=self.K,
                )
                remaining[depth].clear_bits(scheduled_mask)

                if scheduled_mask.none():
                    S = set()
                    group2, mask2 = compress_group_construction(
                        sorted_ops=sorted_ops,
                        remaining=remaining,
                        depth=depth,
                        S=S,
                        K=self.K,
                    )
                    remaining[depth].clear_bits(mask2)
                    if mask2.none():
                        raise RuntimeError(
                            "No progress at depth bucket; check K or construction logic"
                        )
                    group = group2
                    scheduled_mask = mask2

                if open_start_depth is None:
                    open_start_depth = depth
                    open_diag_ops = []
                    open_terminator = None
                    open_stop_reason = "end"
                    open_S_out = set()

                new_targets = diag_targets(group.diag_ops)
                if (
                    open_start_depth is not None
                    and len(open_S_out | new_targets) > self.K
                ):
                    flush_open_group()
                    open_start_depth = depth
                    open_diag_ops = []
                    open_terminator = None
                    open_stop_reason = "end"
                    open_S_out = set()

                open_diag_ops.extend(group.diag_ops)
                open_S_out |= new_targets

                if group.terminator is not None:
                    open_terminator = group.terminator
                    open_stop_reason = group.stop_reason
                    flush_open_group()
                elif len(group.diag_ops) == 0 and open_diag_ops:
                    # No diagonal progress; close the current open group before resetting support.
                    flush_open_group()

                S = set(group.S_out)
                if group.terminator is not None or len(group.diag_ops) == 0:
                    flush_open_group()
                    S = choose_next_support(
                        sorted_ops=sorted_ops,
                        remaining=remaining,
                        start_depth=depth,
                        K=self.K,
                        num_qubits=num_qubits,
                    )

            depth = next_nonempty_depth(start_d=depth + 1)

        flush_open_group()
        self.groups = groups
        return

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
                term_q = group.terminator.qargs[0]
                if hasattr(term_q, "index"):
                    term_idx = term_q.index
                elif hasattr(term_q, "_index"):
                    term_idx = term_q._index
                else:
                    raise AttributeError("Qubit has no index attribute")
                term_targets = {term_idx}
                if group.terminator.op.name == "cx":
                    other_q = group.terminator.qargs[1]
                    other_idx = (
                        other_q.index if hasattr(other_q, "index") else other_q._index
                    )
                    term_targets = {term_idx, other_idx}
                if group.stop_reason in {"hit_sx_conflict", "hit_cx_conflict"}:
                    if not term_targets.issubset(group.S_out):
                        raise RuntimeError("Conflicting terminator not in S_out.")
                elif group.stop_reason in {"hit_sx_nonconflict", "hit_cx_nonconflict"}:
                    if term_targets.intersection(group.S_out):
                        raise RuntimeError("Non-conflicting terminator is in S_out.")
                else:
                    raise RuntimeError("Terminator present with invalid stop_reason.")
                if node_id in scheduled:
                    raise RuntimeError("Node scheduled more than once.")
                scheduled.add(node_id)

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

        def _qubit_index(q: Any) -> int:
            if hasattr(q, "index"):
                return int(q.index)
            if hasattr(q, "_index"):
                return int(q._index)
            raise AttributeError("Qubit has no index attribute")

        for group in self.groups:
            targets: set[int] = set()
            for node in group.diag_ops:
                name = node.op.name
                qs = [_qubit_index(q) for q in node.qargs]
                if name == "rz":
                    targets.add(qs[0])
                elif name == "rzz":
                    targets.update(qs)
                else:
                    raise RuntimeError("diag_ops contains non-diagonal gate.")

            if not targets.issubset(group.S_out):
                missing = targets - group.S_out
                raise RuntimeError(
                    "diag_ops targets are not a subset of S_out. "
                    f"group_depth={group.depth} "
                    f"missing={sorted(missing)} "
                    f"S_out={sorted(group.S_out)} "
                    f"diag_ops={[node.op.name for node in group.diag_ops]}"
                )
            if len(targets) > self.K:
                raise RuntimeError(
                    f"diag_ops target set exceeds K. group_depth={group.depth} "
                    f"|targets|={len(targets)} K={self.K} targets={sorted(targets)}"
                )
            group.S_out = targets

    def print_groups(self) -> None:
        if self.groups is None:
            raise RuntimeError("Call compression_tile() before print_groups().")

        for group in self.groups:
            depth = group.depth
            diag_count = len(group.diag_ops)
            s_card = len(group.S_out)

            s_out_list = sorted(group.S_out)
            if group.terminator is not None:
                if group.stop_reason in {"hit_sx_conflict", "hit_sx_nonconflict"}:
                    term_gate = "sx"
                    term_type = (
                        "conflict"
                        if group.stop_reason.endswith("conflict")
                        else "nonconflict"
                    )
                elif group.stop_reason in {"hit_cx_conflict", "hit_cx_nonconflict"}:
                    term_gate = "cx"
                    term_type = (
                        "conflict"
                        if group.stop_reason.endswith("conflict")
                        else "nonconflict"
                    )
                else:
                    raise RuntimeError(
                        "Terminator present but stop_reason is not a barrier reason."
                    )
                print(
                    f"depth={depth} diag_ops={diag_count} terminator={term_gate}:{term_type} "
                    f"S_out={s_card} S_out_set={s_out_list}"
                )
            else:
                if group.stop_reason in {
                    "hit_sx_conflict",
                    "hit_sx_nonconflict",
                    "hit_cx_conflict",
                    "hit_cx_nonconflict",
                }:
                    raise RuntimeError(
                        "stop_reason indicates barrier terminator but terminator is None."
                    )
                print(
                    f"depth={depth} diag_ops={diag_count} terminator=None "
                    f"S_out={s_card} S_out_set={s_out_list}"
                )


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
    optimizer.validate_group()
    optimizer.post_processing()
    optimizer.print_groups()
