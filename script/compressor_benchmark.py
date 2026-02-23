#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


NUMERIC_FIELDS = ["compress_ms", "decompress_ms", "ratio", "tvd", "fidelity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze compressor benchmark results by grouping rows on "
            "compressor_config (compressor + config)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/outputsv/metrics.csv"),
        help="Path to metrics.csv (default: results/outputsv/metrics.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path for grouped summary. "
            "Default: results/benchmark_compressor/metrics_by_compressor_config.csv"
        ),
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ok",
        help=(
            "Only rows with this status contribute to metric aggregates "
            "(default: ok)."
        ),
    )
    parser.add_argument(
        "--family",
        type=str,
        default="haar_random",
        help="Only analyze rows for this family (default: haar_random).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/benchmark_compressor/plots"),
        help="Directory to write per-group subplot figures.",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing metrics file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"compressor", "config", "dim", "family", "status", *NUMERIC_FIELDS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise SystemExit(f"metrics file missing columns: {missing_str}")
        return list(reader)


def build_summary(rows: list[dict[str, str]], status_filter: str) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "compressor": "",
            "config": "",
            "total_rows": 0,
            "matching_status_rows": 0,
            "values": {name: [] for name in NUMERIC_FIELDS},
        }
    )

    for row in rows:
        compressor = row["compressor"]
        config = row["config"]
        key = f"{compressor}:{config}"
        entry = grouped[key]
        entry["compressor"] = compressor
        entry["config"] = config
        entry["total_rows"] = int(entry["total_rows"]) + 1

        if row["status"] != status_filter:
            continue

        entry["matching_status_rows"] = int(entry["matching_status_rows"]) + 1
        values = entry["values"]
        for field in NUMERIC_FIELDS:
            values[field].append(parse_float(row[field]))

    summary = []
    for compressor_config, entry in grouped.items():
        values = entry["values"]

        def avg(name: str) -> float:
            vals = values[name]
            return sum(vals) / len(vals) if vals else 0.0

        def med(name: str) -> float:
            vals = values[name]
            return median(vals) if vals else 0.0

        summary.append(
            {
                "compressor_config": compressor_config,
                "compressor": str(entry["compressor"]),
                "config": str(entry["config"]),
                "total_rows": str(entry["total_rows"]),
                "matching_status_rows": str(entry["matching_status_rows"]),
                "mean_compress_ms": f"{avg('compress_ms'):.6g}",
                "median_compress_ms": f"{med('compress_ms'):.6g}",
                "mean_decompress_ms": f"{avg('decompress_ms'):.6g}",
                "median_decompress_ms": f"{med('decompress_ms'):.6g}",
                "mean_ratio": f"{avg('ratio'):.6g}",
                "median_ratio": f"{med('ratio'):.6g}",
                "mean_tvd": f"{avg('tvd'):.6g}",
                "median_tvd": f"{med('tvd'):.6g}",
                "mean_fidelity": f"{avg('fidelity'):.6g}",
                "median_fidelity": f"{med('fidelity'):.6g}",
            }
        )

    summary.sort(key=lambda x: float(x["mean_ratio"]), reverse=True)
    return summary


def _safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name)


def plot_group_subplots(
    rows: list[dict[str, str]], status_filter: str, plot_dir: Path, family_label: str
) -> list[Path]:
    per_group_dim = defaultdict(
        lambda: defaultdict(lambda: {field: [] for field in NUMERIC_FIELDS})
    )

    for row in rows:
        if row["status"] != status_filter:
            continue
        group = f"{row['compressor']}:{row['config']}"
        try:
            dim = int(float(row["dim"]))
        except (TypeError, ValueError):
            continue
        for field in NUMERIC_FIELDS:
            per_group_dim[group][dim][field].append(parse_float(row[field]))

    plot_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    for group, dim_map in sorted(per_group_dim.items()):
        dims = sorted(dim_map.keys())
        if not dims:
            continue

        def mean_series(field: str) -> list[float]:
            vals = []
            for dim in dims:
                bucket = dim_map[dim][field]
                vals.append(sum(bucket) / len(bucket) if bucket else 0.0)
            return vals

        compress_ms = mean_series("compress_ms")
        decompress_ms = mean_series("decompress_ms")
        tvd = mean_series("tvd")
        fidelity = mean_series("fidelity")
        ratio = mean_series("ratio")

        fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        fig.suptitle(
            f"{group} (status={status_filter}) - {family_label} state vectors",
            fontsize=12,
        )

        axes[0].plot(dims, compress_ms, marker="o", linewidth=1.5, label="compress_ms")
        axes[0].plot(
            dims, decompress_ms, marker="o", linewidth=1.5, label="decompress_ms"
        )
        axes[0].set_ylabel("time (ms)")
        axes[0].set_title("Compression vs Decompression Time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(dims, tvd, marker="o", linewidth=1.5, label="tvd")
        axes[1].plot(dims, fidelity, marker="o", linewidth=1.5, label="fidelity")
        axes[1].set_ylabel("metric value")
        axes[1].set_title("TVD and Fidelity")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(dims, ratio, marker="o", linewidth=1.5, color="tab:green", label="ratio")
        axes[2].set_ylabel("compression ratio")
        axes[2].set_xlabel("dim")
        axes[2].set_title("Compression Ratio")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        ratio_formatter = ScalarFormatter(useOffset=False)
        ratio_formatter.set_scientific(False)
        axes[2].yaxis.set_major_formatter(ratio_formatter)
        max_ratio = max(ratio) if ratio else 0.0
        axes[2].set_ylim(bottom=0.0, top=max_ratio * 1.05 if max_ratio > 0.0 else 1.0)

        for ax in axes:
            ax.ticklabel_format(axis="x", style="plain")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = plot_dir / f"{_safe_name(group)}_subplots.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_largest_dim_overview(
    rows: list[dict[str, str]], status_filter: str, out_path: Path, family_label: str
) -> tuple[Path | None, int | None]:
    filtered = []
    for row in rows:
        if row["status"] != status_filter:
            continue
        try:
            dim = int(float(row["dim"]))
        except (TypeError, ValueError):
            continue
        filtered.append((dim, row))
    if not filtered:
        return None, None

    largest_dim = max(dim for dim, _ in filtered)
    grouped = defaultdict(
        lambda: {"ratio": [], "tvd": []}
    )
    for dim, row in filtered:
        if dim != largest_dim:
            continue
        key = f"{row['compressor']}:{row['config']}"
        grouped[key]["ratio"].append(parse_float(row["ratio"]))
        grouped[key]["tvd"].append(parse_float(row["tvd"]))
    if not grouped:
        return None, largest_dim

    def mean(vals: list[float]) -> float:
        return (sum(vals) / len(vals)) if vals else 0.0

    configs = sorted(grouped.keys(), key=lambda k: mean(grouped[k]["ratio"]), reverse=True)
    ratio = [mean(grouped[k]["ratio"]) for k in configs]
    tvd = [mean(grouped[k]["tvd"]) for k in configs]

    x = list(range(len(configs)))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, ratio, marker="o", linewidth=1.8, label="compression_ratio")
    tvd_plot = [max(v, 1e-12) for v in tvd]
    ax.plot(x, tvd_plot, marker="o", linewidth=1.8, label="tvd")
    for xi, yi in zip(x, ratio):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    for xi, yi in zip(x, tvd_plot):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("compressor_config")
    ax.set_ylabel("value (log scale)")
    qubits = int(round(math.log2(largest_dim))) if largest_dim > 0 else 0
    if largest_dim > 0 and (1 << qubits) == largest_dim:
        dim_label = f"{qubits} qubits"
    else:
        dim_label = f"{largest_dim} dim (~{math.log2(largest_dim):.3f} qubits)"
    ax.set_title(f"{dim_label} - {family_label} state vectors")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, largest_dim


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no rows to write")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("no grouped rows")
        return

    print("Grouped by compressor_config")
    print(
        "compressor_config,total_rows,matching_status_rows,"
        "mean_ratio,mean_tvd,mean_fidelity"
    )
    for row in rows:
        print(
            f"{row['compressor_config']},"
            f"{row['total_rows']},"
            f"{row['matching_status_rows']},"
            f"{row['mean_ratio']},"
            f"{row['mean_tvd']},"
            f"{row['mean_fidelity']}"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    rows = [row for row in rows if row["family"] == args.family]
    if not rows:
        raise SystemExit(f"no rows found for family: {args.family}")

    summary = build_summary(rows, args.status)

    out_path = args.out
    if out_path is None:
        out_path = Path("results/benchmark_compressor/metrics_by_compressor_config.csv")

    forbidden_dir = Path("results/outputsv").resolve()
    try:
        out_parent = out_path.parent.resolve()
    except OSError:
        out_parent = out_path.parent
    if out_parent == forbidden_dir:
        raise SystemExit(
            "refusing to write into results/outputsv; "
            "use results/benchmark_compressor instead"
        )
    try:
        plot_parent = args.plot_dir.resolve()
    except OSError:
        plot_parent = args.plot_dir
    if plot_parent == forbidden_dir:
        raise SystemExit(
            "refusing to write plots into results/outputsv; "
            "use results/benchmark_compressor instead"
        )

    write_summary(out_path, summary)
    plot_paths = plot_group_subplots(rows, args.status, args.plot_dir, args.family)
    largest_dim_plot_path = Path(
        "results/benchmark_compressor/largest_dim_by_compressor_config.png"
    )
    largest_plot_parent = largest_dim_plot_path.parent.resolve()
    if largest_plot_parent == forbidden_dir:
        raise SystemExit(
            "refusing to write largest-dim plot into results/outputsv; "
            "use results/benchmark_compressor instead"
        )
    largest_plot_written, largest_dim = plot_largest_dim_overview(
        rows, args.status, largest_dim_plot_path, args.family
    )
    print_summary(summary)
    print(f"family filter: {args.family}")
    print(f"\nwrote grouped summary: {out_path}")
    print(f"wrote subplot figures: {len(plot_paths)} files in {args.plot_dir}")
    if largest_plot_written is not None:
        q = int(round(math.log2(largest_dim))) if (largest_dim and largest_dim > 0) else 0
        if largest_dim and (1 << q) == largest_dim:
            label = f"{q} qubits"
        else:
            label = f"{largest_dim} dim"
        print(f"wrote largest-size figure ({label}): {largest_plot_written}")
    else:
        print("largest-dim figure: skipped (no matching rows)")


if __name__ == "__main__":
    main()
