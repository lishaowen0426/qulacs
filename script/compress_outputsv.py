#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

STATE_LINE_RE = re.compile(r"^\|([01]+)\>:\s*(.+?)\s*$")


def _read_cached_amplitudes(cache_path: Path) -> list[complex]:
    amplitudes: list[complex] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"invalid cache format in {cache_path} at line {line_no}: expected 'real imag'"
                )
            real_str, imag_str = parts
            amplitudes.append(complex(float(real_str), float(imag_str)))
    return amplitudes


def _write_cached_amplitudes(cache_path: Path, amplitudes: list[complex]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for amp in amplitudes:
            f.write(f"{amp.real} {amp.imag}\n")


def _parse_amplitudes_from_stdout(out_path: Path) -> list[complex]:
    amplitudes: list[complex] = []
    parsed_count = 0

    with out_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            match = STATE_LINE_RE.match(line)
            if not match:
                continue

            bitstring, amplitude_text = match.groups()
            index = int(bitstring, 2)
            parsed_amplitude = complex(amplitude_text.replace(" ", ""))

            if index == len(amplitudes):
                amplitudes.append(parsed_amplitude)
            elif index > len(amplitudes):
                amplitudes.extend(0j for _ in range(index - len(amplitudes)))
                amplitudes.append(parsed_amplitude)
            else:
                amplitudes[index] = parsed_amplitude
            parsed_count += 1

    if parsed_count == 0:
        raise ValueError(f"no statevector amplitudes found in {out_path}")

    return amplitudes


def parse_statevector_amplitudes(
    xx: str,
    base_dir: Path = Path("results/outputsv"),
    cache_dir: Path = Path("results/outputsvcached"),
) -> list[complex]:
    """
    Parse amplitudes from results/outputsv/<xx>/stdout.1.0 into a state-vector list.

    Each parsed line is expected to look like:
      |0101...>: (<real><+|-><imag>j)
    """

    cache_path = cache_dir / xx / "stdout.1.0"
    if cache_path.is_file():
        return _read_cached_amplitudes(cache_path)

    out_path = base_dir / xx / "stdout.1.0"
    if not out_path.is_file():
        raise FileNotFoundError(f"missing output file: {out_path}")

    amplitudes = _parse_amplitudes_from_stdout(out_path)
    _write_cached_amplitudes(cache_path, amplitudes)
    return amplitudes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and print state-vector amplitudes from outputsv stdout files."
    )
    parser.add_argument(
        "--xx",
        type=str,
        default="a12c30",
        help="dataset key in results/outputsv/<xx>/stdout.1.0 (default: a12c30)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/outputsv"),
        help="base directory containing outputsv results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="print only the first N amplitudes (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/outputsvcached"),
        help="cache directory with files in real imag per line format",
    )
    args = parser.parse_args()

    cache_path = args.cache_dir / args.xx / "stdout.1.0"
    cache_hit = cache_path.is_file()
    amplitudes = parse_statevector_amplitudes(args.xx, args.base_dir, args.cache_dir)
    source_path = cache_path if cache_hit else args.base_dir / args.xx / "stdout.1.0"
    print(f"Loaded {len(amplitudes)} amplitudes from {source_path}")
    if not cache_hit:
        print(f"Wrote cache to {cache_path}")

    end = len(amplitudes) if args.limit is None else min(args.limit, len(amplitudes))
    for i in range(end):
        print(f"{i}: {amplitudes[i]}")

    if args.limit is not None and args.limit < len(amplitudes):
        print(f"... truncated: showed {args.limit} of {len(amplitudes)} amplitudes.")


if __name__ == "__main__":
    main()
