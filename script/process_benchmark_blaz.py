#!/usr/bin/env python3
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_state(path):
    real = []
    imag = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            real.append(float(parts[1]))
            imag.append(float(parts[2]))
    return real, imag


def prob_distribution(real, imag):
    return [(r * r + i * i) for r, i in zip(real, imag)]


def plot_probability(path, probs):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(probs)), probs, linewidth=0.8)
    ax.set_xlabel("index")
    ax.set_ylabel("probability")
    ax.set_title(os.path.basename(path))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = path + ".png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_dir = os.path.join("results", "benchmark_blaz")
    if not os.path.isdir(base_dir):
        raise SystemExit(f"missing directory: {base_dir}")

    for entry in sorted(os.listdir(base_dir)):
        fam_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(fam_dir):
            continue
        original_path = os.path.join(fam_dir, "original.txt")
        if not os.path.isfile(original_path):
            raise SystemExit(f"missing file: {original_path}")

        real, imag = read_state(original_path)
        probs = prob_distribution(real, imag)
        plot_probability(original_path, probs)
        print(f"{entry}: plotted original.txt")


if __name__ == "__main__":
    main()
