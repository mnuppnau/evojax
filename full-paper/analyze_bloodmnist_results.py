"""Generate paper figures from bloodmnist-results.zip.

The script uses only the archived TSV traces and evaluation arrays at the
configured checkpoint iterations. It does not alter the experiment archive.
Run it from the repository root with:

    MPLCONFIGDIR=/tmp/mpl python full-paper/analyze_bloodmnist_results.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "bloodmnist-results.zip"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

RUN_ORDER = [
    "B00",
    "A01",
    "A02",
    "A03",
    "A04-hybrid",
    "A04-hybrid-v2",
    "hybrid+CA",
]

COLORS = {
    "B00": "#202020",
    "A01": "#0072B2",
    "A02": "#D55E00",
    "A03": "#CC79A7",
    "A04-hybrid": "#009E73",
    "A04-hybrid-v2": "#56B4E9",
    "hybrid+CA": "#A6761D",
}

BLEND_RUNS = {"A02", "A03", "hybrid+CA"}

CHECKPOINT_ITERATIONS = (30_000, 100_000, 200_000)

PANEL_STEMS = {
    "B00": "B00",
    "A01": "A01",
    "A02": "A02",
    "A03": "A03",
    "A04-hybrid": "A04_hybrid",
    "A04-hybrid-v2": "A04_hybrid_v2",
    "hybrid+CA": "hybrid_CA",
}


def run_label(member_name: str) -> str:
    directory = PurePosixPath(member_name).parent.name
    if "static-weights-ca-blend-off" in directory:
        return "B00"
    if "dynamic-weights-ca-blend-off" in directory:
        return "A01"
    if "static-weights-ca-blend-on" in directory:
        return "A02"
    if "dynamic-weights-ca-blend-on" in directory:
        return "A03"
    if "A04-v2" in directory:
        return "A04-hybrid-v2"
    if "blend-005" in directory:
        return "hybrid+CA"
    return "A04-hybrid"


def read_tsv(archive: zipfile.ZipFile, member_name: str) -> dict[str, np.ndarray]:
    stream = io.TextIOWrapper(archive.open(member_name), encoding="utf-8")
    rows = list(csv.DictReader(stream, delimiter="\t"))
    return {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in rows[0]
    }


def moving_average(x: np.ndarray, y: np.ndarray, width: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if y.size < width:
        return x, y
    kernel = np.ones(width, dtype=np.float64) / width
    return x[width - 1 :], np.convolve(y, kernel, mode="valid")


def style_for(run: str) -> dict[str, object]:
    return {
        "color": COLORS[run],
        "linestyle": "--" if run in BLEND_RUNS else "-",
        "linewidth": 1.35,
        "label": run,
    }


def plot_metric_trajectories(metrics: dict[str, dict[str, np.ndarray]]) -> None:
    panels = [
        ("mi_avg", "Q-objective proxy"),
        ("r_sense_avg", r"Code separation $r_{\mathrm{sense}}$"),
        ("r_intra_avg", r"Within-code score $r_{\mathrm{intra}}$"),
        ("real_fake_loss", "Real-fake loss"),
        ("code_proto_corr_avg", "Prototype correlation"),
        ("stdev_mean", r"PGPE mean $\sigma$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.6), sharex=True)
    for axis, (column, title) in zip(axes.flat, panels):
        for run in RUN_ORDER:
            trace = metrics[run]
            x, y = moving_average(trace["iter"] / 1000.0, trace[column])
            axis.plot(x, y, **style_for(run))
        axis.set_title(title, fontsize=9)
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.tick_params(labelsize=7)
        axis.set_xlim(0, 290)
    for axis in axes[1, :]:
        axis.set_xlabel("Iteration (thousands)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.0)
    fig.savefig(FIGURE_DIR / "ablation_metric_trajectories.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "ablation_metric_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_diagnostics(
    metrics: dict[str, dict[str, np.ndarray]],
    knowledge: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.25), sharex=True)
    for run in RUN_ORDER:
        x, y = moving_average(metrics[run]["iter"] / 1000.0, metrics[run]["stdev_mean"])
        axes[0].plot(x, y, **style_for(run))
        x, y = moving_average(
            knowledge[run]["iter"] / 1000.0,
            knowledge[run]["ca_stdev_direction"],
        )
        axes[1].plot(x, y, **style_for(run))

    axes[0].set_title(r"PGPE mean $\sigma$", fontsize=9)
    axes[1].set_title(r"Mean sign of $\sigma_{\mathrm{CA}}-\sigma$", fontsize=9)
    axes[1].axhline(0.0, color="#777777", linewidth=0.7)
    for axis in axes:
        axis.set_xlim(0, 290)
        axis.set_xlabel("Iteration (thousands)", fontsize=8)
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.0)
    fig.savefig(FIGURE_DIR / "ca_sigma_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "ca_sigma_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint_panels(archive: zipfile.ZipFile) -> None:
    expected = {
        (run, iteration) for run in RUN_ORDER for iteration in CHECKPOINT_ITERATIONS
    }
    generated: set[tuple[str, int]] = set()

    for member in archive.namelist():
        filename = PurePosixPath(member).name
        if not filename.startswith("iteration-") or not filename.endswith(".npy"):
            continue
        try:
            iteration = int(filename[len("iteration-") : -len(".npy")])
        except ValueError:
            continue
        if iteration not in CHECKPOINT_ITERATIONS:
            continue

        run = run_label(member)
        images = np.load(io.BytesIO(archive.read(member)))[..., 0]
        if images.shape != (64, 28, 28):
            raise ValueError(
                f"Unexpected evaluation-array shape for {run} at iteration "
                f"{iteration}: {images.shape}"
            )

        # The checkpoint cycles through codes 0--7 eight times. Keep the first
        # four repetitions consistently, yielding rows=replicates, cols=codes.
        images = images.reshape(8, 8, 28, 28)[:4]
        panel = np.block([[images[row, code] for code in range(8)] for row in range(4)])

        fig, axis = plt.subplots(figsize=(8, 4), dpi=120)
        axis.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        iteration_label = f"{iteration // 1000}k"
        output_name = f"{PANEL_STEMS[run]}_{iteration_label}_4x8.pdf"
        fig.savefig(FIGURE_DIR / output_name, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        generated.add((run, iteration))

    missing = expected - generated
    if missing:
        missing_labels = [f"{run}@{iteration}" for run, iteration in sorted(missing)]
        raise ValueError(f"Missing evaluation arrays: {missing_labels}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    knowledge: dict[str, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.namelist():
            if member.endswith("/metrics.tsv"):
                metrics[run_label(member)] = read_tsv(archive, member)
            elif member.endswith("/ks_weights.tsv"):
                knowledge[run_label(member)] = read_tsv(archive, member)
        missing = set(RUN_ORDER) - metrics.keys()
        if missing:
            raise ValueError(f"Missing metric traces: {sorted(missing)}")
        plot_metric_trajectories(metrics)
        plot_sigma_diagnostics(metrics, knowledge)
        save_checkpoint_panels(archive)


if __name__ == "__main__":
    main()
