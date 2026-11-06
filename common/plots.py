"""Plotting utilities for FastSpeech 2 training curves and variance adaptor analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    from .text import tokenize
except (ImportError, ValueError):
    from common.text import tokenize  # type: ignore

# Configure font fallbacks for Indic scripts (Nirmala UI / Gautami for Telugu on Windows)
plt.rcParams["font.sans-serif"] = [
    "Nirmala UI",
    "Gautami",
    "Segoe UI",
    "Arial Unicode MS",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def _to_numpy(value: Any) -> np.ndarray:
    """Safely convert torch tensor, list, or array to 1D float numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr.astype(np.float32).flatten()


def plot_loss_curves(
    history_source: str | Path | list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Plot and save training_loss.png and validation_loss.png.

    Both plots include both curves for direct comparison while visually
    emphasizing the primary target metric.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(history_source, (str, Path)):
        history_file = Path(history_source)
        if not history_file.is_file():
            raise FileNotFoundError(f"History file not found: {history_file}")
        with history_file.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    else:
        history = list(history_source)

    if not history:
        raise ValueError("Cannot plot loss curves from empty history.")

    epochs = [int(item["epoch"]) for item in history]
    train_losses = [float(item["train_loss"]) for item in history]
    val_losses = [float(item["validation_loss"]) for item in history]

    train_plot_path = out_path / "training_loss.png"
    val_plot_path = out_path / "validation_loss.png"

    # --- 1. Training Loss Plot (with Validation Reference) ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        epochs,
        train_losses,
        label="Training Loss",
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=4,
    )
    ax.plot(
        epochs,
        val_losses,
        label="Validation Loss (reference)",
        color="#ff7f0e",
        linewidth=1.6,
        linestyle="--",
        marker="s",
        markersize=3.5,
        alpha=0.75,
    )
    ax.set_title("Training Loss vs. Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax.set_ylabel("Loss (MSE)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(train_plot_path, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Validation Loss Plot (with Training Reference) ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        epochs,
        val_losses,
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2.4,
        marker="s",
        markersize=4,
    )
    ax.plot(
        epochs,
        train_losses,
        label="Training Loss (reference)",
        color="#1f77b4",
        linewidth=1.6,
        linestyle="--",
        marker="o",
        markersize=3.5,
        alpha=0.75,
    )
    ax.set_title("Validation Loss vs. Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax.set_ylabel("Loss (MSE)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(val_plot_path, bbox_inches="tight")
    plt.close(fig)

    return train_plot_path, val_plot_path


def plot_variance_adaptor(
    sample_id: str,
    text: str,
    output: dict[str, Any],
    config: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate and save duration, pitch, and energy plots for a test sentence.

    Plots are written to:
      {output_dir}/variance_adaptor_plots/duration/{sample_id}_duration.png
      {output_dir}/variance_adaptor_plots/pitch/{sample_id}_pitch.png
      {output_dir}/variance_adaptor_plots/energy/{sample_id}_energy.png
    """
    base_dir = Path(output_dir)
    dur_dir = base_dir / "variance_adaptor_plots" / "duration"
    pitch_dir = base_dir / "variance_adaptor_plots" / "pitch"
    energy_dir = base_dir / "variance_adaptor_plots" / "energy"

    dur_dir.mkdir(parents=True, exist_ok=True)
    pitch_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    # Extract sequences
    durations = _to_numpy(output.get("duration", []))
    pitch = _to_numpy(output.get("pitch", []))
    energy = _to_numpy(output.get("energy", []))

    # Token alignment for x-axis
    try:
        raw_tokens = tokenize(text)
        token_labels = ["<bos>"] + raw_tokens + ["<eos>"]
    except Exception:
        token_labels = ["<bos>"] + list(text) + ["<eos>"]

    seq_len = len(durations) if len(durations) > 0 else len(token_labels)
    if len(token_labels) != seq_len:
        token_labels = [f"t{i}" for i in range(seq_len)]

    indices = np.arange(seq_len)

    # Adaptive figure width based on sequence length
    fig_width = max(10.0, seq_len * 0.22)
    rotation = 75 if seq_len > 35 else (60 if seq_len > 15 else 0)
    font_size = 7 if seq_len > 50 else (8 if seq_len > 25 else 9)

    results: dict[str, Path] = {}

    # --- 1. Duration Plot (bar chart) ---
    if len(durations) > 0:
        dur_path = dur_dir / f"{sample_id}_duration.png"
        fig, ax = plt.subplots(figsize=(fig_width, 4.5), dpi=150)
        bars = ax.bar(indices, durations, color="#2b5c8f", edgecolor="#1a365d", width=0.7)
        ax.set_title(
            f"Predicted Duration per Token - [{sample_id}]",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Token", fontsize=10, labelpad=6)
        ax.set_ylabel("Duration (mel frames)", fontsize=10, labelpad=6)
        ax.set_xticks(indices)
        ax.set_xticklabels(token_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=font_size)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(dur_path, bbox_inches="tight")
        plt.close(fig)
        results["duration"] = dur_path

    # --- 2. Pitch Plot (line plot with markers) ---
    if len(pitch) > 0:
        pitch_path = pitch_dir / f"{sample_id}_pitch.png"
        fig, ax = plt.subplots(figsize=(fig_width, 4.5), dpi=150)
        ax.plot(
            indices,
            pitch,
            color="#2e7d32",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Predicted Pitch (F0)",
        )
        ax.set_title(
            f"Predicted Pitch (F0) per Token - [{sample_id}]",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Token", fontsize=10, labelpad=6)
        ax.set_ylabel("Pitch (Hz / norm)", fontsize=10, labelpad=6)
        ax.set_xticks(indices)
        ax.set_xticklabels(token_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=font_size)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(pitch_path, bbox_inches="tight")
        plt.close(fig)
        results["pitch"] = pitch_path

    # --- 3. Energy Plot (line plot with markers) ---
    if len(energy) > 0:
        energy_path = energy_dir / f"{sample_id}_energy.png"
        fig, ax = plt.subplots(figsize=(fig_width, 4.5), dpi=150)
        ax.plot(
            indices,
            energy,
            color="#c62828",
            linewidth=2.0,
            marker="s",
            markersize=3.5,
            label="Predicted Energy",
        )
        ax.set_title(
            f"Predicted Energy per Token - [{sample_id}]",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Token", fontsize=10, labelpad=6)
        ax.set_ylabel("Energy (log RMS)", fontsize=10, labelpad=6)
        ax.set_xticks(indices)
        ax.set_xticklabels(token_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=font_size)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(energy_path, bbox_inches="tight")
        plt.close(fig)
        results["energy"] = energy_path

    return results


def plot_all_from_history(results_dir: str | Path) -> tuple[Path, Path] | None:
    """Convenience helper to plot loss curves from history.json in results_dir."""
    target_dir = Path(results_dir)
    history_file = target_dir / "history.json"
    if not history_file.is_file():
        return None
    return plot_loss_curves(history_file, target_dir)
