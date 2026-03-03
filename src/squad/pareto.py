from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Save Pareto table for comparison
def save_pareto_table(rows: List[Dict[str, object]], run_directory: Path) -> Path:
    data_frame = pd.DataFrame(rows)
    output_path = run_directory / "pareto.csv"
    data_frame.to_csv(output_path, index = False)
    return output_path

# Compute axes bounds automatically with padding
def _auto_limits(values: np.ndarray, pad_frac: float = 0.08, min_pad: float = 1e-6) -> Tuple[float, float]:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(vmax - vmin, min_pad)
    pad = span * pad_frac
    return vmin - pad, vmax + pad

# Set up consistent plot styling for all Pareto frontier visualizations
def _setup_plot():
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)

# Plot energy-EM Pareto frontier for max sequence length
def plot_energy_em_max_sequence_length(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["exact_match"].to_numpy()
    labels = data_frame["max_sequence_length"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Exact Match")
    plt.title("Energy-Exact Match Pareto Frontier (Max Sequence Length in Tokens)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_em_max_sequence_length.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-F1 Pareto frontier for max sequence length
def plot_energy_f1_max_sequence_length(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["f1"].to_numpy()
    labels = data_frame["max_sequence_length"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("F1")
    plt.title("Energy-F1 Pareto Frontier (Max Sequence Length in Tokens)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_f1_max_sequence_length.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for max sequence length
def plot_energy_latency_max_sequence_length(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["max_sequence_length"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (2, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Max Sequence Length in Tokens)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_latency_max_sequence_length.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-EM Pareto frontier for layers
def plot_energy_em_layers(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["exact_match"].to_numpy()
    labels = data_frame["num_encoder_layers"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Exact Match")
    plt.title("Energy-Exact Match Pareto Frontier (Encoder Layers)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_em_layers.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-F1 Pareto frontier for layers
def plot_energy_f1_layers(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["f1"].to_numpy()
    labels = data_frame["num_encoder_layers"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("F1")
    plt.title("Energy-F1 Pareto Frontier (Encoder Layers)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_f1_layers.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for layers
def plot_energy_latency_layers(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["num_encoder_layers"].astype(int).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(f"{li}", (xi, yi), textcoords = "offset points", xytext = (2, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Encoder Layers)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_latency_layers.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-EM Pareto frontier for precision
def plot_energy_em_precision(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["exact_match"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(li, (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Exact Match")
    plt.title("Energy-Exact Match Pareto Frontier (Inference Precision)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_em_precision.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-F1 Pareto frontier for precision
def plot_energy_f1_precision(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["f1"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(li, (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("F1")
    plt.title("Energy-F1 Pareto Frontier (Inference Precision)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_f1_precision.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for precision
def plot_energy_latency_precision(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    _setup_plot()
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(li, (xi, yi), textcoords = "offset points", xytext = (5, 5),
                     ha = "left", va = "bottom", fontsize = 9, zorder = 3)

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Inference Precision)")

    plt.xlim(*_auto_limits(x))
    plt.ylim(*_auto_limits(y))

    plt.tight_layout()
    output_path = run_directory / "energy_latency_precision.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path