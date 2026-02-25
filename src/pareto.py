from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd

# Save Pareto table for comparison
def save_pareto_table(rows: List[Dict[str, object]], run_directory: Path) -> Path:
    data_frame = pd.DataFrame(rows)
    output_path = run_directory / "pareto.csv"
    data_frame.to_csv(output_path, index = False)
    return output_path

# Plot energy-accuracy Pareto frontier for max sequence length
def plot_energy_accuracy_max_sequence_length(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["accuracy"].to_numpy()
    labels = data_frame["max_sequence_length"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with max sequence length for interpretability
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            f"{li}",
            (xi, yi),
            textcoords = "offset points",
            xytext = (5, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (Max Sequence Length)")
    
    # Force axes bounds for visual clarity
    plt.ylim(0.82, 0.94)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_accuracy_max_sequence_length.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for max sequence length
def plot_energy_latency_max_sequence_length(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["max_sequence_length"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with max sequence length for interpretability
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            f"{li}",
            (xi, yi),
            textcoords = "offset points",
            xytext = (2, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Max Sequence Length)")
    
    # Force axes bounds for visual clarity
    y_min, y_max = float(y.min()), float(y.max())
    plt.ylim(max(0.0, y_min - 5), y_max + 5)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_latency_max_sequence_length.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-accuracy Pareto frontier for layers
def plot_energy_accuracy_layers(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["accuracy"].to_numpy()
    labels = data_frame["num_encoder_layers"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with layer count for interpretability
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            f"{li}",
            (xi, yi),
            textcoords = "offset points",
            xytext = (5, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (Layer Reduction)")
    
    # Force axes bounds for visual clarity (keep consistent with seq-len plots)
    plt.ylim(0.40, 1.00)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_accuracy_layers.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for layers
def plot_energy_latency_layers(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["num_encoder_layers"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with layer count for interpretability
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            f"{li}",
            (xi, yi),
            textcoords = "offset points",
            xytext = (2, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Layer Reduction)")
    
    # Force axes bounds for visual clarity (tighter latency padding than ±5ms)
    plt.ylim(0.0, 3.5)
    plt.xlim(0.0, 0.25)

    plt.tight_layout()

    output_path = run_directory / "energy_latency_layers.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-accuracy Pareto frontier for precision
def plot_energy_accuracy_precision(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["accuracy"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            li,
            (xi, yi),
            textcoords = "offset points",
            xytext = (5, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3,
        )

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (Precision)")

    y_min, y_max = float(y.min()), float(y.max())
    plt.ylim(max(0.0, y_min - 0.02), min(1.0, y_max + 0.02))
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.02), x_max + 0.02)

    plt.tight_layout()
    output_path = run_directory / "energy_accuracy_precision.png"
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

    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            li,
            (xi, yi),
            textcoords = "offset points",
            xytext = (5, 5),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3,
        )

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Precision)")

    y_min, y_max = float(y.min()), float(y.max())
    plt.ylim(max(0.0, y_min - 0.2), y_max + 0.3)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.02), x_max + 0.02)

    plt.tight_layout()
    output_path = run_directory / "energy_latency_precision.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-accuracy Pareto frontier for precision and sequence-length combination
def plot_energy_accuracy_combination(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)

    # Plot one curve per precision
    for precision in ["fp32", "fp16"]:
        subset = data_frame[data_frame["precision"] == precision].copy()
        if subset.empty:
            continue

        # Sort points by energy for consistent frontier plotting
        subset = subset.sort_values("energy_per_example_j", ascending = True)

        x = subset["energy_per_example_j"].to_numpy()
        y = subset["accuracy"].to_numpy()
        labels = subset["max_sequence_length"].astype(int).to_numpy()

        plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, label = precision, zorder = 1)
        plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

        # Annotate with sequence length
        for xi, yi, li in zip(x, y, labels):
            plt.annotate(
                f"{li}",
                (xi, yi),
                textcoords = "offset points",
                xytext = (4, 4),
                ha = "left",
                va = "bottom",
                fontsize = 9,
                zorder = 3,
            )

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (Precision and Sequence Length)")
    plt.legend(title = "Precision", frameon = False)

    plt.ylim(0.82, 0.94)
    x_min = float(data_frame["energy_per_example_j"].min())
    x_max = float(data_frame["energy_per_example_j"].max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_accuracy_combination.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for precision and sequence-length combination
def plot_energy_latency_combination(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)

    for precision in ["fp32", "fp16"]:
        subset = data_frame[data_frame["precision"] == precision].copy()
        if subset.empty:
            continue

        subset = subset.sort_values("energy_per_example_j", ascending = True)

        x = subset["energy_per_example_j"].to_numpy()
        y = subset["average_latency_per_example_ms"].to_numpy()
        labels = subset["max_sequence_length"].astype(int).to_numpy()

        plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, label = precision, zorder = 1)
        plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

        for xi, yi, li in zip(x, y, labels):
            plt.annotate(
                f"{li}",
                (xi, yi),
                textcoords = "offset points",
                xytext = (4, 4),
                ha = "left",
                va = "bottom",
                fontsize = 9,
                zorder = 3,
            )

    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Precision × Sequence Length)")
    plt.legend(title = "Precision", frameon = False)

    plt.ylim(0.0, 4.0)
    x_min = float(data_frame["energy_per_example_j"].min())
    x_max = float(data_frame["energy_per_example_j"].max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_latency_precision.png"
    plt.savefig(output_path)
    plt.close()
    return output_path