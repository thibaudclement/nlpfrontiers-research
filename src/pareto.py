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
    plt.title("Energy-Accuracy Pareto Frontier (Max Sequence Length in Tokens)")
    
    # Set axes bounds for visual clarity
    plt.ylim(0.82, 0.94)
    plt.xlim(0.00, 0.25)

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
    plt.title("Energy-Latency Pareto Frontier (Max Sequence Length in Tokens)")
    
    # Set axes bounds for visual clarity
    plt.ylim(0.00, 4.00)
    plt.xlim(0.00, 0.25)

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
    plt.title("Energy-Accuracy Pareto Frontier (Encoder Layers)")
    
    # Set axes bounds for visual clarity
    plt.ylim(0.40, 1.00)
    plt.xlim(0.00, 0.25)

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
    plt.title("Energy-Latency Pareto Frontier (Encoder Layers)")
    
    # Set axes bounds for visual clarity
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

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["accuracy"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with precision for interpretability
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

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (Inference Precision)")

    # Set axes bounds for visual clarity
    plt.ylim(0.92, 0.94)
    plt.xlim(0.025, 0.250)

    plt.tight_layout()
    output_path = run_directory / "energy_accuracy_precision.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for precision
def plot_energy_latency_precision(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()
    labels = data_frame["label"].astype(str).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points with precision for interpretability
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

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (Inference Precision)")

    # Set axes bounds for visual clarity
    plt.ylim(0.0, 3.5)
    plt.xlim(0.025, 0.250)

    plt.tight_layout()
    output_path = run_directory / "energy_latency_precision.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

# Plot energy-accuracy Pareto frontier for precision and sequence-length combination
def plot_energy_accuracy_combination(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Plot only FP16 data points
    fp16_frame = data_frame[data_frame["precision"] == "fp16"].copy()

    # Sort points by energy for consistent frontier plotting
    fp16_frame = fp16_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = fp16_frame["energy_per_example_j"].to_numpy()
    y = fp16_frame["accuracy"].to_numpy()
    labels = fp16_frame["max_sequence_length"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, color = "C0", zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, color = "C0", zorder = 2)

    # Annotate data points with max sequence length for interpretability
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

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier (FP16 and Max Sequence Length in Tokens)")

    # Set axes bounds for visual clarity
    plt.ylim(0.82, 0.94)
    plt.xlim(0.00, 0.06)

    plt.tight_layout()
    output_path = run_directory / "energy_accuracy_combination.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot energy-latency Pareto frontier for precision and sequence-length combination
def plot_energy_latency_combination(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Plot only FP16 data points
    fp16_frame = data_frame[data_frame["precision"] == "fp16"].copy()
    fp16_frame = fp16_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = fp16_frame["energy_per_example_j"].to_numpy()
    y = fp16_frame["average_latency_per_example_ms"].to_numpy()
    labels = fp16_frame["max_sequence_length"].astype(int).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, color = "C0", zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, color = "C0", zorder = 2)

    # Annotate data points with max sequence length for interpretability
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

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier (FP16 and Max Sequence Length in Tokens)")

    # Set axes bounds for visual clarity
    plt.ylim(0.0, 1.0)
    plt.xlim(0.00, 0.06)

    plt.tight_layout()
    output_path = run_directory / "energy_latency_precision.png"
    plt.savefig(output_path)
    plt.close()
    return output_path