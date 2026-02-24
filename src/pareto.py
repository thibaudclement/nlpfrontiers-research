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

# Plot Pareto frontier for energy vs accuracy
def plot_energy_accuracy_pareto_frontier(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["accuracy"].to_numpy()

    # Use labels for annotations when they exist
    if "label" in data_frame.columns:
        labels = data_frame["label"].astype(str).to_numpy()
    else:
        labels = data_frame["max_sequence_length"].astype(int).astype(str).to_numpy()

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            li,
            (xi, yi),
            textcoords = "offset points",
            xytext = (4, 4),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier")
    
    # Force axes bounds for visual clarity
    plt.ylim(0.82, 0.94)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_accuracy_pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot Pareto frontier for energy vs latency
def plot_energy_latency_pareto_frontier(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    # Extract values and labels for plotting and annotation
    x = data_frame["energy_per_example_j"].to_numpy()
    y = data_frame["average_latency_per_example_ms"].to_numpy()

    # Use labels for annotations when they exist
    if "label" in data_frame.columns:
        labels = data_frame["label"].astype(str).to_numpy()
    else:
        labels = data_frame["max_sequence_length"].astype(int).astype(str).to_numpy()   

    # Create scatter plot with dashed lines to visualize frontier
    plt.figure()
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(x, y, linestyle = "--", linewidth = 1.0, alpha = 0.35, zorder = 1)
    plt.scatter(x, y, s = 36, alpha = 1.0, zorder = 2)

    # Annotate data points
    for xi, yi, li in zip(x, y, labels):
        plt.annotate(
            li,
            (xi, yi),
            textcoords = "offset points",
            xytext = (4, 4),
            ha = "left",
            va = "bottom",
            fontsize = 9,
            zorder = 3
        )

    # Implement label axes and title
    plt.xlabel("Energy (J / Example)")
    plt.ylabel("Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier")
    
    # Force axes bounds for visual clarity
    y_min, y_max = float(y.min()), float(y.max())
    plt.ylim(max(0.0, y_min - 5), y_max + 5)
    x_min, x_max = float(x.min()), float(x.max())
    plt.xlim(max(0.0, x_min - 0.05), x_max + 0.05)

    plt.tight_layout()

    output_path = run_directory / "energy_latency_pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path