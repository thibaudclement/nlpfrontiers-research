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

    plt.figure()

    # Add grid and plot points with dashed lines to visualize frontier
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(
        data_frame["energy_per_example_j"],
        data_frame["accuracy"],
        marker = "o",
        linestyle = "--",
        linewidth = 1.0,
        alpha = 0.6,
    )

    # Annotate each point with max sequence length for interpretability
    for _, row in data_frame.iterrows():
        plt.annotate(
            f"{int(row['max_sequence_length'])}",
            (row["energy_per_example_j"], row["accuracy"]),
            textcoords = "offset points",
            xytext = (6, 6),
            ha = "left",
            fontsize = 9
        )

    # Add some padding around the min/max points for better visualization
    x_min = data_frame["energy_per_example_j"].min()
    x_max = data_frame["energy_per_example_j"].max()
    y_min = data_frame["accuracy"].min()
    y_max = data_frame["accuracy"].max()
    plt.xlim(max(0.0, x_min - 0.02), x_max + 0.03)
    plt.ylim(y_min - 0.002, y_max + 0.002)

    # Implement label axes and title
    plt.xlabel("Energy (Joule / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier")
    plt.tight_layout()

    output_path = run_directory / "pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Plot Pareto frontier for energy vs latency
def plot_energy_latency_pareto_frontier(pareto_csv_path: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_path)

    # Sort points by energy for consistent frontier plotting
    data_frame = data_frame.sort_values("energy_per_example_j", ascending = True)

    plt.figure()

    # Add grid and plot points with dashed lines to visualize frontier
    plt.grid(True, which = "both", linestyle = ":", linewidth = 0.6, alpha = 0.35)
    plt.plot(
        data_frame["energy_per_example_j"],
        data_frame["average_latency_per_example_ms"],
        marker = "o",
        linestyle = "--",
        linewidth = 1.0,
        alpha = 0.6,
    )

    # Annotate each point with max sequence length for interpretability
    for _, row in data_frame.iterrows():
        plt.annotate(
            f"{int(row['max_sequence_length'])}",
            (row["energy_per_example_j"], row["average_latency_per_example_ms"]),
            textcoords = "offset points",
            xytext = (6, 6),
            ha = "left",
            fontsize = 9
        )

    # Add some padding around the min/max points for better visualization
    x_min = data_frame["energy_per_example_j"].min()
    x_max = data_frame["energy_per_example_j"].max()
    y_min = data_frame["average_latency_per_example_ms"].min()
    y_max = data_frame["average_latency_per_example_ms"].max()
    plt.xlim(max(0.0, x_min - 0.02), x_max + 0.03)
    plt.ylim(max(0.0, y_min - 1.0), y_max + 1.5)

    # Implement label axes and title
    plt.xlabel("Energy (Joule / Example)")
    plt.ylabel("Average Latency (ms / Example)")
    plt.title("Energy-Latency Pareto Frontier")
    plt.tight_layout()

    output_path = run_directory / "energy_latency_pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path