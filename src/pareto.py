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
    plt.plot(data_frame["energy_per_example_j"], data_frame["accuracy"], marker="o")

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

    plt.xlabel("Energy (Joule / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier")
    plt.tight_layout()

    output_path = run_directory / "pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path