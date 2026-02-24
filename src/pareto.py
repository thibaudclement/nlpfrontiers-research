from pathlib import Path
from turtle import pd
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas

# Save Pareto table for comparison
def save_pareto_table(rows: List[Dict[str, object]], run_directory: Path) -> Path:
    data_frame = pd.DataFrame(rows)
    output_path = run_directory / "pareto.csv"
    data_frame.to_csv(output_path, index = False)
    return output_path

# Plot Pareto frontier for energy vs accuracy
def plot_energy_accuracy_pareto_frontier(pareto_csv_paths: Path, run_directory: Path) -> Path:
    data_frame = pd.read_csv(pareto_csv_paths)
    plt.figure()
    plt.scatter(data_frame["energy_per_example_j"], data_frame["accuracy"])
    plt.xlabel("Energy (Joule / Example)")
    plt.ylabel("Accuracy")
    plt.title("Energy-Accuracy Pareto Frontier")
    plt.tight_layout()
    output_path = run_directory / "pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    return output_path