from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

# Read a JSON file from disk
def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

# Extract one model's total training and inference energy from a summary.json file
def extract_model_energy_from_summary(summary_json_path: Path, label: str) -> Dict[str, Any]:
    payload = read_json_file(summary_json_path)

    return {
        "label": label,
        "training_energy_joules": float(payload["training_energy"]["energy_joules"]),
        "inference_energy_joules": float(payload["inference_energy"]["energy_joules"]),
    }

# Load the five architecture points from the outputs directory
def load_default_model_energies(outputs_root: Path) -> List[Dict[str, Any]]:
    model_specs = [
        ("BERT", "base_baseline_squad_v2/summary.json"),
        ("DistilBERT", "distilbert_baseline_squad_v2/summary.json"),
        ("BERT-Freeze6", "freeze_bottom_six_baseline_squad_v2/summary.json"),
        ("BERT-Edge", "train_first_last_base_baseline_squad_v2/summary.json"),
        ("BERT-Prune50", "prune_refinetune_base_squad_v2/summary.json"),
    ]

    model_energies: List[Dict[str, Any]] = []

    for label, relative_summary_path in model_specs:
        summary_json_path = outputs_root / relative_summary_path
        model_energies.append(
            extract_model_energy_from_summary(
                summary_json_path=summary_json_path,
                label=label,
            )
        )

    return model_energies

# Build one color per model from a matplotlib colormap
def build_model_color_map(model_points: List[Dict[str, Any]], color_map_name: str) -> Dict[str, Any]:
    color_map = plt.get_cmap(color_map_name)
    number_of_models = len(model_points)

    if number_of_models == 1:
        color_positions = [0.5]
    else:
        color_positions = [
            index / (number_of_models - 1)
            for index in range(number_of_models)
        ]

    return {
        str(model_point["label"]): color_map(color_position)
        for model_point, color_position in zip(model_points, color_positions)
    }

# Plot side-by-side bar charts for total training and inference energy
def plot_training_inference_energy_split(
    model_energies: List[Dict[str, Any]],
    output_path: Path,
    plot_title: str,
    color_map_name: str,
) -> None:
    labels = [str(model_energy["label"]) for model_energy in model_energies]
    training_values = [
        float(model_energy["training_energy_joules"]) / 1000.0
        for model_energy in model_energies
    ]

    inference_values = [
        float(model_energy["inference_energy_joules"]) / 1000.0
        for model_energy in model_energies
    ]

    model_colors = build_model_color_map(
        model_points=model_energies,
        color_map_name=color_map_name,
    )
    bar_colors = [model_colors[label] for label in labels]

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Training energy subplot
    training_bars = axes[0].bar(
        labels,
        training_values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].set_title("Total Training Energy")
    axes[0].set_ylabel("Energy (kJ)")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].tick_params(axis="x", rotation=30)

    # Inference energy subplot
    inference_bars = axes[1].bar(
        labels,
        inference_values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].set_title("Total Inference Energy (Validation Set)")
    axes[1].set_ylabel("Energy (kJ)")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].tick_params(axis="x", rotation=30)

    # Add compact value labels above bars
    for bar in training_bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in inference_bars:
        height = bar.get_height()
        axes[1].annotate(
            f"{height:,.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(plot_title)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

# Build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Plot total training and inference energy for model architectures."
    )
    argument_parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Path to the outputs directory.",
    )
    argument_parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the generated PNG file.",
    )
    argument_parser.add_argument(
        "--plot-title",
        default="Training vs. Inference Energy by Model",
        help="Title of the plot.",
    )
    argument_parser.add_argument(
        "--color-map",
        default="viridis",
        help="Matplotlib colormap name to use for model colors (for example: viridis, plasma, cividis, magma).",
    )
    return argument_parser

# Run the CLI entry point
def main() -> None:
    arguments = build_argument_parser().parse_args()

    outputs_root = Path(arguments.outputs_root)
    output_path = Path(arguments.output_path)
    plot_title = str(arguments.plot_title)
    color_map_name = str(arguments.color_map)

    model_energies = load_default_model_energies(outputs_root=outputs_root)

    plot_training_inference_energy_split(
        model_energies=model_energies,
        output_path=output_path,
        plot_title=plot_title,
        color_map_name=color_map_name,
    )

    print(f"Plot written to: {output_path}")

if __name__ == "__main__":
    main()