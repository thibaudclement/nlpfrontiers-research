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

# Extract one architecture-level point from a baseline summary.json file
def extract_model_point_from_summary(summary_json_path: Path, label: str) -> Dict[str, Any]:
    payload = read_json_file(summary_json_path)

    return {
        "label": label,
        "f1": float(payload["metrics_thresholded"]["f1"]),
        "joules_per_inference_example": float(
            payload["inference_energy"]["joules_per_inference_example"]
        ),
    }

# Load the five architecture points from the outputs directory
def load_default_model_points(outputs_root: Path) -> List[Dict[str, Any]]:
    model_specs = [
        ("BERT", "base_baseline_squad_v2/summary.json"),
        ("DistilBERT", "distilbert_baseline_squad_v2/summary.json"),
        ("BERT-Freeze6", "freeze_bottom_six_baseline_squad_v2/summary.json"),
        ("BERT-Edge", "train_first_last_base_baseline_squad_v2/summary.json"),
        ("BERT-Prune50", "prune_refinetune_base_squad_v2/summary.json"),
    ]

    model_points: List[Dict[str, Any]] = []

    for label, relative_summary_path in model_specs:
        summary_json_path = outputs_root / relative_summary_path
        model_points.append(
            extract_model_point_from_summary(
                summary_json_path=summary_json_path,
                label=label,
            )
        )

    return model_points

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

# Plot one point per model for the architecture-level Pareto frontier
def plot_model_architecture_pareto(
    model_points: List[Dict[str, Any]],
    output_path: Path,
    plot_title: str,
    color_map_name: str,
) -> None:
    annotation_offsets = {
        "DistilBERT": (10, 5),
        "BERT-Edge": (-10, 5),
        "BERT-Freeze6": (-10, -5),
        "BERT-Prune50": (10, 0),
        "BERT": (-10, 5),
    }

    annotation_alignments = {
        "DistilBERT": ("left", "bottom"),
        "BERT-Edge": ("right", "bottom"),
        "BERT-Freeze6": ("right", "top"),
        "BERT-Prune50": ("left", "top"),
        "BERT": ("right", "bottom"),
    }

    # Marker style per model
    marker_styles = {
        "BERT": "o",           
        "DistilBERT": "s",     
        "BERT-Freeze6": "^",   
        "BERT-Edge": "D",      
        "BERT-Prune50": "P",   
    }

    model_colors = build_model_color_map(
        model_points=model_points,
        color_map_name=color_map_name,
    )

    figure = plt.figure(figsize=(8, 6))

    for model_point in model_points:
        label = str(model_point["label"])
        x_value = float(model_point["joules_per_inference_example"])
        y_value = float(model_point["f1"])
        color = model_colors[label]
        marker = marker_styles[label]

        plt.scatter(
            [x_value],
            [y_value],
            s=90,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        x_offset, y_offset = annotation_offsets[label]
        horizontal_alignment, vertical_alignment = annotation_alignments[label]

        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
        )

    plt.xlim(0.20, 0.50)
    plt.ylim(68, 77)

    plt.xlabel("Energy per Example (J)")
    plt.ylabel("F1")
    plt.title(plot_title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

# Build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Plot the architecture-level energy-accuracy Pareto frontier."
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
        default="Inference Energy–Accuracy Trade-offs Across Model Architectures",
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

    model_points = load_default_model_points(outputs_root=outputs_root)

    plot_model_architecture_pareto(
        model_points=model_points,
        output_path=output_path,
        plot_title=plot_title,
        color_map_name=color_map_name,
    )

    print(f"Plot written to: {output_path}")

if __name__ == "__main__":
    main()