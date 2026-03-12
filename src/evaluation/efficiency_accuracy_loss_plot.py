from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

# Point display configuration in plotting order
POINT_SPECS = [
    {"key": "bert_baseline", "label": "BERT", "family": "BERT"},
    {"key": "distilbert", "label": "DistilBERT", "family": "DistilBERT"},
    {"key": "bert_fp16", "label": "BERT + FP16", "family": "BERT"},
    {"key": "bert_ee_0.6", "label": "BERT-EE + 0.6", "family": "BERT-EE"},
    {"key": "bert_ee_0.5", "label": "BERT-EE + 0.5", "family": "BERT-EE"},
    {"key": "bert_freeze6_0.8_tp", "label": "BERT-Freeze6 + 0.8 TP", "family": "BERT-Freeze6"},
    {"key": "distilbert_0.8_tp", "label": "DistilBERT + 0.8 TP", "family": "DistilBERT"},
    {"key": "bert_length_320", "label": "BERT + length 320", "family": "BERT"},
    {"key": "bert_edge_length_288", "label": "BERT-Edge + length 288", "family": "BERT-Edge"},
]

# Marker style per model family
MARKER_STYLES = {
    "BERT": "o",
    "DistilBERT": "s",
    "BERT-Freeze6": "^",
    "BERT-Edge": "D",
    "BERT-Prune50": "P",
    "BERT-EE": "X",
}

# Annotation style per displayed label
ANNOTATION_STYLES = {
    "BERT": {"xytext": (5, 5), "ha": "left", "va": "bottom"},
    "DistilBERT": {"xytext": (-5, 5), "ha": "right", "va": "bottom"},
    "BERT + FP16": {"xytext": (-5, 5), "ha": "right", "va": "bottom"},
    "BERT-EE + 0.6": {"xytext": (-5, -5), "ha": "right", "va": "top"},
    "BERT-EE + 0.5": {"xytext": (-5, -5), "ha": "right", "va": "top"},
    "BERT-Freeze6 + 0.8 TP": {"xytext": (5, -8), "ha": "left", "va": "top"},
    "DistilBERT + 0.8 TP": {"xytext": (-5, 5), "ha": "right", "va": "bottom"},
    "BERT + length 320": {"xytext": (5, 5), "ha": "left", "va": "bottom"},
    "BERT-Edge + length 288": {"xytext": (5, 5), "ha": "left", "va": "bottom"},
}

# Read a JSON file from disk
def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

# Build one color per model family from a matplotlib colormap
def build_family_color_map(color_map_name: str) -> Dict[str, Any]:
    family_order = [
        "BERT",
        "DistilBERT",
        "BERT-Freeze6",
        "BERT-Edge",
        "BERT-Prune50",
        "BERT-EE",
    ]

    color_map = plt.get_cmap(color_map_name)
    number_of_families = len(family_order)

    color_positions = [
        index / (number_of_families - 1)
        for index in range(number_of_families)
    ]

    return {
        family: color_map(color_position)
        for family, color_position in zip(family_order, color_positions)
    }

# Compute energy savings (%) and accuracy loss (%) relative to the BERT baseline
def compute_relative_metrics(
    baseline_energy: float,
    baseline_f1: float,
    technique_energy: float,
    technique_f1: float,
) -> Dict[str, float]:
    energy_savings_percent = 100.0 * (
        (baseline_energy - technique_energy) / baseline_energy
    )

    accuracy_loss_percent = 100.0 * (
        (baseline_f1 - technique_f1) / baseline_f1
    )

    return {
        "energy_savings_percent": energy_savings_percent,
        "accuracy_loss_percent": accuracy_loss_percent,
    }

# Load and normalize comparison points from efficiency_accuracy_loss.json
def load_comparison_points(data_json_path: Path) -> List[Dict[str, Any]]:
    payload = read_json_file(data_json_path)

    baseline_energy = float(payload["bert_baseline"]["joules_per_inference_example"])
    baseline_f1 = float(payload["bert_baseline"]["f1"])

    comparison_points: List[Dict[str, Any]] = []

    for point_spec in POINT_SPECS:
        point_payload = payload[point_spec["key"]]

        technique_energy = float(point_payload["joules_per_inference_example"])
        technique_f1 = float(point_payload["f1"])

        relative_metrics = compute_relative_metrics(
            baseline_energy=baseline_energy,
            baseline_f1=baseline_f1,
            technique_energy=technique_energy,
            technique_f1=technique_f1,
        )

        comparison_points.append(
            {
                "label": point_spec["label"],
                "family": point_spec["family"],
                "energy_savings_percent": relative_metrics["energy_savings_percent"],
                "accuracy_loss_percent": relative_metrics["accuracy_loss_percent"],
            }
        )

    return comparison_points

# Plot the chart
def plot_energy_savings_accuracy_loss(
    comparison_points: List[Dict[str, Any]],
    output_path: Path,
    plot_title: str,
    color_map_name: str,
) -> None:
    family_colors = build_family_color_map(color_map_name=color_map_name)

    figure = plt.figure(figsize=(8, 6))

    for point in comparison_points:
        label = str(point["label"])
        family = str(point["family"])
        x_value = float(point["energy_savings_percent"])
        y_value = float(point["accuracy_loss_percent"])

        plt.scatter(
            [x_value],
            [y_value],
            s=90,
            color=family_colors[family],
            marker=MARKER_STYLES[family],
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        annotation_style = ANNOTATION_STYLES.get(
            label,
            {"xytext": (6, 6), "ha": "left", "va": "bottom"},
        )

        plt.annotate(
            label,
            (x_value, y_value),
            xytext=annotation_style["xytext"],
            textcoords="offset points",
            ha=annotation_style["ha"],
            va=annotation_style["va"],
        )

    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.xlabel("Energy Savings (%)")
    plt.ylabel("Accuracy Loss (%)")
    plt.title(plot_title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

# Build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Plot energy savings (%) vs. accuracy loss (%) relative to the BERT baseline."
    )
    argument_parser.add_argument(
        "--data-path",
        default=str(Path(__file__).with_name("efficiency_accuracy_loss.json")),
        help="Path to the efficiency_accuracy_loss.json file.",
    )
    argument_parser.add_argument(
        "--output-path",
        default="outputs/figures/energy_savings_vs_accuracy_loss.png",
        help="Path to the generated PNG file.",
    )
    argument_parser.add_argument(
        "--plot-title",
        default="Energy Savings vs. Accuracy Loss Relative to BERT",
        help="Title of the plot.",
    )
    argument_parser.add_argument(
        "--color-map",
        default="viridis",
        help="Matplotlib colormap name to use for family colors (for example: viridis, plasma, cividis, magma).",
    )
    return argument_parser

# Run the CLI entry point
def main() -> None:
    arguments = build_argument_parser().parse_args()

    data_path = Path(arguments.data_path)
    output_path = Path(arguments.output_path)
    plot_title = str(arguments.plot_title)
    color_map_name = str(arguments.color_map)

    comparison_points = load_comparison_points(data_json_path=data_path)

    plot_energy_savings_accuracy_loss(
        comparison_points=comparison_points,
        output_path=output_path,
        plot_title=plot_title,
        color_map_name=color_map_name,
    )

    print(f"Plot written to: {output_path}")

if __name__ == "__main__":
    main()