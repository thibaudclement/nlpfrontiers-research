from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

# Model display configuration in plotting order
POINT_SPECS = [
    {"key": "bert_fp16", "label": "BERT + FP16", "family": "BERT"},
    {"key": "distilbert", "label": "DistilBERT", "family": "DistilBERT"},
    {"key": "bert_prune50_08_tp", "label": "BERT-Prune50 + 0.8 Token Pruning", "family": "BERT-Prune50"},
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

# Read a JSON file from disk
def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

# Build a fixed color per model family so colors stay consistent across figures
def build_family_color_map(color_map_name: str) -> Dict[str, Any]:
    color_map = plt.get_cmap(color_map_name)

    # Give BERT-EE its own distinct slot while keeping the original five unchanged.
    family_positions = {
        "BERT": 0.00,
        "DistilBERT": 0.25,
        "BERT-Freeze6": 0.50,
        "BERT-Edge": 0.75,
        "BERT-Prune50": 1.00,
        "BERT-EE": 0.60,
    }

    return {
        family: color_map(position)
        for family, position in family_positions.items()
    }

# Compute total energy in joules for a given deployment regime
def compute_total_energy_joules(
    training_energy_joules: float,
    joules_per_inference_example: float,
    number_of_queries: int,
) -> float:
    return float(training_energy_joules) + (
        float(number_of_queries) * float(joules_per_inference_example)
    )

# Load and normalize model points from deployment_regime_analysis.json
def load_model_points(data_json_path: Path) -> List[Dict[str, Any]]:
    payload = read_json_file(data_json_path)

    model_points: List[Dict[str, Any]] = []

    for point_spec in POINT_SPECS:
        point_payload = payload[point_spec["key"]]

        model_points.append(
            {
                "label": point_spec["label"],
                "family": point_spec["family"],
                "training_energy_joules": float(point_payload["training_energy"]),
                "joules_per_inference_example": float(point_payload["joules_per_inference_example"]),
            }
        )

    return model_points

# Plot deployment regime analysis
def plot_deployment_regime_analysis(
    model_points: List[Dict[str, Any]],
    output_path: Path,
    plot_title: str,
    color_map_name: str,
) -> None:
    family_colors = build_family_color_map(color_map_name=color_map_name)

    # Use one point per order of magnitude from 10^4 to 10^8
    query_counts = [10**exponent for exponent in range(4, 9)]

    figure = plt.figure(figsize=(8, 6))

    for model_point in model_points:
        label = str(model_point["label"])
        family = str(model_point["family"])
        training_energy_joules = float(model_point["training_energy_joules"])
        joules_per_inference_example = float(model_point["joules_per_inference_example"])

        total_energy_kj_values = [
            compute_total_energy_joules(
                training_energy_joules=training_energy_joules,
                joules_per_inference_example=joules_per_inference_example,
                number_of_queries=query_count,
            ) / 1000.0
            for query_count in query_counts
        ]

        plt.plot(
            query_counts,
            total_energy_kj_values,
            marker=MARKER_STYLES[family],
            markersize=7,
            color=family_colors[family],
            markeredgecolor="black",
            markeredgewidth=0.7,
            linewidth=1.8,
            label=label,
        )

    plt.xscale("log")
    plt.xlabel("Number of Inference Queries (Log Scale)")
    plt.ylabel("Total Energy (kJ)")
    plt.title(plot_title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

# Build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Plot deployment regime analysis using E_total(N) = E_train + N * E_inference."
    )
    argument_parser.add_argument(
        "--data-path",
        default=str(Path(__file__).with_name("deployment_regime_analysis.json")),
        help="Path to the deployment_regime_analysis.json file.",
    )
    argument_parser.add_argument(
        "--output-path",
        default="outputs/figures/deployment_regime_analysis.png",
        help="Path to the generated PNG file.",
    )
    argument_parser.add_argument(
        "--plot-title",
        default="Deployment Regime Analysis",
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

    model_points = load_model_points(data_json_path=data_path)

    plot_deployment_regime_analysis(
        model_points=model_points,
        output_path=output_path,
        plot_title=plot_title,
        color_map_name=color_map_name,
    )

    print(f"Plot written to: {output_path}")

if __name__ == "__main__":
    main()