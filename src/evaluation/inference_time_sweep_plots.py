from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# Read a JSON file from disk
def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

# Build one color per model from a matplotlib colormap
def build_model_color_map(model_labels: List[str], color_map_name: str) -> Dict[str, Any]:
    color_map = plt.get_cmap(color_map_name)
    number_of_models = len(model_labels)

    if number_of_models == 1:
        color_positions = [0.5]
    else:
        color_positions = [
            index / (number_of_models - 1)
            for index in range(number_of_models)
        ]

    return {
        model_label: color_map(color_position)
        for model_label, color_position in zip(model_labels, color_positions)
    }

# Return fixed marker styles per model to stay visually consistent across plots
def build_model_marker_map() -> Dict[str, str]:
    return {
        "BERT": "o",
        "DistilBERT": "s",
        "BERT-Freeze6": "^",
        "BERT-Edge": "D",
        "BERT-Prune50": "P",
    }

# Load one sweep summary from outputs and return plot-ready rows
def load_sweep_rows_from_summary(
    summary_json_path: Path,
    model_label: str,
) -> List[Dict[str, Any]]:
    payload = read_json_file(summary_json_path)
    rows = payload.get("rows", [])

    if len(rows) == 0:
        raise ValueError(f"No rows found in {summary_json_path}")

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row = dict(row)
        normalized_row["model_label_display"] = model_label
        normalized_rows.append(normalized_row)

    return normalized_rows

# Filter rows to those with numeric x/y values and a label field
def filter_rows_for_plotting(
    rows: List[Dict[str, Any]],
    x_field_name: str,
    y_field_name: str,
    label_field_name: str,
) -> List[Dict[str, Any]]:
    filtered_rows: List[Dict[str, Any]] = []

    for row in rows:
        x_value = row.get(x_field_name)
        y_value = row.get(y_field_name)
        label_value = row.get(label_field_name)

        if x_value in [None, "", "None"]:
            continue
        if y_value in [None, "", "None"]:
            continue
        if label_value in [None, "", "None"]:
            continue

        filtered_rows.append(row)

    return filtered_rows

# Plot a combined comparison figure for one inference-time sweep across all models
def plot_combined_sweep_comparison(
    sweep_rows_by_model: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    label_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
    color_map_name: str,
    annotated_model_label: Optional[str] = None,
    annotation_offset: Tuple[int, int] = (4, 4),
    annotation_alignment: Tuple[str, str] = ("left", "bottom"),
) -> None:
    model_labels = list(sweep_rows_by_model.keys())
    model_colors = build_model_color_map(
        model_labels=model_labels,
        color_map_name=color_map_name,
    )
    model_markers = build_model_marker_map()

    figure = plt.figure(figsize=(8, 6))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    all_x_values: List[float] = []
    all_y_values: List[float] = []

    plotting_rows_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for model_label, rows in sweep_rows_by_model.items():
        rows_filtered = filter_rows_for_plotting(
            rows=rows,
            x_field_name=x_field_name,
            y_field_name=y_field_name,
            label_field_name=label_field_name,
        )
        rows_sorted = sorted(rows_filtered, key=lambda row: float(row[x_field_name]))

        plotting_rows_by_model[model_label] = rows_sorted
        all_x_values.extend(float(row[x_field_name]) for row in rows_sorted)
        all_y_values.extend(float(row[y_field_name]) for row in rows_sorted)

    if len(all_x_values) == 0 or len(all_y_values) == 0:
        raise ValueError("No plottable rows found across the supplied sweep summaries.")

    plt.xlim(0, 1.0)
    plt.ylim(45, 80)

    for model_label, rows_sorted in plotting_rows_by_model.items():
        x_values = [float(row[x_field_name]) for row in rows_sorted]
        y_values = [float(row[y_field_name]) for row in rows_sorted]

        plt.plot(
            x_values,
            y_values,
            marker=model_markers.get(model_label, "o"),
            color=model_colors[model_label],
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=0.8,
            label=model_label,
        )

        if annotated_model_label is not None and model_label == annotated_model_label:
            for row in rows_sorted:
                x_value = float(row[x_field_name])
                y_value = float(row[y_field_name])

                label = str(row[label_field_name]).lower()

                # Special annotation placement for precision sweep
                if label in {"fp32", "fp16"}:
                    offset = (5, 5)
                    alignment = ("left", "bottom")   # top-right of point
                elif label == "bf16":
                    offset = (-5, 5)
                    alignment = ("right", "bottom")  # top-left of point
                else:
                    offset = annotation_offset
                    alignment = annotation_alignment

                plt.annotate(
                    str(row[label_field_name]),
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=offset,
                    ha=alignment[0],
                    va=alignment[1],
                    fontsize=8,
                )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

# Load all five model summaries for the precision sweep
def load_precision_sweep_rows(outputs_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    model_specs = [
        ("BERT", "precision_sweep_bert_full/summary.json"),
        ("DistilBERT", "precision_sweep_distilbert/summary.json"),
        ("BERT-Freeze6", "precision_sweep_bert_freeze_bottom_6/summary.json"),
        ("BERT-Edge", "precision_sweep_bert_edge/summary.json"),
        ("BERT-Prune50", "precision_sweep_bert_prune50/summary.json"),
    ]

    return {
        model_label: load_sweep_rows_from_summary(
            summary_json_path=outputs_root / relative_summary_path,
            model_label=model_label,
        )
        for model_label, relative_summary_path in model_specs
    }

# Load all five model summaries for the sequence-length sweep
def load_sequence_length_sweep_rows(outputs_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    model_specs = [
        ("BERT", "sequence_length_sweep_bert_full/summary.json"),
        ("DistilBERT", "sequence_length_sweep_distilbert/summary.json"),
        ("BERT-Freeze6", "sequence_length_sweep_bert_freeze_bottom_6/summary.json"),
        ("BERT-Edge", "sequence_length_sweep_bert_edge/summary.json"),
        ("BERT-Prune50", "sequence_length_sweep_bert_prune50/summary.json"),
    ]

    return {
        model_label: load_sweep_rows_from_summary(
            summary_json_path=outputs_root / relative_summary_path,
            model_label=model_label,
        )
        for model_label, relative_summary_path in model_specs
    }

# Load all five model summaries for the token-pruning sweep
def load_token_pruning_sweep_rows(outputs_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    model_specs = [
        ("BERT", "token_pruning_sweep_bert_full/summary.json"),
        ("DistilBERT", "token_pruning_sweep_distilbert/summary.json"),
        ("BERT-Freeze6", "token_pruning_sweep_bert_freeze_bottom_6/summary.json"),
        ("BERT-Edge", "token_pruning_sweep_bert_edge/summary.json"),
        ("BERT-Prune50", "token_pruning_sweep_bert_prune50/summary.json"),
    ]

    return {
        model_label: load_sweep_rows_from_summary(
            summary_json_path=outputs_root / relative_summary_path,
            model_label=model_label,
        )
        for model_label, relative_summary_path in model_specs
    }

# Build the three requested combined comparison figures
def generate_all_inference_time_sweep_plots(
    outputs_root: Path,
    figures_root: Path,
    color_map_name: str,
) -> None:
    precision_rows = load_precision_sweep_rows(outputs_root=outputs_root)
    plot_combined_sweep_comparison(
        sweep_rows_by_model=precision_rows,
        output_path=figures_root / "precision_sweep_energy_example_vs_f1.png",
        x_field_name="joules_per_inference_example",
        y_field_name="f1",
        label_field_name="precision_mode",
        x_axis_label="Energy (J / Example)",
        y_axis_label="F1",
        plot_title="Energy–Accuracy Trade-offs under Precision Reduction",
        color_map_name=color_map_name,
        annotated_model_label="BERT-Edge",
        annotation_offset=(0, 6),
        annotation_alignment=("center", "bottom"),
    )

    sequence_length_rows = load_sequence_length_sweep_rows(outputs_root=outputs_root)
    plot_combined_sweep_comparison(
        sweep_rows_by_model=sequence_length_rows,
        output_path=figures_root / "sequence_length_sweep_energy_example_vs_f1.png",
        x_field_name="joules_per_inference_example",
        y_field_name="f1",
        label_field_name="maximum_sequence_length",
        x_axis_label="Energy (J / Example)",
        y_axis_label="F1",
        plot_title="Energy–Accuracy Trade-offs under Sequence Length Truncation",
        color_map_name=color_map_name,
        annotated_model_label="BERT-Edge",
        annotation_offset=(-3, 5),
        annotation_alignment=("right", "bottom"),
    )

    token_pruning_rows = load_token_pruning_sweep_rows(outputs_root=outputs_root)
    plot_combined_sweep_comparison(
        sweep_rows_by_model=token_pruning_rows,
        output_path=figures_root / "token_pruning_sweep_energy_example_vs_f1.png",
        x_field_name="joules_per_inference_example",
        y_field_name="f1",
        label_field_name="token_pruning_keep_ratio_label",
        x_axis_label="Energy (J / Example)",
        y_axis_label="F1",
        plot_title="Energy–Accuracy Trade-offs under Dynamic Token Pruning",
        color_map_name=color_map_name,
        annotated_model_label="BERT-Edge",
        annotation_offset=(6, -6),
        annotation_alignment=("left", "top"),
    )

# Build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Generate combined inference-time sweep comparison plots from outputs summaries."
    )
    argument_parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Path to the outputs directory.",
    )
    argument_parser.add_argument(
        "--figures-root",
        default="outputs/figures",
        help="Directory where generated figures will be written.",
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
    figures_root = Path(arguments.figures_root)
    color_map_name = str(arguments.color_map)

    generate_all_inference_time_sweep_plots(
        outputs_root=outputs_root,
        figures_root=figures_root,
        color_map_name=color_map_name,
    )

    print(f"Figures written to: {figures_root}")

if __name__ == "__main__":
    main()