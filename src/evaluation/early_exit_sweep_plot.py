from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

# Read a JSON file from disk
def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

# Load sweep rows from a JSON file
def load_early_exit_sweep_rows(sweep_rows_json_path: Path) -> List[Dict[str, Any]]:
    payload = read_json_file(sweep_rows_json_path)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
    else:
        raise ValueError(
            f"Unsupported sweep rows format in {sweep_rows_json_path}. "
            f"Expected a list or a dict with a 'rows' field."
        )

    if len(rows) == 0:
        raise ValueError(f"No rows found in {sweep_rows_json_path}")

    return rows

# Filter rows to those with numeric x/y values and threshold labels
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

# Plot the early-exit sweep curve
def plot_early_exit_sweep(
    rows: List[Dict[str, Any]],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    label_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    rows_filtered = filter_rows_for_plotting(
        rows=rows,
        x_field_name=x_field_name,
        y_field_name=y_field_name,
        label_field_name=label_field_name,
    )
    rows_sorted = sorted(rows_filtered, key=lambda row: float(row[x_field_name]))

    if len(rows_sorted) == 0:
        raise ValueError("No plottable rows were found for the requested fields.")

    x_values = [float(row[x_field_name]) for row in rows_sorted]
    y_values = [float(row[y_field_name]) for row in rows_sorted]
    labels = [f"{float(row[label_field_name]):.2f}" for row in rows_sorted]

    figure = plt.figure(figsize=(8, 6))

    # Distinct visual identity relative to the five-model sweep figures
    plt.plot(
        x_values,
        y_values,
        marker="X",
        markersize=9,
        color="darkorange",
        markeredgecolor="black",
        markeredgewidth=0.7,
        linewidth=0.8,
        label="BERT-EE",
    )

    for x_value, y_value, label in zip(x_values, y_values, labels):
        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    plt.xlim(0.0, 1.0)
    plt.ylim(45.0, 80.0)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
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
        description="Plot the early-exit threshold sweep."
    )
    argument_parser.add_argument(
        "--sweep-rows-path",
        default="outputs/bert_early_exit_threshold_sweep/sweep_rows.json",
        help="Path to the early-exit sweep_rows.json file.",
    )
    argument_parser.add_argument(
        "--output-path",
        default="outputs/figures/bert_early_exit_energy_example_vs_f1.png",
        help="Path to the generated PNG file.",
    )
    argument_parser.add_argument(
        "--plot-title",
        default="Energy–Accuracy Trade-offs under Early Exit",
        help="Title of the plot.",
    )
    return argument_parser

# Run the CLI entry point
def main() -> None:
    arguments = build_argument_parser().parse_args()

    sweep_rows_path = Path(arguments.sweep_rows_path)
    output_path = Path(arguments.output_path)
    plot_title = str(arguments.plot_title)

    rows = load_early_exit_sweep_rows(sweep_rows_json_path=sweep_rows_path)

    plot_early_exit_sweep(
        rows=rows,
        output_path=output_path,
        x_field_name="joules_per_inference_example",
        y_field_name="f1",
        label_field_name="early_exit_confidence_threshold",
        x_axis_label="Energy (J / Example)",
        y_axis_label="F1",
        plot_title=plot_title,
    )

    print(f"Plot written to: {output_path}")

if __name__ == "__main__":
    main()