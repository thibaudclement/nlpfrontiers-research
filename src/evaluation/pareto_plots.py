from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# Save sweep rows as a CSV table
def save_sequence_length_sweep_rows_to_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if len(rows) == 0:
        raise ValueError("Cannot save an empty sweep table.")

    field_names = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# Load sweep rows from CSV table
def load_sequence_length_sweep_rows_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)

# Filter rows to those that have numeric x/y values and a label
def filter_rows_for_plotting(
    rows: List[Dict[str, object]],
    x_field_name: str,
    y_field_name: str,
    label_field_name: str,
) -> List[Dict[str, object]]:
    filtered_rows: List[Dict[str, object]] = []

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

# Plot one sweep curve for a single model checkpoint
def plot_single_sweep(
    rows: List[Dict[str, object]],
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
    labels = [str(row[label_field_name]) for row in rows_sorted]

    # Create a simple scatter-line plot and annotate each point.
    plt.figure()
    plt.plot(x_values, y_values, marker="o")

    # Add margins so point labels do not clip.
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)

    x_margin = x_range * 0.08 if x_range > 0 else 0.01
    y_margin = y_range * 0.08 if y_range > 0 else 0.5

    x_min_with_margin = min(x_values) - x_margin
    x_max_with_margin = max(x_values) + x_margin
    y_min_with_margin = min(y_values) - y_margin
    y_max_with_margin = max(y_values) + y_margin

    plt.xlim(x_min_with_margin, x_max_with_margin)
    plt.ylim(y_min_with_margin, y_max_with_margin)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    for x_value, y_value, label in zip(x_values, y_values, labels):
        x_offset = 4
        y_offset = 4

        plt.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot overlay comparison across multiple sweep CSV files
def plot_sweep_comparison(
    csv_paths: List[Path],
    model_labels: List[str],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    label_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    if len(csv_paths) != len(model_labels):
        raise ValueError("csv_paths and model_labels must have the same length.")

    plt.figure()
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    all_x_values: List[float] = []
    all_y_values: List[float] = []
    all_rows_by_model: List[tuple[List[Dict[str, str]], str]] = []

    # Load all rows first so we can compute global axis margins.
    for csv_path, model_label in zip(csv_paths, model_labels):
        rows = load_sequence_length_sweep_rows_from_csv(csv_path=csv_path)
        rows_filtered = filter_rows_for_plotting(
            rows=rows,
            x_field_name=x_field_name,
            y_field_name=y_field_name,
            label_field_name=label_field_name,
        )
        rows_sorted = sorted(rows_filtered, key=lambda row: float(row[x_field_name]))

        all_rows_by_model.append((rows_sorted, model_label))
        all_x_values.extend(float(row[x_field_name]) for row in rows_sorted)
        all_y_values.extend(float(row[y_field_name]) for row in rows_sorted)

    if len(all_x_values) == 0 or len(all_y_values) == 0:
        raise ValueError("No plottable rows were found across the supplied CSV files.")

    # Add margins so labels do not clip at the chart boundaries.
    x_range = max(all_x_values) - min(all_x_values)
    y_range = max(all_y_values) - min(all_y_values)

    x_margin = x_range * 0.08 if x_range > 0 else 0.01
    y_margin = y_range * 0.08 if y_range > 0 else 0.5

    x_min_with_margin = min(all_x_values) - x_margin
    x_max_with_margin = max(all_x_values) + x_margin
    y_min_with_margin = min(all_y_values) - y_margin
    y_max_with_margin = max(all_y_values) + y_margin

    plt.xlim(x_min_with_margin, x_max_with_margin)
    plt.ylim(y_min_with_margin, y_max_with_margin)

    # Overlay one line per model / checkpoint.
    for rows_sorted, model_label in all_rows_by_model:
        x_values = [float(row[x_field_name]) for row in rows_sorted]
        y_values = [float(row[y_field_name]) for row in rows_sorted]

        plt.plot(x_values, y_values, marker="o", label=model_label)

        for row in rows_sorted:
            x_value = float(row[x_field_name])
            y_value = float(row[y_field_name])

            x_offset = 4
            y_offset = 4

            plt.annotate(
                str(row[label_field_name]),
                (x_value, y_value),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
            )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot one sequence-length sweep curve for a single model checkpoint
def plot_single_sequence_length_sweep(
    rows: List[Dict[str, object]],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    plot_single_sweep(
        rows=rows,
        output_path=output_path,
        x_field_name=x_field_name,
        y_field_name=y_field_name,
        label_field_name="maximum_sequence_length",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        plot_title=plot_title,
    )

# Plot overlay comparison across multiple sequence-length sweep CSV files
def plot_sequence_length_sweep_comparison(
    csv_paths: List[Path],
    model_labels: List[str],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    plot_sweep_comparison(
        csv_paths=csv_paths,
        model_labels=model_labels,
        output_path=output_path,
        x_field_name=x_field_name,
        y_field_name=y_field_name,
        label_field_name="maximum_sequence_length",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        plot_title=plot_title,
    )

# Plot one precision sweep curve for a single model checkpoint
def plot_single_precision_sweep(
    rows: List[Dict[str, object]],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    plot_single_sweep(
        rows=rows,
        output_path=output_path,
        x_field_name=x_field_name,
        y_field_name=y_field_name,
        label_field_name="precision_mode",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        plot_title=plot_title,
    )

# Plot overlay comparison across multiple precision sweep CSV files
def plot_precision_sweep_comparison(
    csv_paths: List[Path],
    model_labels: List[str],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    plot_sweep_comparison(
        csv_paths=csv_paths,
        model_labels=model_labels,
        output_path=output_path,
        x_field_name=x_field_name,
        y_field_name=y_field_name,
        label_field_name="precision_mode",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        plot_title=plot_title,
    )