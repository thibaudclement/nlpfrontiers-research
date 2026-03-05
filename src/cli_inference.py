from __future__ import annotations
import argparse
import datetime as datetime_module
import re
from pathlib import Path
from typing import Any, Dict, List
import torch
from src.evaluation.inference_benchmark import (
    evaluate_checkpoint_on_squad_v2_at_sequence_length,
    load_raw_squad_v2_evaluation_split,
)
from src.evaluation.pareto_plots import (
    load_sequence_length_sweep_rows_from_csv,
    plot_sequence_length_sweep_comparison,
    plot_single_sequence_length_sweep,
    save_sequence_length_sweep_rows_to_csv,
)
from src.models.hf_qa import load_question_answering_model_and_tokenizer
from src.utils.io import (
    append_line_to_text_file,
    ensure_directory_exists,
    read_yaml_file,
    write_json_file,
    write_yaml_file,
)

# Create a timestamped run identifier for reproducible run folders
def create_timestamped_run_identifier(prefix: str) -> str:
    timestamp = datetime_module.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{prefix}"

# Sanitize model label to use in file and folder names
def sanitize_label_for_file_name(label: str) -> str:
    sanitized_label = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(label).strip().lower())
    return sanitized_label.strip("_")

# Run max sequence length sweep for one checkpoint
def run_squad_v2_sequence_length_sweep(arguments: argparse.Namespace) -> None:
    dataset_config = read_yaml_file(arguments.dataset_config_path)
    model_config = read_yaml_file(arguments.model_config_path)
    inference_config = read_yaml_file(arguments.inference_config_path)

    model_label = str(arguments.model_label)
    sanitized_model_label = sanitize_label_for_file_name(label=model_label)

    run_identifier = arguments.run_identifier or create_timestamped_run_identifier(
        f"sequence_length_sweep_{sanitized_model_label}"
    )
    run_directory = ensure_directory_exists(Path("runs") / run_identifier)
    log_file_path = run_directory / "logs.txt"

    # Write resolved configuration for reproducibility
    resolved_configuration = {
        "run_identifier": run_identifier,
        "dataset": dataset_config,
        "model": model_config,
        "inference": inference_config,
        "checkpoint_path": arguments.checkpoint_path,
        "model_label": model_label,
    }
    write_yaml_file(resolved_configuration, run_directory / "config_resolved.yaml")

    append_line_to_text_file(log_file_path, f"[run] {run_identifier}")
    append_line_to_text_file(log_file_path, f"[model_label] {model_label}")
    append_line_to_text_file(log_file_path, f"[checkpoint] {arguments.checkpoint_path}")

    # Load tokenizer and model once for the full sweep
    qa_artifacts = load_question_answering_model_and_tokenizer(
        model_name_or_path=arguments.checkpoint_path,
        tokenizer_name_or_path=model_config.get("tokenizer_name_or_path", model_config["model_name_or_path"]),
    )
    tokenizer = qa_artifacts.tokenizer
    model = qa_artifacts.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load raw evaluation split once and reuse it across all sequence lengths
    raw_evaluation_split = load_raw_squad_v2_evaluation_split(
        huggingface_dataset_id=dataset_config["huggingface_dataset_id"],
        evaluation_split_name=dataset_config["evaluation_split_name"],
        maximum_evaluation_examples=dataset_config.get("maximum_evaluation_examples"),
    )

    sweep_rows: List[Dict[str, object]] = []

    # Evaluate checkpoint at each sequence length
    for maximum_sequence_length in inference_config["sequence_lengths"]:
        sub_run_directory = ensure_directory_exists(
            run_directory / f"max_sequence_length_{int(maximum_sequence_length)}"
        )

        append_line_to_text_file(
            log_file_path,
            f"[sequence_length={maximum_sequence_length}] starting evaluation",
        )

        evaluation_result = evaluate_checkpoint_on_squad_v2_at_sequence_length(
            run_directory=sub_run_directory,
            log_file_path=log_file_path,
            model=model,
            tokenizer=tokenizer,
            raw_evaluation_split=raw_evaluation_split,
            maximum_sequence_length=int(maximum_sequence_length),
            configured_document_stride=int(inference_config["document_stride"]),
            pad_to_maximum_length=bool(inference_config["pad_to_maximum_length"]),
            pad_to_multiple_of=inference_config.get("pad_to_multiple_of"),
            per_device_evaluation_batch_size=int(inference_config["per_device_evaluation_batch_size"]),
            dataloader_num_workers=int(inference_config["dataloader_num_workers"]),
            number_of_warmup_batches=int(inference_config["number_of_warmup_batches"]),
            power_sampling_interval_seconds=float(inference_config["power_sampling_interval_seconds"]),
            n_best_size=int(inference_config["n_best_size"]),
            maximum_answer_length=int(inference_config["maximum_answer_length"]),
            no_answer_probability_threshold=float(inference_config["no_answer_probability_threshold"]),
        )

        metrics_thresholded = evaluation_result["metrics_thresholded"]

        sweep_rows.append(
            {
                "model_label": model_label,
                "checkpoint_path": arguments.checkpoint_path,
                "maximum_sequence_length": int(maximum_sequence_length),
                "effective_document_stride": int(evaluation_result["effective_document_stride"]),
                "exact": float(metrics_thresholded["exact"]),
                "f1": float(metrics_thresholded["f1"]),
                "inference_energy_joules": float(evaluation_result["inference_energy_joules"]),
                "joules_per_inference_example": float(evaluation_result["joules_per_inference_example"]),
                "joules_per_feature_window": float(evaluation_result["joules_per_feature_window"]),
                "joules_per_inference_token": float(evaluation_result["joules_per_inference_token"]),
                "joules_per_exact_match_correct_example": float(
                    evaluation_result["joules_per_exact_match_correct_example"]
                ),
                "average_latency_per_raw_example_milliseconds": float(
                    evaluation_result["average_latency_per_raw_example_milliseconds"]
                ),
                "average_latency_per_feature_window_milliseconds": float(
                    evaluation_result["average_latency_per_feature_window_milliseconds"]
                ),
                "number_of_raw_evaluation_examples": int(evaluation_result["number_of_raw_evaluation_examples"]),
                "number_of_feature_windows": int(evaluation_result["number_of_feature_windows"]),
                "number_of_inference_tokens": int(evaluation_result["number_of_inference_tokens"]),
            }
        )

        append_line_to_text_file(
            log_file_path,
            f"[sequence_length={maximum_sequence_length}] done "
            f"f1={metrics_thresholded['f1']:.4f} "
            f"exact={metrics_thresholded['exact']:.4f} "
            f"joules_per_inference_token={evaluation_result['joules_per_inference_token']:.8f}",
        )

    # Save sweep table
    pareto_csv_path = run_directory / "pareto.csv"
    save_sequence_length_sweep_rows_to_csv(rows=sweep_rows, csv_path=pareto_csv_path)

    # Save summary JSON
    write_json_file(
        {
            "run_identifier": run_identifier,
            "model_label": model_label,
            "checkpoint_path": arguments.checkpoint_path,
            "pareto_csv_path": str(pareto_csv_path),
            "rows": sweep_rows,
        },
        run_directory / "summary.json",
    )

    # Plot per-model sweep curves
    plot_single_sequence_length_sweep(
        rows=sweep_rows,
        output_path=run_directory / "energy_token_vs_f1.png",
        x_field_name="joules_per_inference_token",
        y_field_name="f1",
        x_axis_label="Energy (J / Token)",
        y_axis_label="F1",
        plot_title=f"Max Sequence Length Sweep - {model_label}",
    )

    plot_single_sequence_length_sweep(
        rows=sweep_rows,
        output_path=run_directory / "energy_example_vs_f1.png",
        x_field_name="joules_per_inference_example",
        y_field_name="f1",
        x_axis_label="Energy (J / Example)",
        y_axis_label="F1",
        plot_title=f"Max Sequence Length Sweep - {model_label}",
    )

    plot_single_sequence_length_sweep(
        rows=sweep_rows,
        output_path=run_directory / "latency_vs_f1.png",
        x_field_name="average_latency_per_raw_example_milliseconds",
        y_field_name="f1",
        x_axis_label="Latency (ms / Example)",
        y_axis_label="F1",
        plot_title=f"Max Sequence Length Sweep - {model_label}",
    )

    print(f"Run complete: {run_directory}")
    print(f"Pareto table written to: {pareto_csv_path}")

# Plot an overlay comparison across several already-generated sweep CSV files
def plot_squad_v2_sequence_length_sweep_comparison(arguments: argparse.Namespace) -> None:
    output_path = Path(arguments.output_path)

    plot_sequence_length_sweep_comparison(
        csv_paths=[Path(path) for path in arguments.pareto_csv_paths],
        model_labels=list(arguments.model_labels),
        output_path=output_path,
        x_field_name=str(arguments.x_field_name),
        y_field_name=str(arguments.y_field_name),
        x_axis_label=str(arguments.x_axis_label),
        y_axis_label=str(arguments.y_axis_label),
        plot_title=str(arguments.plot_title),
    )

    print(f"Comparison plot written to: {output_path}")


# Build CLI argument parser
def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="NLP Frontiers - Inference-time sweep CLI")
    subparsers = argument_parser.add_subparsers(dest="command", required=True)

    # Create command for one max sequence length sweep
    sweep_parser = subparsers.add_parser(
        "run_squad_v2_sequence_length_sweep",
        help="Run a max sequence length inference sweep for one checkpoint on SQuAD v2.",
    )
    sweep_parser.add_argument("--dataset-config-path", required=True)
    sweep_parser.add_argument("--model-config-path", required=True)
    sweep_parser.add_argument("--inference-config-path", required=True)
    sweep_parser.add_argument("--checkpoint-path", required=True)
    sweep_parser.add_argument("--model-label", required=True)
    sweep_parser.add_argument("--run-identifier", default=None)
    sweep_parser.set_defaults(handler=run_squad_v2_sequence_length_sweep)

    # Create command for overlay comparison plots
    comparison_parser = subparsers.add_parser(
        "plot_squad_v2_sequence_length_sweep_comparison",
        help="Overlay several sequence length sweep CSV files in one comparison figure.",
    )
    comparison_parser.add_argument("--pareto-csv-paths", nargs="+", required=True)
    comparison_parser.add_argument("--model-labels", nargs="+", required=True)
    comparison_parser.add_argument("--output-path", required=True)
    comparison_parser.add_argument("--x-field-name", default="joules_per_inference_token")
    comparison_parser.add_argument("--y-field-name", default="f1")
    comparison_parser.add_argument("--x-axis-label", default="Energy (J / Token)")
    comparison_parser.add_argument("--y-axis-label", default="F1")
    comparison_parser.add_argument("--plot-title", default="Max Sequence Length Sweep Comparison")
    comparison_parser.set_defaults(handler=plot_squad_v2_sequence_length_sweep_comparison)

    return argument_parser

# Run CLI
def main() -> None:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    arguments.handler(arguments)

if __name__ == "__main__":
    main()