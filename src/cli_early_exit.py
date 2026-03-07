from __future__ import annotations
import argparse
import datetime as datetime_module
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, set_seed
from src.evaluation.early_exit_benchmark import (
    evaluate_early_exit_threshold_on_squad_v2,
    load_raw_squad_v2_splits,
    plot_single_early_exit_sweep,
    save_early_exit_rows_to_csv,
    train_early_exit_model_on_squad_v2,
)
from src.models.bert_early_exit import (
    BertForQuestionAnsweringEarlyExit,
    initialize_early_exit_model_from_base_checkpoint,
)
from src.utils.io import (
    append_line_to_text_file,
    ensure_directory_exists,
    read_yaml_file,
    write_json_file,
    write_yaml_file,
)

# Create timestamped run identifier
def create_timestamped_run_identifier(prefix: str) -> str:
    timestamp = datetime_module.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{prefix}"

# Train three-exit BERT QA model starting from existing BERT-base QA checkpoint
def train_squad_v2_bert_early_exit(arguments: argparse.Namespace) -> None:
    dataset_config = read_yaml_file(arguments.dataset_config_path)
    model_config = read_yaml_file(arguments.model_config_path)
    training_config = read_yaml_file(arguments.training_config_path)

    run_identifier = arguments.run_identifier or create_timestamped_run_identifier(
        "bert_early_exit_squad_v2"
    )
    run_directory = ensure_directory_exists(Path("runs") / run_identifier)
    log_file_path = run_directory / "logs.txt"

    resolved_configuration = {
        "run_identifier": run_identifier,
        "dataset": dataset_config,
        "model": model_config,
        "training": training_config,
        "base_checkpoint_path": arguments.base_checkpoint_path,
    }
    write_yaml_file(resolved_configuration, run_directory / "config_resolved.yaml")

    set_seed(int(training_config["random_seed"]))

    append_line_to_text_file(log_file_path, f"[run] {run_identifier}")
    append_line_to_text_file(log_file_path, f"[base_checkpoint] {arguments.base_checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("tokenizer_name_or_path", model_config["model_name_or_path"])
    )

    model = initialize_early_exit_model_from_base_checkpoint(
        base_checkpoint_path=arguments.base_checkpoint_path,
        early_exit_layers=list(training_config["early_exit_layers"]),
        early_exit_loss_weights=list(training_config["early_exit_loss_weights"]),
    )

    raw_train_split, raw_evaluation_split = load_raw_squad_v2_splits(
        huggingface_dataset_id=dataset_config["huggingface_dataset_id"],
        train_split_name=dataset_config["train_split_name"],
        evaluation_split_name=dataset_config["evaluation_split_name"],
        maximum_train_examples=dataset_config.get("maximum_train_examples"),
        maximum_evaluation_examples=dataset_config.get("maximum_evaluation_examples"),
    )

    training_result = train_early_exit_model_on_squad_v2(
        run_directory=run_directory,
        log_file_path=log_file_path,
        model=model,
        tokenizer=tokenizer,
        raw_train_split=raw_train_split,
        raw_evaluation_split=raw_evaluation_split,
        maximum_sequence_length=int(training_config["maximum_sequence_length"]),
        configured_document_stride=int(training_config["document_stride"]),
        pad_to_maximum_length=bool(training_config["pad_to_maximum_length"]),
        pad_to_multiple_of=training_config.get("pad_to_multiple_of"),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        number_of_training_epochs=float(training_config["number_of_training_epochs"]),
        per_device_training_batch_size=int(training_config["per_device_training_batch_size"]),
        per_device_evaluation_batch_size=int(training_config["per_device_evaluation_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        warmup_steps=int(training_config["warmup_steps"]),
        logging_steps=int(training_config["logging_steps"]),
        save_strategy=str(training_config["save_strategy"]),
        save_steps=int(training_config["save_steps"]),
        save_total_limit=int(training_config["save_total_limit"]),
        save_only_model=bool(training_config["save_only_model"]),
        evaluation_strategy=str(training_config["evaluation_strategy"]),
        evaluation_steps=int(training_config["evaluation_steps"]),
        dataloader_num_workers=int(training_config["dataloader_num_workers"]),
        power_sampling_interval_seconds=float(training_config["power_sampling_interval_seconds"]),
    )

    print(f"Run complete: {run_directory}")
    print(f"Saved early-exit model to: {run_directory / 'best_model'}")
    print(f"Training energy (J): {training_result['training_energy_joules']:.4f}")

# Run threshold sweep for true dynamic early-exit inference
def run_squad_v2_bert_early_exit_threshold_sweep(arguments: argparse.Namespace) -> None:
    dataset_config = read_yaml_file(arguments.dataset_config_path)
    model_config = read_yaml_file(arguments.model_config_path)
    inference_config = read_yaml_file(arguments.inference_config_path)

    run_identifier = arguments.run_identifier or create_timestamped_run_identifier(
        "bert_early_exit_threshold_sweep"
    )
    run_directory = ensure_directory_exists(Path("runs") / run_identifier)
    log_file_path = run_directory / "logs.txt"

    resolved_configuration = {
        "run_identifier": run_identifier,
        "dataset": dataset_config,
        "model": model_config,
        "inference": inference_config,
        "checkpoint_path": arguments.checkpoint_path,
    }
    write_yaml_file(resolved_configuration, run_directory / "config_resolved.yaml")

    append_line_to_text_file(log_file_path, f"[run] {run_identifier}")
    append_line_to_text_file(log_file_path, f"[checkpoint] {arguments.checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("tokenizer_name_or_path", model_config["model_name_or_path"])
    )
    model = BertForQuestionAnsweringEarlyExit.from_pretrained(arguments.checkpoint_path)

    # Fail fast if CUDA is unavailable because GPU energy measurement would be invalid.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            "The early-exit sweep must run on GPU for energy measurement to be meaningful."
        )

    # Move the model to CUDA and print the resolved device for verification.
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    print(f"[device] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"[device] using device={device}", flush=True)
    print(f"[device] model parameter device={next(model.parameters()).device}", flush=True)

    _, raw_evaluation_split = load_raw_squad_v2_splits(
        huggingface_dataset_id=dataset_config["huggingface_dataset_id"],
        train_split_name=dataset_config["train_split_name"],
        evaluation_split_name=dataset_config["evaluation_split_name"],
        maximum_train_examples=dataset_config.get("maximum_train_examples"),
        maximum_evaluation_examples=dataset_config.get("maximum_evaluation_examples"),
    )

    sweep_rows: List[Dict[str, object]] = []

    for early_exit_confidence_threshold in inference_config["early_exit_confidence_thresholds"]:
        sub_run_directory = ensure_directory_exists(
            run_directory / f"threshold_{str(early_exit_confidence_threshold).replace('.', '_')}"
        )

        append_line_to_text_file(
            log_file_path,
            f"[early_exit_threshold={float(early_exit_confidence_threshold):.2f}] starting evaluation",
        )

        evaluation_result = evaluate_early_exit_threshold_on_squad_v2(
            run_directory=sub_run_directory,
            log_file_path=log_file_path,
            model=model,
            tokenizer=tokenizer,
            raw_evaluation_split=raw_evaluation_split,
            maximum_sequence_length=int(inference_config["maximum_sequence_length"]),
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
            early_exit_confidence_threshold=float(early_exit_confidence_threshold),
            minimum_measurement_seconds=float(inference_config["minimum_measurement_seconds"]),
            maximum_measurement_passes=int(inference_config["maximum_measurement_passes"]),
        )

        metrics_thresholded = evaluation_result["metrics_thresholded"]

        sweep_rows.append(
            {
                "early_exit_confidence_threshold": float(
                    evaluation_result["early_exit_confidence_threshold"]
                ),
                "f1": float(metrics_thresholded["f1"]),
                "exact": float(metrics_thresholded["exact"]),
                "inference_energy_joules": float(evaluation_result["inference_energy_joules"]),
                "joules_per_inference_example": float(evaluation_result["joules_per_inference_example"]),
                "joules_per_inference_feature_window": float(
                    evaluation_result["joules_per_inference_feature_window"]
                ),
                "joules_per_inference_token": float(evaluation_result["joules_per_inference_token"]),
                "average_latency_per_raw_example_milliseconds": float(
                    evaluation_result["average_latency_per_raw_example_milliseconds"]
                ),
                "average_latency_per_feature_window_milliseconds": float(
                    evaluation_result["average_latency_per_feature_window_milliseconds"]
                ),
                "average_exited_layer": float(evaluation_result["average_exited_layer"]),
                "average_executed_layer_count": float(
                    evaluation_result["average_executed_layer_count"]
                ),
                "number_of_raw_evaluation_examples": int(
                    evaluation_result["number_of_raw_evaluation_examples"]
                ),
                "number_of_feature_windows": int(evaluation_result["number_of_feature_windows"]),
                "number_of_inference_tokens": int(evaluation_result["number_of_inference_tokens"]),
                "number_of_measurement_passes": int(evaluation_result["number_of_measurement_passes"]),
                "number_of_energy_samples": int(evaluation_result["number_of_energy_samples"]),
            }
        )

        append_line_to_text_file(
            log_file_path,
            f"[early_exit_threshold={float(early_exit_confidence_threshold):.2f}] done "
            f"f1={float(metrics_thresholded['f1']):.4f} "
            f"exact={float(metrics_thresholded['exact']):.4f} "
            f"energy_j={float(evaluation_result['inference_energy_joules']):.4f} "
            f"avg_exit_layer={float(evaluation_result['average_exited_layer']):.4f}",
        )

    write_json_file(sweep_rows, run_directory / "sweep_rows.json")
    save_early_exit_rows_to_csv(sweep_rows, run_directory / "sweep_rows.csv")

    # Plot F1 versus energy
    plot_single_early_exit_sweep(
        rows=sweep_rows,
        output_path=run_directory / "energy_vs_f1.png",
        x_field_name="inference_energy_joules",
        y_field_name="f1",
        x_axis_label="Inference Energy (J)",
        y_axis_label="F1",
        plot_title="Dynamic Early Exit Threshold Sweep: Energy vs F1",
    )

    # Plot latency versus F1
    plot_single_early_exit_sweep(
        rows=sweep_rows,
        output_path=run_directory / "latency_vs_f1.png",
        x_field_name="average_latency_per_raw_example_milliseconds",
        y_field_name="f1",
        x_axis_label="Average Latency per Raw Example (ms)",
        y_axis_label="F1",
        plot_title="Dynamic Early Exit Threshold Sweep: Latency vs F1",
    )

    # Plot average exit depth versus F1
    plot_single_early_exit_sweep(
        rows=sweep_rows,
        output_path=run_directory / "average_exit_layer_vs_f1.png",
        x_field_name="average_exited_layer",
        y_field_name="f1",
        x_axis_label="Average Exited Layer",
        y_axis_label="F1",
        plot_title="Dynamic Early Exit Threshold Sweep: Average Exit Layer vs F1",
    )

    print(f"Run complete: {run_directory}")
    print(f"Saved sweep rows to: {run_directory / 'sweep_rows.csv'}")

# Register CLI commands
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register training command
    train_parser = subparsers.add_parser("train_squad_v2_bert_early_exit")
    train_parser.add_argument("--dataset-config-path", type=str, required=True)
    train_parser.add_argument("--model-config-path", type=str, required=True)
    train_parser.add_argument("--training-config-path", type=str, required=True)
    train_parser.add_argument("--base-checkpoint-path", type=str, required=True)
    train_parser.add_argument("--run-identifier", type=str, default=None)
    train_parser.set_defaults(handler=train_squad_v2_bert_early_exit)

    # Register threshold sweep command
    sweep_parser = subparsers.add_parser("run_squad_v2_bert_early_exit_threshold_sweep")
    sweep_parser.add_argument("--dataset-config-path", type=str, required=True)
    sweep_parser.add_argument("--model-config-path", type=str, required=True)
    sweep_parser.add_argument("--inference-config-path", type=str, required=True)
    sweep_parser.add_argument("--checkpoint-path", type=str, required=True)
    sweep_parser.add_argument("--run-identifier", type=str, default=None)
    sweep_parser.set_defaults(handler=run_squad_v2_bert_early_exit_threshold_sweep)

    return parser

# Parse arguments and dispatch to selected handler
def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()