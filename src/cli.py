from __future__ import annotations
import argparse
import datetime as datetime_module
import math
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from transformers import TrainingArguments, DataCollatorWithPadding, set_seed
from src.data.squad_v2 import (
    load_raw_squad_v2_splits,
    prepare_squad_v2_evaluation_features,
    prepare_squad_v2_training_features,
    postprocess_squad_v2_predictions,
)
from src.energy.meter import EnergyMeter
from src.evaluation.squad_metrics import compute_squad_v2_metrics
from src.models.counting_trainer import CountingTrainer
from src.models.hf_qa import load_question_answering_model_and_tokenizer
from src.utils.io import (
    append_line_to_text_file,
    ensure_directory_exists,
    merge_dictionaries,
    read_yaml_file,
    write_json_file,
    write_yaml_file,
)
from src.utils.token_count import count_non_padding_tokens_in_feature_dataset

# Create timestamped run identifier for reproducible run folders
def create_timestamped_run_identifier(prefix: str) -> str:
    timestamp = datetime_module.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{prefix}"

# Build Hugging Face TrainingArguments from training configuration
def build_training_arguments(training_config: Dict[str, Any], run_directory: Path) -> TrainingArguments:
    # Place Hugging Face Trainer internal outputs under the run folder
    trainer_output_directory = str(run_directory / "huggingface_trainer")

    # Import here to inspect installed transformers version at runtime
    import inspect

    # Determine which evaluation keyword this transformers version supports
    training_arguments_signature = inspect.signature(TrainingArguments.__init__)
    supports_evaluation_strategy = "evaluation_strategy" in training_arguments_signature.parameters
    supports_eval_strategy = "eval_strategy" in training_arguments_signature.parameters

    # Build common TrainingArguments fields
    training_arguments_fields: Dict[str, Any] = {
        "output_dir": trainer_output_directory,
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "num_train_epochs": float(training_config["number_of_training_epochs"]),
        "per_device_train_batch_size": int(training_config["per_device_training_batch_size"]),
        "per_device_eval_batch_size": int(training_config["per_device_evaluation_batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "warmup_steps": int(training_config["warmup_steps"]),
        "logging_steps": int(training_config["logging_steps"]),
        "save_strategy": str(training_config["save_strategy"]),
        "save_steps": int(training_config.get("save_steps", 0)),
        "save_total_limit": int(training_config.get("save_total_limit", 1)),
        "save_only_model": bool(training_config.get("save_only_model", False)),
        "eval_steps": int(training_config["evaluation_steps"]),
        "fp16": bool(training_config["use_fp16"]),
        "bf16": bool(training_config["use_bf16"]),
        "dataloader_num_workers": int(training_config["dataloader_num_workers"]),
        "report_to": [],
        "seed": int(training_config["random_seed"]),
    }

    # Add the correct evaluation strategy key depending on transformers version
    if supports_evaluation_strategy:
        training_arguments_fields["evaluation_strategy"] = str(training_config["evaluation_strategy"])
    elif supports_eval_strategy:
        training_arguments_fields["eval_strategy"] = str(training_config["evaluation_strategy"])
    else:
        raise ValueError(
            "This transformers version supports neither 'evaluation_strategy' nor 'eval_strategy'. "
            "Please upgrade transformers."
        )

    return TrainingArguments(**training_arguments_fields)

# Run baseline training and evaluation for SQuAD v2 and record energy metrics
def run_squad_v2_baseline_training_and_evaluation(arguments: argparse.Namespace) -> None:
    dataset_config = read_yaml_file(arguments.dataset_config_path)
    model_config = read_yaml_file(arguments.model_config_path)
    training_config = read_yaml_file(arguments.training_config_path)

    # Create run directory and write resolved configuration for reproducibility
    run_identifier = arguments.run_identifier or create_timestamped_run_identifier("baseline_squad_v2")
    run_directory = ensure_directory_exists(Path("runs") / run_identifier)
    log_file_path = run_directory / "logs.txt"

    resolved_config = merge_dictionaries(
        {"run_identifier": run_identifier},
        {"dataset": dataset_config},
        {"model": model_config},
        {"training": training_config},
    )
    write_yaml_file(resolved_config, run_directory / "config_resolved.yaml")

    append_line_to_text_file(log_file_path, f"[run] {run_identifier}")
    append_line_to_text_file(
        log_file_path,
        f"[configs] dataset={arguments.dataset_config_path} model={arguments.model_config_path} training={arguments.training_config_path}",
    )

    # Set random seed for reproducibility
    set_seed(int(training_config["random_seed"]))

    # Load model and tokenizer for question answering
    qa_artifacts = load_question_answering_model_and_tokenizer(
        model_name_or_path=model_config["model_name_or_path"],
        tokenizer_name_or_path=model_config.get("tokenizer_name_or_path"),
    )
    tokenizer = qa_artifacts.tokenizer
    model = qa_artifacts.model

    # Load raw SQuAD v2 train/evaluation splits (optionally truncated for harness validation)
    raw_train_split, raw_evaluation_split = load_raw_squad_v2_splits(
        huggingface_dataset_id=dataset_config["huggingface_dataset_id"],
        train_split_name=dataset_config["train_split_name"],
        evaluation_split_name=dataset_config["evaluation_split_name"],
        maximum_train_examples=dataset_config.get("maximum_train_examples"),
        maximum_evaluation_examples=dataset_config.get("maximum_evaluation_examples"),
    )

    maximum_sequence_length = int(training_config["maximum_sequence_length"])
    document_stride = int(training_config["document_stride"])
    pad_to_maximum_length = bool(training_config["pad_to_maximum_length"])

    # Tokenize training split into model-ready features
    append_line_to_text_file(log_file_path, "[data] tokenizing training features")
    tokenized_training_features = raw_train_split.map(
        lambda examples: prepare_squad_v2_training_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=maximum_sequence_length,
            document_stride=document_stride,
            pad_to_maximum_length=pad_to_maximum_length,
        ),
        batched=True,
        remove_columns=raw_train_split.column_names,
        desc="Tokenizing SQuAD v2 train",
    )

    # Tokenize evaluation split into model-ready features with offset mappings
    append_line_to_text_file(log_file_path, "[data] tokenizing evaluation features")
    tokenized_evaluation_features = raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=maximum_sequence_length,
            document_stride=document_stride,
            pad_to_maximum_length=pad_to_maximum_length,
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc="Tokenizing SQuAD v2 eval",
    )

    # Keep full evaluation features for postprocessing (needs example_id and offset_mapping)
    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features

    # Remove non-tensor columns for Trainer evaluation/prediction to avoid collator issues
    tokenized_evaluation_features_for_trainer = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    # Use dynamic padding so batches can be stacked even when pad_to_maximum_length is false
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Build training arguments and Trainer
    import inspect
    training_arguments = build_training_arguments(training_config=training_config, run_directory=run_directory)
    trainer_init_signature = inspect.signature(CountingTrainer.__init__)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": tokenized_training_features,
        "eval_dataset": tokenized_evaluation_features_for_trainer,
        "data_collator": data_collator
    }

    # Pass tokenizer only if the installed transformers version supports it.
    if "tokenizer" in trainer_init_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    # Create the Trainer (with counting enabled)
    trainer = CountingTrainer(**trainer_kwargs)

    # Training energy measurement
    append_line_to_text_file(log_file_path, "[energy][train] starting energy meter")
    training_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(training_config["power_sampling_interval_seconds"])
    )
    training_energy_meter.start()

    # Execute training with Hugging Face Trainer
    append_line_to_text_file(log_file_path, "[train] calling trainer.train()")
    training_result = trainer.train()

    # Stop training energy measurement
    training_energy_meter.stop()
    append_line_to_text_file(log_file_path, "[energy][train] stopped energy meter")

    # Extract training counters for normalized energy reporting
    number_of_training_steps = int(trainer.training_counters.number_of_training_steps)
    number_of_training_examples = int(trainer.training_counters.number_of_training_examples)
    number_of_training_tokens = int(trainer.training_counters.number_of_training_tokens)
    training_energy_joules = float(training_energy_meter.get_energy_joules())

    # Compute normalized training energy metrics
    training_energy_report_additional_fields = {
        "phase": "training",
        "number_of_training_steps": number_of_training_steps,
        "number_of_training_examples": number_of_training_examples,
        "number_of_training_tokens": number_of_training_tokens,
        "joules_per_training_step": (training_energy_joules / number_of_training_steps) if number_of_training_steps > 0 else None,
        "joules_per_training_example": (training_energy_joules / number_of_training_examples) if number_of_training_examples > 0 else None,
        "joules_per_training_token": (training_energy_joules / number_of_training_tokens) if number_of_training_tokens > 0 else None,
    }
    training_energy_meter.save_report(
        path=run_directory / "energy_train.json",
        additional_fields=training_energy_report_additional_fields,
    )

    # Save Trainer training metrics for debugging
    write_json_file({"trainer_training_metrics": training_result.metrics}, run_directory / "trainer_training_metrics.json")

    # Inference energy measurement
    append_line_to_text_file(log_file_path, "[energy][inference] starting energy meter for trainer.predict()")
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(training_config["power_sampling_interval_seconds"])
    )
    inference_energy_meter.start()

    append_line_to_text_file(log_file_path, "[inference] calling trainer.predict(evaluation_features)")
    prediction_output = trainer.predict(tokenized_evaluation_features_for_trainer)

    inference_energy_meter.stop()
    append_line_to_text_file(log_file_path, "[energy][inference] stopped energy meter for trainer.predict()")

    inference_energy_joules = float(inference_energy_meter.get_energy_joules())

    # Compute normalization units for inference reporting
    per_device_evaluation_batch_size = int(training_config["per_device_evaluation_batch_size"])
    number_of_inference_examples = int(len(tokenized_evaluation_features))
    number_of_inference_batches = int(math.ceil(number_of_inference_examples / per_device_evaluation_batch_size))
    number_of_inference_tokens = int(count_non_padding_tokens_in_feature_dataset(tokenized_evaluation_features))

    inference_energy_report_additional_fields = {
        "phase": "inference",
        "number_of_inference_batches": number_of_inference_batches,
        "number_of_inference_examples": number_of_inference_examples,
        "number_of_inference_tokens": number_of_inference_tokens,
        "joules_per_inference_batch": (inference_energy_joules / number_of_inference_batches) if number_of_inference_batches > 0 else None,
        "joules_per_inference_example": (inference_energy_joules / number_of_inference_examples) if number_of_inference_examples > 0 else None,
        "joules_per_inference_token": (inference_energy_joules / number_of_inference_tokens) if number_of_inference_tokens > 0 else None,
    }
    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields=inference_energy_report_additional_fields,
    )

    # SQuAD v2 metric computation
    raw_predictions = prediction_output.predictions
    if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
        start_logits, end_logits = raw_predictions
    else:
        start_logits, end_logits = raw_predictions

    # Convert logits into final answer strings per example id
    append_line_to_text_file(log_file_path, "[evaluation] postprocessing logits into text predictions")
    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(np.array(start_logits), np.array(end_logits)),
        tokenizer=tokenizer,
        n_best_size=20,
        maximum_answer_length=30,
    )

    # Log basic statistics about no-answer probabilities to validate calibration.
    no_answer_probabilities = list(no_answer_probability_by_example_id.values())
    if len(no_answer_probabilities) > 0:
        append_line_to_text_file(
            log_file_path,
            f"[debug] no_answer_probability stats: "
            f"min={min(no_answer_probabilities):.6f}, "
            f"max={max(no_answer_probabilities):.6f}, "
            f"mean={sum(no_answer_probabilities)/len(no_answer_probabilities):.6f}"
        )
        append_line_to_text_file(
            log_file_path,
            f"[debug] no_answer_probability samples: {no_answer_probabilities[:10]}"
        )

    # Compute official SQuAD v2 metrics
    append_line_to_text_file(log_file_path, "[evaluation] computing SQuAD v2 metrics")
    metrics = compute_squad_v2_metrics(
        predictions_by_example_id=predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )
    write_json_file(metrics, run_directory / "metrics.json")
    append_line_to_text_file(log_file_path, f"[evaluation] metrics: {metrics}")

    # One-file summary for quick inspection
    summary = {
        "run_identifier": run_identifier,
        "dataset_id": dataset_config["huggingface_dataset_id"],
        "model_name_or_path": model_config["model_name_or_path"],
        "metrics": metrics,
        "training_energy": {
            "energy_joules": training_energy_joules,
            **training_energy_report_additional_fields,
        },
        "inference_energy": {
            "energy_joules": inference_energy_joules,
            **inference_energy_report_additional_fields,
        },
    }
    write_json_file(summary, run_directory / "summary.json")
    append_line_to_text_file(log_file_path, "[done] baseline harness run complete")

    print(f"Run complete: {run_directory}")
    print(f"Summary written to: {run_directory / 'summary.json'}")

# Build CLI argument parser for harness
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NLP Frontiers v2 - Harness (Hugging Face Trainer)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command for baseline SQuAD v2 training and evaluation
    baseline_parser = subparsers.add_parser(
        "train_squad_v2_baseline",
        help="Train and evaluate a QA model on SQuAD v2 while measuring training and inference energy.",
    )
    baseline_parser.add_argument("--dataset-config-path", required=True)
    baseline_parser.add_argument("--model-config-path", required=True)
    baseline_parser.add_argument("--training-config-path", required=True)
    baseline_parser.add_argument("--run-identifier", default=None)
    baseline_parser.set_defaults(handler=run_squad_v2_baseline_training_and_evaluation)

    return parser

# Entrypoint for running the CLI
def main() -> None:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    arguments.handler(arguments)

if __name__ == "__main__":
    main()