from __future__ import annotations
import inspect
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase, TrainingArguments
from src.data.squad_v2 import (
    postprocess_squad_v2_predictions,
    prepare_squad_v2_evaluation_features,
    prepare_squad_v2_training_features,
)
from src.energy.meter import EnergyMeter
from src.evaluation.squad_metrics import compute_squad_v2_metrics
from src.models.counting_trainer import CountingTrainer
from src.models.bert_early_exit import BertForQuestionAnsweringEarlyExit
from src.utils.io import append_line_to_text_file, write_json_file
from src.utils.token_count import count_non_padding_tokens_in_feature_dataset

# Load raw SQuAD v2 train/eval splits
def load_raw_squad_v2_splits(
    huggingface_dataset_id: str,
    train_split_name: str,
    evaluation_split_name: str,
    maximum_train_examples: Optional[int] = None,
    maximum_evaluation_examples: Optional[int] = None,
):
    raw_train_split = load_dataset(huggingface_dataset_id, split=train_split_name)
    raw_evaluation_split = load_dataset(huggingface_dataset_id, split=evaluation_split_name)

    if maximum_train_examples is not None:
        raw_train_split = raw_train_split.select(range(int(maximum_train_examples)))

    if maximum_evaluation_examples is not None:
        raw_evaluation_split = raw_evaluation_split.select(range(int(maximum_evaluation_examples)))

    return raw_train_split, raw_evaluation_split

# Clamp document stride for shorter sequence lengths
def clamp_document_stride_for_sequence_length(
    configured_document_stride: int,
    maximum_sequence_length: int,
) -> int:
    return int(min(int(configured_document_stride), max(1, int(maximum_sequence_length) // 2)))

# Tokenize SQuAD v2 training features
def tokenize_squad_v2_training_features(
    raw_train_split,
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    document_stride: int,
    pad_to_maximum_length: bool,
):
    return raw_train_split.map(
        lambda examples: prepare_squad_v2_training_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_train_split.column_names,
        desc=f"Tokenizing SQuAD v2 train at max_sequence_length={maximum_sequence_length}",
    )

# Tokenize SQuAD v2 evaluation features
def tokenize_squad_v2_evaluation_features(
    raw_evaluation_split,
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    document_stride: int,
    pad_to_maximum_length: bool,
):
    return raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc=f"Tokenizing SQuAD v2 eval at max_sequence_length={maximum_sequence_length}",
    )

# Build dynamic-padded dataloader
def build_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    per_device_batch_size: int,
    dataloader_num_workers: int,
    pad_to_multiple_of: Optional[int],
) -> DataLoader:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    return DataLoader(
        dataset,
        batch_size=int(per_device_batch_size),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=int(dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
    )

# Compute normalized entropy-based confidence for one exit
def compute_entropy_based_exit_confidence(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
) -> torch.Tensor:
    start_probabilities = torch.softmax(start_logits, dim=-1)
    end_probabilities = torch.softmax(end_logits, dim=-1)

    start_log_probabilities = torch.log(start_probabilities.clamp_min(1e-12))
    end_log_probabilities = torch.log(end_probabilities.clamp_min(1e-12))

    start_entropy = -(start_probabilities * start_log_probabilities).sum(dim=-1)
    end_entropy = -(end_probabilities * end_log_probabilities).sum(dim=-1)

    sequence_length = start_logits.size(-1)
    normalization_denominator = math.log(max(2, int(sequence_length)))

    normalized_start_entropy = start_entropy / normalization_denominator
    normalized_end_entropy = end_entropy / normalization_denominator

    confidence = 1.0 - 0.5 * (normalized_start_entropy + normalized_end_entropy)
    return confidence

# Select earliest exit (where confidence exceeds the threshold)
def select_exit_index_per_example(
    exit_start_logits: Tuple[torch.Tensor, ...],
    exit_end_logits: Tuple[torch.Tensor, ...],
    exit_layers: List[int],
    confidence_threshold: float,
) -> torch.Tensor:
    batch_size = exit_start_logits[0].size(0)
    selected_exit_indices = torch.full(
        (batch_size,),
        fill_value=len(exit_layers) - 1,
        dtype=torch.long,
        device=exit_start_logits[0].device,
    )

    has_exited = torch.zeros(batch_size, dtype=torch.bool, device=exit_start_logits[0].device)

    for exit_index, (start_logits, end_logits) in enumerate(zip(exit_start_logits, exit_end_logits)):
        exit_confidence = compute_entropy_based_exit_confidence(
            start_logits=start_logits,
            end_logits=end_logits,
        )

        should_exit_here = (exit_confidence >= float(confidence_threshold)) & (~has_exited)

        selected_exit_indices = torch.where(
            should_exit_here,
            torch.full_like(selected_exit_indices, fill_value=exit_index),
            selected_exit_indices,
        )
        has_exited = has_exited | should_exit_here

    return selected_exit_indices

# Gather per-example logits from selected exits
def gather_selected_exit_logits(
    exit_logits: Tuple[torch.Tensor, ...],
    selected_exit_indices: torch.Tensor,
) -> torch.Tensor:
    stacked_exit_logits = torch.stack(exit_logits, dim=0)
    batch_size = selected_exit_indices.size(0)
    batch_indices = torch.arange(batch_size, device=selected_exit_indices.device)
    return stacked_exit_logits[selected_exit_indices, batch_indices, :]

# Run short warmup pass for early-exit inference
def run_early_exit_inference_warmup(
    model: BertForQuestionAnsweringEarlyExit,
    dataloader: DataLoader,
    device: torch.device,
    number_of_warmup_batches: int,
) -> None:
    if int(number_of_warmup_batches) <= 0:
        return

    model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= int(number_of_warmup_batches):
                break

            batch_on_device = {name: tensor.to(device) for name, tensor in batch.items()}
            _ = model(**batch_on_device, return_dict=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

# Train early-exit model from base checkpoint
def train_early_exit_model_on_squad_v2(
    run_directory: Path,
    log_file_path: Path,
    model: BertForQuestionAnsweringEarlyExit,
    tokenizer: PreTrainedTokenizerBase,
    raw_train_split,
    raw_evaluation_split,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
    pad_to_multiple_of: Optional[int],
    learning_rate: float,
    weight_decay: float,
    number_of_training_epochs: float,
    per_device_training_batch_size: int,
    per_device_evaluation_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    logging_steps: int,
    save_strategy: str,
    save_steps: int,
    save_total_limit: int,
    save_only_model: bool,
    evaluation_strategy: str,
    evaluation_steps: int,
    dataloader_num_workers: int,
    power_sampling_interval_seconds: float,
) -> Dict[str, Any]:
    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    append_line_to_text_file(log_file_path, "[data] tokenizing early-exit training features")
    tokenized_training_features = tokenize_squad_v2_training_features(
        raw_train_split=raw_train_split,
        tokenizer=tokenizer,
        maximum_sequence_length=int(maximum_sequence_length),
        document_stride=int(effective_document_stride),
        pad_to_maximum_length=bool(pad_to_maximum_length),
    )

    append_line_to_text_file(log_file_path, "[data] tokenizing early-exit evaluation features")
    tokenized_evaluation_features = tokenize_squad_v2_evaluation_features(
        raw_evaluation_split=raw_evaluation_split,
        tokenizer=tokenizer,
        maximum_sequence_length=int(maximum_sequence_length),
        document_stride=int(effective_document_stride),
        pad_to_maximum_length=bool(pad_to_maximum_length),
    )

    tokenized_evaluation_features_for_trainer = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    training_arguments = TrainingArguments(
        output_dir=str(run_directory / "huggingface_trainer"),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        num_train_epochs=float(number_of_training_epochs),
        per_device_train_batch_size=int(per_device_training_batch_size),
        per_device_eval_batch_size=int(per_device_evaluation_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        warmup_steps=int(warmup_steps),
        logging_steps=int(logging_steps),
        save_strategy=str(save_strategy),
        save_steps=int(save_steps),
        save_total_limit=int(save_total_limit),
        evaluation_strategy=str(evaluation_strategy),
        eval_steps=int(evaluation_steps),
        dataloader_num_workers=int(dataloader_num_workers),
        bf16=False,
        fp16=False,
        report_to=[],
    )

    trainer_init_signature = inspect.signature(CountingTrainer.__init__)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": tokenized_training_features,
        "eval_dataset": tokenized_evaluation_features_for_trainer,
        "data_collator": data_collator,
    }
    if "tokenizer" in trainer_init_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = CountingTrainer(**trainer_kwargs)

    append_line_to_text_file(log_file_path, "[energy][train] starting energy meter")
    training_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    training_energy_meter.start()

    append_line_to_text_file(log_file_path, "[train] calling trainer.train() for early-exit model")
    train_output = trainer.train()

    training_energy_meter.stop()
    append_line_to_text_file(log_file_path, "[energy][train] stopped energy meter")

    trainer.save_model(run_directory / "best_model")

    training_energy_joules = float(training_energy_meter.get_energy_joules())
    number_of_training_examples = int(len(raw_train_split))
    number_of_training_samples_seen = int(number_of_training_examples * float(number_of_training_epochs))

    result = {
        "training_energy_joules": training_energy_joules,
        "number_of_training_examples": number_of_training_examples,
        "number_of_training_samples_seen": number_of_training_samples_seen,
        "joules_per_training_example": (
            training_energy_joules / number_of_training_samples_seen
            if number_of_training_samples_seen > 0
            else None
        ),
        "train_runtime_seconds": float(train_output.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_output.metrics.get("train_samples_per_second", 0.0)),
        "train_steps_per_second": float(train_output.metrics.get("train_steps_per_second", 0.0)),
    }

    write_json_file(result, run_directory / "training_summary.json")

    training_energy_meter.save_report(
        path=run_directory / "energy_train.json",
        additional_fields={
            "phase": "training",
            "number_of_training_examples": number_of_training_examples,
            "number_of_training_samples_seen": number_of_training_samples_seen,
            "joules_per_training_example": result["joules_per_training_example"],
        },
    )

    return result

# Evaluate confidence threshold for dynamic early-exit inference
def evaluate_early_exit_threshold_on_squad_v2(
    run_directory: Path,
    log_file_path: Path,
    model: BertForQuestionAnsweringEarlyExit,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
    pad_to_multiple_of: Optional[int],
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
    early_exit_confidence_threshold: float,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    append_line_to_text_file(
        log_file_path,
        f"[early_exit_threshold={early_exit_confidence_threshold:.2f}] effective_document_stride={effective_document_stride}",
    )

    tokenized_evaluation_features = tokenize_squad_v2_evaluation_features(
        raw_evaluation_split=raw_evaluation_split,
        tokenizer=tokenizer,
        maximum_sequence_length=int(maximum_sequence_length),
        document_stride=int(effective_document_stride),
        pad_to_maximum_length=bool(pad_to_maximum_length),
    )

    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features
    tokenized_evaluation_features_for_model = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    evaluation_dataloader = build_dataloader(
        dataset=tokenized_evaluation_features_for_model,
        tokenizer=tokenizer,
        per_device_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
    )

    append_line_to_text_file(
        log_file_path,
        f"[early_exit_threshold={early_exit_confidence_threshold:.2f}] running warmup batches={number_of_warmup_batches}",
    )
    run_early_exit_inference_warmup(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
        number_of_warmup_batches=int(number_of_warmup_batches),
    )

    append_line_to_text_file(
        log_file_path,
        f"[early_exit_threshold={early_exit_confidence_threshold:.2f}] starting inference energy meter",
    )
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    inference_energy_meter.start()

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start_time_seconds = time.perf_counter()

    selected_start_logits_batches = []
    selected_end_logits_batches = []
    exited_layer_values = []

    model.eval()
    with torch.inference_mode():
        for batch in evaluation_dataloader:
            batch_on_device = {name: tensor.to(device) for name, tensor in batch.items()}

            model_output = model(**batch_on_device, return_dict=True)

            selected_exit_indices = select_exit_index_per_example(
                exit_start_logits=model_output.exit_start_logits,
                exit_end_logits=model_output.exit_end_logits,
                exit_layers=model.early_exit_layers,
                confidence_threshold=float(early_exit_confidence_threshold),
            )

            selected_start_logits = gather_selected_exit_logits(
                exit_logits=model_output.exit_start_logits,
                selected_exit_indices=selected_exit_indices,
            )
            selected_end_logits = gather_selected_exit_logits(
                exit_logits=model_output.exit_end_logits,
                selected_exit_indices=selected_exit_indices,
            )

            selected_start_logits_batches.append(selected_start_logits.cpu().numpy())
            selected_end_logits_batches.append(selected_end_logits.cpu().numpy())

            for selected_exit_index in selected_exit_indices.cpu().tolist():
                exited_layer_values.append(int(model.early_exit_layers[int(selected_exit_index)]))

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_end_time_seconds = time.perf_counter()

    inference_energy_meter.stop()
    append_line_to_text_file(
        log_file_path,
        f"[early_exit_threshold={early_exit_confidence_threshold:.2f}] stopped inference energy meter",
    )

    start_logits = np.concatenate(selected_start_logits_batches, axis=0)
    end_logits = np.concatenate(selected_end_logits_batches, axis=0)

    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
    )

    thresholded_predictions_by_example_id: Dict[str, str] = {}
    for example_id, prediction_text in predictions_by_example_id.items():
        no_answer_probability = float(no_answer_probability_by_example_id.get(example_id, 0.0))
        thresholded_predictions_by_example_id[example_id] = (
            ""
            if no_answer_probability >= float(no_answer_probability_threshold)
            else prediction_text
        )

    raw_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )
    thresholded_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=thresholded_predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )

    inference_runtime_seconds = float(inference_end_time_seconds - inference_start_time_seconds)
    inference_energy_joules = float(inference_energy_meter.get_energy_joules())
    number_of_energy_samples = int(len(inference_energy_meter.samples))

    number_of_raw_evaluation_examples = int(len(raw_evaluation_split))
    number_of_feature_windows = int(len(tokenized_evaluation_features_for_model))
    number_of_inference_tokens = int(
        count_non_padding_tokens_in_feature_dataset(tokenized_evaluation_features_for_model)
    )

    average_latency_per_raw_example_milliseconds = (
        (inference_runtime_seconds / number_of_raw_evaluation_examples) * 1000.0
        if number_of_raw_evaluation_examples > 0
        else None
    )

    exit_layer_histogram: Dict[str, int] = {}
    for exit_layer in model.early_exit_layers:
        exit_layer_histogram[str(int(exit_layer))] = int(sum(1 for value in exited_layer_values if value == int(exit_layer)))

    average_exited_layer = (
        float(sum(exited_layer_values)) / float(len(exited_layer_values))
        if len(exited_layer_values) > 0
        else float(model.early_exit_layers[-1])
    )

    result = {
        "early_exit_confidence_threshold": float(early_exit_confidence_threshold),
        "maximum_sequence_length": int(maximum_sequence_length),
        "effective_document_stride": int(effective_document_stride),
        "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
        "number_of_feature_windows": number_of_feature_windows,
        "number_of_inference_tokens": number_of_inference_tokens,
        "inference_runtime_seconds": inference_runtime_seconds,
        "average_latency_per_raw_example_milliseconds": average_latency_per_raw_example_milliseconds,
        "inference_energy_joules": inference_energy_joules,
        "number_of_energy_samples": number_of_energy_samples,
        "joules_per_inference_example": (
            inference_energy_joules / number_of_raw_evaluation_examples
            if number_of_raw_evaluation_examples > 0
            else None
        ),
        "joules_per_inference_token": (
            inference_energy_joules / number_of_inference_tokens
            if number_of_inference_tokens > 0
            else None
        ),
        "metrics_raw": raw_metrics,
        "metrics_thresholded": thresholded_metrics,
        "average_exited_layer": average_exited_layer,
        "exit_layer_histogram": exit_layer_histogram,
    }

    write_json_file(result, run_directory / "result.json")

    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields={
            "phase": "inference",
            "early_exit_confidence_threshold": float(early_exit_confidence_threshold),
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
            "number_of_feature_windows": number_of_feature_windows,
            "number_of_inference_tokens": number_of_inference_tokens,
            "joules_per_inference_example": result["joules_per_inference_example"],
            "joules_per_inference_token": result["joules_per_inference_token"],
            "average_exited_layer": average_exited_layer,
        },
    )

    return result

# Save sweep rows to CSV
def save_early_exit_rows_to_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    field_names = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# Plot one early-exit sweep
def plot_single_early_exit_sweep(
    rows: List[Dict[str, object]],
    output_path: Path,
    x_field_name: str,
    y_field_name: str,
    x_axis_label: str,
    y_axis_label: str,
    plot_title: str,
) -> None:
    plot_rows = [
        row for row in rows
        if row.get(x_field_name) is not None and row.get(y_field_name) is not None
    ]

    x_values = [float(row[x_field_name]) for row in plot_rows]
    y_values = [float(row[y_field_name]) for row in plot_rows]
    labels = [f"{float(row['early_exit_confidence_threshold']):.2f}" for row in plot_rows]

    figure = plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker="o")

    for x_value, y_value, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points")

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)