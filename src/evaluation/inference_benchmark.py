from __future__ import annotations
import inspect
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase
from src.data.squad_v2 import prepare_squad_v2_evaluation_features, postprocess_squad_v2_predictions
from src.energy.meter import EnergyMeter
from src.evaluation.squad_metrics import compute_squad_v2_metrics
from src.utils.io import append_line_to_text_file, write_json_file
from src.utils.token_count import count_non_padding_tokens_in_feature_dataset

# Load the raw SQuAD v2 evaluation split only
def load_raw_squad_v2_evaluation_split(
    huggingface_dataset_id: str,
    evaluation_split_name: str,
    maximum_evaluation_examples: Optional[int] = None,
):
    raw_evaluation_split = load_dataset(huggingface_dataset_id, split=evaluation_split_name)

    # Optionally truncate the evaluation split for fast debugging
    if maximum_evaluation_examples is not None:
        raw_evaluation_split = raw_evaluation_split.select(range(int(maximum_evaluation_examples)))

    return raw_evaluation_split

# Clamp document stride for shorter sequence lengths to avoid tokenizer errors
def clamp_document_stride_for_sequence_length(configured_document_stride: int, maximum_sequence_length: int) -> int:
    return int(min(int(configured_document_stride), max(1, int(maximum_sequence_length) // 2)))

# Filter batch fields so they match the model forward signature
def filter_batch_for_model_forward(model: PreTrainedModel, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    supported_argument_names = set(inspect.signature(model.forward).parameters.keys())
    return {name: value for name, value in batch.items() if name in supported_argument_names}

# Build evaluation DataLoader with dynamic padding
def build_evaluation_dataloader(
    tokenized_evaluation_features_for_model,
    tokenizer: PreTrainedTokenizerBase,
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    pad_to_multiple_of: Optional[int],
) -> DataLoader:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    return DataLoader(
        tokenized_evaluation_features_for_model,
        batch_size=int(per_device_evaluation_batch_size),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=int(dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
    )

# Run short warmup pass before measured inference
def run_inference_warmup(
    model: PreTrainedModel,
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
            batch_for_model = filter_batch_for_model_forward(model=model, batch=batch_on_device)
            _ = model(**batch_for_model)

    if device.type == "cuda":
        torch.cuda.synchronize()

# Run measured inference and collect logits
def run_measured_question_answering_inference(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    start_logits_batches = []
    end_logits_batches = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start_time_seconds = time.perf_counter()

    with torch.inference_mode():
        for batch in dataloader:
            batch_on_device = {name: tensor.to(device) for name, tensor in batch.items()}
            batch_for_model = filter_batch_for_model_forward(model=model, batch=batch_on_device)

            outputs = model(**batch_for_model)

            start_logits_batches.append(outputs.start_logits.detach().cpu().numpy())
            end_logits_batches.append(outputs.end_logits.detach().cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_end_time_seconds = time.perf_counter()

    all_start_logits = np.concatenate(start_logits_batches, axis=0)
    all_end_logits = np.concatenate(end_logits_batches, axis=0)

    return {
        "start_logits": all_start_logits,
        "end_logits": all_end_logits,
        "inference_runtime_seconds": float(inference_end_time_seconds - inference_start_time_seconds),
    }

# Evaluate one checkpoint on SQuAD v2 at specific inference max sequence length
def evaluate_checkpoint_on_squad_v2_at_sequence_length(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
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
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    # Clamp document stride for shorter sequence lengths
    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] effective_document_stride={effective_document_stride}",
    )

    # Tokenize evaluation features for this specific sequence length
    tokenized_evaluation_features = raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(effective_document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc=f"Tokenizing SQuAD v2 eval at max_sequence_length={maximum_sequence_length}",
    )

    # Keep full feature set for postprocessing
    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features

    # Remove non-tensor columns for model batching
    tokenized_evaluation_features_for_model = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    # Build the measured dataloader
    evaluation_dataloader = build_evaluation_dataloader(
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        tokenizer=tokenizer,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Run short warmup pass
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] running warmup batches={number_of_warmup_batches}",
    )
    run_inference_warmup(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
        number_of_warmup_batches=int(number_of_warmup_batches),
    )

    # Measure energy during full evaluation pass
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] starting inference energy meter",
    )
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    inference_energy_meter.start()

    measured_inference_output = run_measured_question_answering_inference(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
    )

    inference_energy_meter.stop()
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] stopped inference energy meter",
    )

    start_logits = measured_inference_output["start_logits"]
    end_logits = measured_inference_output["end_logits"]
    inference_runtime_seconds = float(measured_inference_output["inference_runtime_seconds"])
    inference_energy_joules = float(inference_energy_meter.get_energy_joules())

    # Postprocess logits into text predictions and no-answer probabilities
    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
    )

    # Apply fixed threshold to emit explicit empty-string predictions
    thresholded_predictions_by_example_id: Dict[str, str] = {}
    for example_id, prediction_text in predictions_by_example_id.items():
        no_answer_probability = float(no_answer_probability_by_example_id.get(example_id, 0.0))
        thresholded_predictions_by_example_id[example_id] = (
            ""
            if no_answer_probability >= float(no_answer_probability_threshold)
            else prediction_text
        )

    # Compute raw and thresholded SQuAD v2 metrics
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

    # Count raw examples, feature windows, and tokens for normalized reporting
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
    average_latency_per_feature_window_milliseconds = (
        (inference_runtime_seconds / number_of_feature_windows) * 1000.0
        if number_of_feature_windows > 0
        else None
    )

    thresholded_exact_match_correct_examples = int(
        round((float(thresholded_metrics["exact"]) / 100.0) * number_of_raw_evaluation_examples)
    )
    thresholded_exact_match_correct_examples = max(1, thresholded_exact_match_correct_examples)

    result = {
        "maximum_sequence_length": int(maximum_sequence_length),
        "effective_document_stride": int(effective_document_stride),
        "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
        "number_of_feature_windows": number_of_feature_windows,
        "number_of_inference_tokens": number_of_inference_tokens,
        "inference_runtime_seconds": inference_runtime_seconds,
        "average_latency_per_raw_example_milliseconds": average_latency_per_raw_example_milliseconds,
        "average_latency_per_feature_window_milliseconds": average_latency_per_feature_window_milliseconds,
        "inference_energy_joules": inference_energy_joules,
        "joules_per_inference_example": (
            inference_energy_joules / number_of_raw_evaluation_examples
            if number_of_raw_evaluation_examples > 0
            else None
        ),
        "joules_per_feature_window": (
            inference_energy_joules / number_of_feature_windows
            if number_of_feature_windows > 0
            else None
        ),
        "joules_per_inference_token": (
            inference_energy_joules / number_of_inference_tokens
            if number_of_inference_tokens > 0
            else None
        ),
        "joules_per_exact_match_correct_example": (
            inference_energy_joules / thresholded_exact_match_correct_examples
            if thresholded_exact_match_correct_examples > 0
            else None
        ),
        "no_answer_probability_threshold": float(no_answer_probability_threshold),
        "metrics_raw": raw_metrics,
        "metrics_thresholded": thresholded_metrics,
    }

    # Persist full per-length result for debugging and reproducibility
    write_json_file(result, run_directory / "result.json")

    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields={
            "phase": "inference",
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
            "number_of_feature_windows": number_of_feature_windows,
            "number_of_inference_tokens": number_of_inference_tokens,
            "joules_per_inference_example": result["joules_per_inference_example"],
            "joules_per_feature_window": result["joules_per_feature_window"],
            "joules_per_inference_token": result["joules_per_inference_token"],
        },
    )

    return result